# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ContainerBackend
----------------

Unified local execution backend for `CustomTrainer` jobs using containers.

This backend automatically detects and uses either Docker or Podman.
It provides a single interface regardless of the underlying container runtime.

Key behaviors:
- Auto-detection: Tries Docker first, then Podman. Can be overridden via config.
- Multi-node jobs: one container per node connected via a per-job network.
- Entry script generation: we serialize the user's training function to a small
  Python file and invoke it inside the container using `torchrun` (preferred) or
  `python` as a fallback.
- Runtimes: we use `config/local_runtimes` to define runtime images and
  characteristics (e.g., torch). Defaults to `torch-distributed` if no runtime
  is provided.
- Image pulling: controlled via `pull_policy` and performed automatically if
  needed.
- Logs and lifecycle: streaming logs and deletion semantics similar to the
  Docker/Podman backends, but with automatic runtime detection.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import shutil
import string
import uuid

from kubeflow.trainer.backends.base import ExecutionBackend
from kubeflow.trainer.backends.container.client_adapter import (
    ContainerClientAdapter,
    DockerClientAdapter,
    PodmanClientAdapter,
)
from kubeflow.trainer.backends.container.runtime_loader import (
    LOCAL_RUNTIMES_DIR,
    get_local_runtime,
    list_local_runtimes,
)
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class ContainerBackend(ExecutionBackend):
    """
    Unified container backend that auto-detects Docker or Podman.

    This backend uses the adapter pattern to abstract away differences between
    Docker and Podman, providing a single consistent interface.
    """

    def __init__(self, cfg: ContainerBackendConfig):
        self.cfg = cfg
        self.label_prefix = "trainer.kubeflow.org"

        # Initialize the container client adapter
        self._adapter = self._create_adapter()

    def _create_adapter(self) -> ContainerClientAdapter:
        """
        Create the appropriate container client adapter.

        Tries Docker first, then Podman if Docker fails, unless a specific
        runtime is requested in the config.

        Raises RuntimeError if neither Docker nor Podman are available.
        """
        if self.cfg.runtime:
            # User specified a runtime explicitly
            if self.cfg.runtime == "docker":
                adapter = DockerClientAdapter(self.cfg.container_host)
                adapter.ping()
                logger.info("Using Docker as container runtime")
                return adapter
            elif self.cfg.runtime == "podman":
                adapter = PodmanClientAdapter(self.cfg.container_host)
                adapter.ping()
                logger.info("Using Podman as container runtime")
                return adapter
        else:
            # Auto-detect: try Docker first, then Podman
            try:
                adapter = DockerClientAdapter(self.cfg.container_host)
                adapter.ping()
                logger.info("Using Docker as container runtime")
                return adapter
            except Exception as docker_error:
                logger.debug(f"Docker initialization failed: {docker_error}")
                try:
                    adapter = PodmanClientAdapter(self.cfg.container_host)
                    adapter.ping()
                    logger.info("Using Podman as container runtime")
                    return adapter
                except Exception as podman_error:
                    logger.debug(f"Podman initialization failed: {podman_error}")
                    raise RuntimeError(
                        "Neither Docker nor Podman is available. "
                        "Please install Docker or Podman, or use LocalProcessBackendConfig instead."
                    ) from podman_error

    @property
    def _runtime_type(self) -> str:
        """Get the runtime type for debugging/logging."""
        return self._adapter._runtime_type

    # ---- Runtime APIs ----
    def list_runtimes(self) -> list[types.Runtime]:
        return list_local_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        return get_local_runtime(name)

    def get_runtime_packages(self, runtime: types.Runtime):
        """
        Spawn a short-lived container to report Python version, pip list, and nvidia-smi.
        """
        image = self._resolve_image(runtime)
        self._maybe_pull_image(image)

        command = [
            "bash",
            "-lc",
            "python -c \"import sys; print(f'Python: {sys.version}')\" && "
            "(pip list || echo 'pip not found') && "
            "(nvidia-smi || echo 'nvidia-smi not found')",
        ]

        logs = self._adapter.run_oneoff_container(image=image, command=command)
        print(logs)

    def train(
        self,
        runtime: types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainer | types.BuiltinTrainer | None = None,
    ) -> str:
        if runtime is None:
            runtime = self.get_runtime("torch-distributed")

        if not isinstance(trainer, types.CustomTrainer):
            raise ValueError(f"{self.__class__.__name__} supports only CustomTrainer in v1")

        # Generate job name
        job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]
        logger.info(f"Starting training job: {job_name}")

        try:
            # Create per-job working directory on host
            workdir = self._create_workdir(job_name)
            logger.debug(f"Created working directory: {workdir}")

            _ = self._write_training_script(workdir, trainer)
            logger.debug(f"Wrote training script to {workdir}/train.py")

            # Resolve image and pull if needed
            image = self._resolve_image(runtime)
            logger.debug(f"Using image: {image}")

            self._maybe_pull_image(image)
            logger.debug(f"Image ready: {image}")

            # Build base environment
            env = self._build_environment(trainer)

            # Construct pre-run command to install packages
            pre_install_cmd = self._build_pip_install_cmd(trainer)

            # Create network for multi-node communication
            num_nodes = trainer.num_nodes or runtime.trainer.num_nodes or 1
            logger.debug(f"Creating network for {num_nodes} nodes")

            network_id = self._adapter.create_network(
                name=f"{job_name}-net",
                labels={
                    f"{self.label_prefix}/trainjob-name": job_name,
                    f"{self.label_prefix}/runtime-name": runtime.name,
                    f"{self.label_prefix}/workdir": workdir,
                    f"{self.label_prefix}/created": datetime.now().isoformat(),
                },
            )
            logger.info(f"Created network: {network_id}")

            # Create N containers (one per node)
            container_ids: list[str] = []
            master_container_id = None
            master_ip = None

            for rank in range(num_nodes):
                container_name = f"{job_name}-node-{rank}"

                # Get master address and port for torchrun
                master_port = 29500

                # For Podman: use IP address to avoid DNS timing issues
                # For Docker: use hostname (DNS is reliable)
                if rank == 0:
                    # Master node - will be created first
                    master_addr = f"{job_name}-node-0"
                else:
                    # Worker nodes - determine master address based on runtime
                    if self._runtime_type == "podman" and master_ip:
                        master_addr = master_ip
                        logger.debug(f"Using master IP address for Podman: {master_ip}")
                    else:
                        master_addr = f"{job_name}-node-0"
                        logger.debug(f"Using master hostname: {master_addr}")

                # Prefer torchrun; fall back to python if torchrun is unavailable
                # For worker nodes, wait for master to be reachable before starting torchrun
                wait_for_master = ""
                if rank > 0:
                    wait_for_master = (
                        f"echo 'Waiting for master node {master_addr}:{master_port}...'; "
                        f"for i in {{1..60}}; do "
                        f"  if timeout 1 bash -c 'cat < /dev/null > "
                        f"/dev/tcp/{master_addr}/{master_port}' 2>/dev/null; then "
                        f"    echo 'Master node is reachable'; break; "
                        f"  fi; "
                        f"  if [ $i -eq 60 ]; then "
                        f"echo 'Timeout waiting for master node'; exit 1; fi; "
                        f"  sleep 2; "
                        f"done; "
                    )

                entry_cmd = (
                    f"{pre_install_cmd}"
                    f"{wait_for_master}"
                    "if command -v torchrun >/dev/null 2>&1; then "
                    f"  torchrun --nproc_per_node=1 --nnodes={num_nodes} "
                    f"  --node-rank={rank} --rdzv-backend=c10d "
                    f"  --rdzv-endpoint={master_addr}:{master_port} "
                    f"  /workspace/train.py; "
                    "else "
                    f"  python /workspace/train.py; "
                    "fi"
                )

                full_cmd = ["bash", "-lc", entry_cmd]

                labels = {
                    f"{self.label_prefix}/trainjob-name": job_name,
                    f"{self.label_prefix}/step": f"node-{rank}",
                    f"{self.label_prefix}/network-id": network_id,
                    f"{self.label_prefix}/num-nodes": str(num_nodes),
                }

                volumes = {
                    workdir: {
                        "bind": "/workspace",
                        "mode": "rw",
                    }
                }

                logger.debug(f"Creating container {rank}/{num_nodes}: {container_name}")

                container_id = self._adapter.create_and_start_container(
                    image=image,
                    command=full_cmd,
                    name=container_name,
                    network_id=network_id,
                    environment=env,
                    labels=labels,
                    volumes=volumes,
                    working_dir="/workspace",
                )

                logger.info(f"Started container {container_name} (ID: {container_id[:12]})")
                container_ids.append(container_id)

                # If this is the master node and we're using Podman, get its IP address
                if rank == 0:
                    master_container_id = container_id
                    if self._runtime_type == "podman":
                        # Get master IP for worker nodes to use
                        master_ip = self._adapter.get_container_ip(master_container_id, network_id)
                        if master_ip:
                            logger.info(f"Master node IP address: {master_ip}")
                        else:
                            logger.warning(
                                "Could not retrieve master IP address. "
                                "Worker nodes will fall back to DNS resolution."
                            )

            logger.info(
                f"Training job {job_name} created successfully with {len(container_ids)} container(s)"
            )
            return job_name

        except Exception as e:
            # Clean up on failure
            logger.error(f"Failed to create training job {job_name}: {e}")
            logger.exception("Full traceback:")

            # Try to clean up any resources that were created
            from contextlib import suppress

            try:
                # Stop and remove any containers that were created
                if "container_ids" in locals():
                    for container_id in container_ids:
                        with suppress(Exception):
                            self._adapter.stop_container(container_id, timeout=5)
                            self._adapter.remove_container(container_id, force=True)

                # Remove network if it was created
                if "network_id" in locals():
                    with suppress(Exception):
                        self._adapter.delete_network(network_id)

                # Remove working directory if it was created
                if "workdir" in locals() and os.path.isdir(workdir):
                    shutil.rmtree(workdir, ignore_errors=True)

            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {cleanup_error}")

            # Re-raise the original exception
            raise

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        """List all training jobs by querying container runtime."""
        # Get all containers with our label prefix
        filters = {"label": [f"{self.label_prefix}/trainjob-name"]}
        containers = self._adapter.list_containers(filters=filters)

        # Group containers by job name
        jobs_map: dict[str, list[dict]] = {}
        for container in containers:
            job_name = container["labels"].get(f"{self.label_prefix}/trainjob-name")
            if job_name:
                if job_name not in jobs_map:
                    jobs_map[job_name] = []
                jobs_map[job_name].append(container)

        result: list[types.TrainJob] = []
        for job_name, job_containers in jobs_map.items():
            # Get metadata from first container's network
            if not job_containers:
                continue

            network_id = job_containers[0]["labels"].get(f"{self.label_prefix}/network-id")
            if not network_id:
                continue

            network_info = self._adapter.get_network(network_id)
            if not network_info:
                continue

            network_labels = network_info.get("labels", {})
            runtime_name = network_labels.get(f"{self.label_prefix}/runtime-name")

            # Filter by runtime if specified
            if runtime and runtime_name != runtime.name:
                continue

            # Get runtime object
            try:
                job_runtime = self.get_runtime(runtime_name) if runtime_name else None
            except Exception:
                job_runtime = None

            if not job_runtime:
                continue

            # Parse creation timestamp
            created_str = network_labels.get(f"{self.label_prefix}/created", "")
            try:
                from dateutil import parser
                creation_timestamp = parser.isoparse(created_str)
            except Exception:
                creation_timestamp = datetime.now()

            # Build steps from containers
            steps = []
            for container in sorted(job_containers, key=lambda c: c["name"]):
                step_name = container["labels"].get(f"{self.label_prefix}/step", "")
                steps.append(
                    types.Step(
                        name=step_name,
                        pod_name=container["name"],
                        status=self._container_status(container["id"]),
                    )
                )

            # Get num_nodes from labels
            num_nodes = int(job_containers[0]["labels"].get(f"{self.label_prefix}/num-nodes", len(job_containers)))

            result.append(
                types.TrainJob(
                    name=job_name,
                    creation_timestamp=creation_timestamp,
                    runtime=job_runtime,
                    steps=steps,
                    num_nodes=num_nodes,
                    status=self._aggregate_status_from_containers(job_containers),
                )
            )

        return result

    def get_job(self, name: str) -> types.TrainJob:
        """Get a specific training job by querying container runtime."""
        # Find containers for this job
        filters = {"label": [f"{self.label_prefix}/trainjob-name={name}"]}
        containers = self._adapter.list_containers(filters=filters)

        if not containers:
            raise ValueError(f"No TrainJob with name {name}")

        # Get metadata from network
        network_id = containers[0]["labels"].get(f"{self.label_prefix}/network-id")
        if not network_id:
            raise ValueError(f"TrainJob {name} is missing network metadata")

        network_info = self._adapter.get_network(network_id)
        if not network_info:
            raise ValueError(f"TrainJob {name} network not found")

        network_labels = network_info.get("labels", {})
        runtime_name = network_labels.get(f"{self.label_prefix}/runtime-name")

        # Get runtime object
        try:
            job_runtime = self.get_runtime(runtime_name) if runtime_name else None
        except Exception:
            raise ValueError(f"Runtime {runtime_name} not found for job {name}")

        if not job_runtime:
            raise ValueError(f"Runtime {runtime_name} not found for job {name}")

        # Parse creation timestamp
        created_str = network_labels.get(f"{self.label_prefix}/created", "")
        try:
            from dateutil import parser
            creation_timestamp = parser.isoparse(created_str)
        except Exception:
            creation_timestamp = datetime.now()

        # Build steps from containers
        steps = []
        for container in sorted(containers, key=lambda c: c["name"]):
            step_name = container["labels"].get(f"{self.label_prefix}/step", "")
            steps.append(
                types.Step(
                    name=step_name,
                    pod_name=container["name"],
                    status=self._container_status(container["id"]),
                )
            )

        # Get num_nodes from labels
        num_nodes = int(containers[0]["labels"].get(f"{self.label_prefix}/num-nodes", len(containers)))

        return types.TrainJob(
            name=name,
            creation_timestamp=creation_timestamp,
            runtime=job_runtime,
            steps=steps,
            num_nodes=num_nodes,
            status=self._aggregate_status_from_containers(containers),
        )

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        """Get logs for a training job by querying container runtime."""
        # Find containers for this job
        filters = {"label": [f"{self.label_prefix}/trainjob-name={name}"]}
        containers = self._adapter.list_containers(filters=filters)

        if not containers:
            raise ValueError(f"No TrainJob with name {name}")

        want_all = step == constants.NODE + "-0"
        for container in sorted(containers, key=lambda c: c["name"]):
            container_step = container["labels"].get(f"{self.label_prefix}/step", "")
            if not want_all and container_step != step:
                continue
            try:
                yield from self._adapter.container_logs(container["id"], follow)
            except Exception as e:
                logger.warning(f"Failed to get logs for {container['name']}: {e}")
                yield f"Error getting logs: {e}\n"

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        import time

        end = time.time() + timeout
        while time.time() < end:
            tj = self.get_job(name)
            logger.debug(f"TrainJob {name}, status {tj.status}")
            if tj.status in status:
                return tj
            if constants.TRAINJOB_FAILED not in status and tj.status == constants.TRAINJOB_FAILED:
                raise RuntimeError(f"TrainJob {name} is Failed")
            time.sleep(polling_interval)
        raise TimeoutError(f"Timeout waiting for TrainJob {name} to reach status: {status}")

    def delete_job(self, name: str):
        """Delete a training job by querying container runtime."""
        # Find containers for this job
        filters = {"label": [f"{self.label_prefix}/trainjob-name={name}"]}
        containers = self._adapter.list_containers(filters=filters)

        if not containers:
            raise ValueError(f"No TrainJob with name {name}")

        # Get network_id and workdir from labels
        network_id = containers[0]["labels"].get(f"{self.label_prefix}/network-id")

        # Get workdir from network labels
        workdir_host = None
        if network_id:
            network_info = self._adapter.get_network(network_id)
            if network_info:
                network_labels = network_info.get("labels", {})
                workdir_host = network_labels.get(f"{self.label_prefix}/workdir")

        # Stop containers and remove
        from contextlib import suppress

        for container in containers:
            with suppress(Exception):
                self._adapter.stop_container(container["id"], timeout=10)
            with suppress(Exception):
                self._adapter.remove_container(container["id"], force=True)

        # Remove network (best-effort)
        if network_id:
            with suppress(Exception):
                self._adapter.delete_network(network_id)

        # Remove working directory if configured
        if self.cfg.auto_remove and workdir_host and os.path.isdir(workdir_host):
            shutil.rmtree(workdir_host, ignore_errors=True)

    # Helper methods

    def _create_workdir(self, job_name: str) -> str:
        """Create per-job working directory on host."""
        workdir_base = self.cfg.workdir_base
        if workdir_base:
            base = Path(workdir_base)
            base.mkdir(parents=True, exist_ok=True)
            workdir = str((base / f"{job_name}").resolve())
            os.makedirs(workdir, exist_ok=True)
        else:
            backend_name = (
                self.__class__.__name__.lower().replace("local", "").replace("backend", "")
            )
            home_base = Path.home() / ".kubeflow_trainer" / f"local{backend_name}"
            home_base.mkdir(parents=True, exist_ok=True)
            workdir = str((home_base / f"{job_name}").resolve())
            os.makedirs(workdir, exist_ok=True)
        return workdir

    def _write_training_script(self, workdir: str, trainer: types.CustomTrainer) -> Path:
        """Write the training script to the working directory."""
        script_path = Path(workdir) / "train.py"
        import inspect
        import textwrap

        code = inspect.getsource(trainer.func)
        code = textwrap.dedent(code)
        if trainer.func_args is None:
            code += f"\n{trainer.func.__name__}()\n"
        else:
            code += f"\n{trainer.func.__name__}({trainer.func_args})\n"
        script_path.write_text(code)
        return script_path

    def _build_environment(self, trainer: types.CustomTrainer) -> dict[str, str]:
        """Build environment variables for containers."""
        env = dict(self.cfg.env or {})
        if trainer.env:
            env.update(trainer.env)
        return env

    def _build_pip_install_cmd(self, trainer: types.CustomTrainer) -> str:
        """Build pip install command for packages."""
        pkgs = trainer.packages_to_install or []
        if not pkgs:
            return ""

        index_urls = trainer.pip_index_urls or list(constants.DEFAULT_PIP_INDEX_URLS)
        main_idx = index_urls[0]
        extras = " ".join(f"--extra-index-url {u}" for u in index_urls[1:])
        quoted = " ".join(f'"{p}"' for p in pkgs)
        return (
            "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install --no-warn-script-location "
            f"--index-url {main_idx} {extras} {quoted} && "
        )

    def _maybe_pull_image(self, image: str):
        """Pull image based on pull policy."""
        policy = (self.cfg.pull_policy or "IfNotPresent").lower()
        try:
            if policy == "never":
                if not self._adapter.image_exists(image):
                    raise RuntimeError(
                        f"Image '{image}' not found locally and pull policy is Never"
                    )
                return
            if policy == "always":
                logger.debug(f"Pulling image (Always): {image}")
                self._adapter.pull_image(image)
                return
            # IfNotPresent
            if not self._adapter.image_exists(image):
                logger.debug(f"Pulling image (IfNotPresent): {image}")
                self._adapter.pull_image(image)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure image '{image}': {e}") from e

    def _container_status(self, container_id: str) -> str:
        """Get the status of a container."""
        try:
            status, exit_code = self._adapter.container_status(container_id)
            if status == "running":
                return constants.TRAINJOB_RUNNING
            if status == "created":
                return constants.TRAINJOB_CREATED
            if status == "exited":
                # Exit code 0 -> complete, else failed
                return constants.TRAINJOB_COMPLETE if exit_code == 0 else constants.TRAINJOB_FAILED
        except Exception:
            return constants.UNKNOWN
        return constants.UNKNOWN

    def _aggregate_status_from_containers(self, containers: list[dict]) -> str:
        """Aggregate status from container info dicts."""
        statuses = [self._container_status(c["id"]) for c in containers]
        if constants.TRAINJOB_FAILED in statuses:
            return constants.TRAINJOB_FAILED
        if constants.TRAINJOB_RUNNING in statuses:
            return constants.TRAINJOB_RUNNING
        if all(s == constants.TRAINJOB_COMPLETE for s in statuses if s != constants.UNKNOWN):
            return constants.TRAINJOB_COMPLETE
        if any(s == constants.TRAINJOB_CREATED for s in statuses):
            return constants.TRAINJOB_CREATED
        return constants.UNKNOWN

    def _resolve_image(self, runtime: types.Runtime) -> str:
        """Resolve the container image for a runtime."""
        if self.cfg.image:
            return self.cfg.image

        import yaml

        for f in sorted(LOCAL_RUNTIMES_DIR.glob("*.yaml")):
            try:
                data = yaml.safe_load(Path(f).read_text())
                if (
                    data.get("kind") in {"ClusterTrainingRuntime", "TrainingRuntime"}
                    and data.get("metadata", {}).get("name") == runtime.name
                ):
                    replicated = (
                        data.get("spec", {})
                        .get("template", {})
                        .get("spec", {})
                        .get("replicatedJobs", [])
                    )
                    node_jobs = [j for j in replicated if j.get("name") == "node"]
                    if node_jobs:
                        node_spec = (
                            node_jobs[0]
                            .get("template", {})
                            .get("spec", {})
                            .get("template", {})
                            .get("spec", {})
                        )
                        containers = node_spec.get("containers", [])
                        if containers and containers[0].get("image"):
                            return str(containers[0]["image"])
            except Exception:
                continue

        raise ValueError(
            f"No image specified for runtime '{runtime.name}'. "
            f"Provide ContainerBackendConfig.image or add an 'image' field "
            f"to its YAML in {LOCAL_RUNTIMES_DIR}."
        )
