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
Shared base class for container-based backends (Docker, Podman).

This module provides common functionality for backends that use container runtimes,
reducing code duplication while allowing each backend to implement runtime-specific
behaviors.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
import random
import shutil
import string
import uuid

from kubeflow.trainer.backends.base import ExecutionBackend
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


@dataclass
class _Node:
    name: str
    container_id: str
    status: str = constants.TRAINJOB_CREATED


@dataclass
class _Job:
    name: str
    created: datetime
    runtime: types.Runtime
    network_id: str  # Network or pod ID depending on backend
    nodes: list[_Node]
    workdir_host: str


class ContainerBackend(ExecutionBackend):
    """
    Abstract base class for container-based execution backends.

    Subclasses must implement:
    - _get_client(): Return the container client (Docker/Podman)
    - _create_network(job_name): Create network/pod for multi-node jobs
    - _delete_network(network_id): Clean up network/pod
    - _get_master_addr(job_name, rank): Return master address for torchrun
    - _get_master_port(rank): Return master port for torchrun
    - _create_and_start_container(): Create and start a container
    - _get_container(): Get container object by ID
    - _container_logs(): Get logs from a container
    - _stop_container(): Stop a container
    - _remove_container(): Remove a container
    - _pull_image(): Pull an image
    - _image_exists(): Check if image exists locally
    - _run_oneoff_container(): Run a one-off container and return logs
    """

    def __init__(self, label_prefix: str = "trainer.kubeflow.org"):
        self._jobs: dict[str, _Job] = {}
        self.label_prefix = label_prefix

    @abc.abstractmethod
    def _get_client(self):
        """Return the container client instance."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_network(self, job_name: str) -> str:
        """Create a network or pod for the job. Return network/pod ID."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _delete_network(self, network_id: str):
        """Delete the network or pod."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_master_addr(self, job_name: str, rank: int) -> str:
        """Get the master address for torchrun rendezvous."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_master_port(self, rank: int) -> int:
        """Get the master port for torchrun rendezvous."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _create_and_start_container(
        self,
        image: str,
        command: list[str],
        name: str,
        network_id: str,
        environment: dict[str, str],
        labels: dict[str, str],
        volumes: dict[str, dict[str, str]],
        working_dir: str,
    ) -> str:
        """Create and start a container. Return container ID."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_container(self, container_id: str):
        """Get container object by ID."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _stop_container(self, container_id: str, timeout: int = 10):
        """Stop a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _remove_container(self, container_id: str, force: bool = True):
        """Remove a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _pull_image(self, image: str):
        """Pull an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _image_exists(self, image: str) -> bool:
        """Check if image exists locally."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _run_oneoff_container(self, image: str, command: list[str]) -> str:
        """Run a one-off container and return its logs."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_pull_policy(self) -> str:
        """Get the pull policy for images."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_workdir_base(self) -> str | None:
        """Get the base directory for working directories."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_auto_remove(self) -> bool:
        """Get whether to auto-remove containers."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _resolve_image(self, runtime: types.Runtime) -> str:
        """Resolve the container image for a runtime."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _get_runtimes_dir(self) -> Path:
        """Get the path to the runtimes directory."""
        raise NotImplementedError()

    # ---- Shared implementation ----

    def _resolve_image_from_runtime(self, runtime: types.Runtime, config_image: str | None) -> str:
        """
        Shared logic for resolving the container image for a runtime.

        If config_image is provided, use it. Otherwise, load from runtime YAML.
        """
        if config_image:
            return config_image

        import yaml

        runtimes_dir = self._get_runtimes_dir()
        for f in sorted(runtimes_dir.glob("*.yaml")):
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

        backend_name = self.__class__.__name__.replace("Local", "").replace("Backend", "")
        raise ValueError(
            f"No image specified for runtime '{runtime.name}'. Provide {backend_name}BackendConfig.image or "
            f"add an 'image' field to its YAML in {runtimes_dir}."
        )

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

        logs = self._run_oneoff_container(image=image, command=command)
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

        # Create per-job working directory on host
        workdir = self._create_workdir(job_name)
        _ = self._write_training_script(workdir, trainer)

        # Resolve image and pull if needed
        image = self._resolve_image(runtime)
        self._maybe_pull_image(image)

        # Build base environment
        env = self._build_environment(trainer)

        # Construct pre-run command to install packages
        pre_install_cmd = self._build_pip_install_cmd(trainer)

        # Create network/pod for multi-node communication
        num_nodes = trainer.num_nodes or runtime.trainer.num_nodes or 1
        network_id = self._create_network(job_name)

        # Create N containers (one per node)
        containers: list[_Node] = []

        for rank in range(num_nodes):
            container_name = f"{job_name}-node-{rank}"

            # Get master address and port for torchrun
            master_addr = self._get_master_addr(job_name, rank)
            master_port = self._get_master_port(rank)

            # Prefer torchrun; fall back to python if torchrun is unavailable
            # For worker nodes, wait for master to be reachable before starting torchrun
            wait_for_master = ""
            if rank > 0:
                wait_for_master = (
                    f"echo 'Waiting for master node {master_addr}:{master_port}...'; "
                    f"for i in {{1..60}}; do "
                    f"  if timeout 1 bash -c 'cat < /dev/null > /dev/tcp/{master_addr}/{master_port}' 2>/dev/null; then "
                    f"    echo 'Master node is reachable'; break; "
                    f"  fi; "
                    f"  if [ $i -eq 60 ]; then echo 'Timeout waiting for master node'; exit 1; fi; "
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
                f"  {self.cfgd_workspace_path()}train.py; "
                "else "
                f"  python {self.cfgd_workspace_path()}train.py; "
                "fi"
            )

            full_cmd = ["bash", "-lc", entry_cmd]

            labels = {
                f"{self.label_prefix}/trainjob-name": job_name,
                f"{self.label_prefix}/step": f"node-{rank}",
            }

            volumes = {
                workdir: {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            }

            container_id = self._create_and_start_container(
                image=image,
                command=full_cmd,
                name=container_name,
                network_id=network_id,
                environment=env,
                labels=labels,
                volumes=volumes,
                working_dir="/workspace",
            )

            containers.append(_Node(name=container_name, container_id=container_id))

        self._jobs[job_name] = _Job(
            name=job_name,
            created=datetime.now(),
            runtime=runtime,
            network_id=network_id,
            nodes=containers,
            workdir_host=workdir,
        )

        return job_name

    def list_jobs(self, runtime: types.Runtime | None = None) -> list[types.TrainJob]:
        result: list[types.TrainJob] = []
        for job in self._jobs.values():
            if runtime and job.runtime.name != runtime.name:
                continue
            steps = []
            for node in job.nodes:
                steps.append(
                    types.Step(
                        name=node.name.split(f"{job.name}-")[-1],
                        pod_name=node.name,
                        status=self._container_status(node.container_id),
                    )
                )
            result.append(
                types.TrainJob(
                    name=job.name,
                    creation_timestamp=job.created,
                    runtime=job.runtime,
                    steps=steps,
                    num_nodes=len(job.nodes),
                    status=self._aggregate_status(job),
                )
            )
        return result

    def get_job(self, name: str) -> types.TrainJob:
        job = self._jobs.get(name)
        if not job:
            raise ValueError(f"No TrainJob with name {name}")
        # Refresh container statuses on demand
        steps: list[types.Step] = []
        for node in job.nodes:
            status = self._container_status(node.container_id)
            steps.append(
                types.Step(
                    name=node.name.split(f"{job.name}-")[-1],
                    pod_name=node.name,
                    status=status,
                )
            )
        return types.TrainJob(
            name=job.name,
            creation_timestamp=job.created,
            runtime=job.runtime,
            steps=steps,
            num_nodes=len(job.nodes),
            status=self._aggregate_status(job),
        )

    def get_job_logs(
        self,
        name: str,
        follow: bool = False,
        step: str = constants.NODE + "-0",
    ) -> Iterator[str]:
        job = self._jobs.get(name)
        if not job:
            raise ValueError(f"No TrainJob with name {name}")

        want_all = step == constants.NODE + "-0"
        for node in job.nodes:
            node_step = node.name.split(f"{job.name}-")[-1]
            if not want_all and node_step != step:
                continue
            try:
                yield from self._container_logs(node.container_id, follow)
            except Exception as e:
                logger.warning(f"Failed to get logs for {node.name}: {e}")
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
        job = self._jobs.get(name)
        if not job:
            raise ValueError(f"No TrainJob with name {name}")

        # Stop containers and remove
        from contextlib import suppress

        for node in job.nodes:
            with suppress(Exception):
                self._stop_container(node.container_id, timeout=10)
            with suppress(Exception):
                self._remove_container(node.container_id, force=True)

        # Remove network/pod (best-effort)
        with suppress(Exception):
            self._delete_network(job.network_id)

        # Remove working directory if configured
        if self._get_auto_remove() and os.path.isdir(job.workdir_host):
            shutil.rmtree(job.workdir_host, ignore_errors=True)

        del self._jobs[name]

    # ---- Helper methods ----

    def _create_workdir(self, job_name: str) -> str:
        """Create per-job working directory on host."""
        workdir_base = self._get_workdir_base()
        if workdir_base:
            base = Path(workdir_base)
            base.mkdir(parents=True, exist_ok=True)
            workdir = str((base / f"{job_name}").resolve())
            os.makedirs(workdir, exist_ok=True)
        else:
            backend_name = self.__class__.__name__.lower().replace("local", "").replace("backend", "")
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
        env = {}
        # Subclasses should add their own env vars here
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
        policy = (self._get_pull_policy() or "IfNotPresent").lower()
        try:
            if policy == "never":
                if not self._image_exists(image):
                    raise RuntimeError(f"Image '{image}' not found locally and pull policy is Never")
                return
            if policy == "always":
                logger.debug(f"Pulling image (Always): {image}")
                self._pull_image(image)
                return
            # IfNotPresent
            if not self._image_exists(image):
                logger.debug(f"Pulling image (IfNotPresent): {image}")
                self._pull_image(image)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure image '{image}': {e}") from e

    def _container_status(self, container_id: str) -> str:
        """Get the status of a container."""
        try:
            container = self._get_container(container_id)
            status = container.status
            if status == "running":
                return constants.TRAINJOB_RUNNING
            if status == "created":
                return constants.TRAINJOB_CREATED
            if status == "exited":
                # Exit code 0 -> complete, else failed
                inspect = container.attrs if hasattr(container, 'attrs') else container.inspect()
                code = inspect.get("State", {}).get("ExitCode")
                return constants.TRAINJOB_COMPLETE if code == 0 else constants.TRAINJOB_FAILED
        except Exception:
            return constants.UNKNOWN
        return constants.UNKNOWN

    def _aggregate_status(self, job: _Job) -> str:
        """Aggregate status from all containers in a job."""
        statuses = [self._container_status(n.container_id) for n in job.nodes]
        if constants.TRAINJOB_FAILED in statuses:
            return constants.TRAINJOB_FAILED
        if constants.TRAINJOB_RUNNING in statuses:
            return constants.TRAINJOB_RUNNING
        if all(s == constants.TRAINJOB_COMPLETE for s in statuses if s != constants.UNKNOWN):
            return constants.TRAINJOB_COMPLETE
        if any(s == constants.TRAINJOB_CREATED for s in statuses):
            return constants.TRAINJOB_CREATED
        return constants.UNKNOWN

    def cfgd_workspace_path(self) -> str:
        """Location inside the container where the host workdir is mounted."""
        return "/workspace/"
