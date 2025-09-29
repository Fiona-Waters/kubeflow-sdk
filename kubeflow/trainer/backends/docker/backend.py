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
DockerBackend
-------------

Local execution backend for `CustomTrainer` jobs using Docker containers.

Key behaviors:
- Multi-node jobs: one container per node connected via a per-job Docker network.
- Entry script generation: we serialize the user's training function to a small
  Python file and invoke it inside the container using `torchrun` (preferred) or
  `python` as a fallback.
- Runtimes: we use `config/local_runtimes` to define runtime images and
  characteristics (e.g., torch). Defaults to `torch-distributed` if no runtime
  is provided.
- Image pulling: controlled via `pull_policy` and performed automatically if
  needed.
- Logs and lifecycle: streaming logs and deletion semantics similar to the
  Kubernetes backend, but tailored to Docker.
"""

from __future__ import annotations

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

try:
    import docker  # type: ignore
    from docker.models.containers import Container  # type: ignore
except Exception:  # pragma: no cover - optional dependency, validated at runtime
    docker = None  # type: ignore
    Container = object  # type: ignore

from kubeflow.trainer.backends.base import ExecutionBackend
from kubeflow.trainer.backends.docker.runtime_loader import (
    get_local_runtime,
    list_local_runtimes,
)
from kubeflow.trainer.backends.docker.types import LocalDockerBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


DOCKER_LABEL_PREFIX = "trainer.kubeflow.ai"


@dataclass
class _Node:
    name: str
    container: Container
    status: str = constants.TRAINJOB_CREATED


@dataclass
class _Job:
    name: str
    created: datetime
    runtime: types.Runtime
    network_name: str
    nodes: list[_Node]
    workdir_host: str


class LocalDockerBackend(ExecutionBackend):
    def __init__(self, cfg: LocalDockerBackendConfig):
        if docker is None:
            raise ImportError(
                "The 'docker' Python package is not installed. Install with extras: "
                "pip install kubeflow[docker]"
            )

        # initialize docker client (env-based resolution by default)
        if cfg.docker_host:
            self.client = docker.DockerClient(base_url=cfg.docker_host)
        else:
            self.client = docker.from_env()

        self.cfg = cfg
        self._jobs: dict[str, _Job] = {}

    # ---- Runtime APIs ----
    def list_runtimes(self) -> list[types.Runtime]:
        return list_local_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        return get_local_runtime(name)

    def get_runtime_packages(self, runtime: types.Runtime):
        """
        Spawn a short-lived container to report Python version, pip list, and nvidia-smi.
        We follow the Kubernetes backend semantics: create a one-off job and print outputs.
        """
        image = self._resolve_image(runtime)
        self._maybe_pull_image(image)

        command = (
            "bash",
            "-lc",
            "python -c \"import sys; print(f'Python: {sys.version}')\" && "
            "(pip list || echo 'pip not found') && "
            "(nvidia-smi || echo 'nvidia-smi not found')",
        )

        logs = self._run_oneoff_container(image=image, command=command)
        print(logs)

    # ---- Train/Jobs APIs ----
    def train(
        self,
        runtime: types.Runtime | None = None,
        initializer: types.Initializer | None = None,
        trainer: types.CustomTrainer | types.BuiltinTrainer | None = None,
    ) -> str:
        if runtime is None:
            runtime = self.get_runtime("torch-distributed")

        if not isinstance(trainer, types.CustomTrainer):
            raise ValueError("DockerBackend supports only CustomTrainer in v1")

        # Generate job name
        job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]

        # Create per-job working directory on host where we'll place the generated script
        # Prefer a location under the user's home directory to ensure Docker Desktop file
        # sharing includes it by default on macOS/Windows.
        if self.cfg.workdir_base:
            base = Path(self.cfg.workdir_base)
            base.mkdir(parents=True, exist_ok=True)
            workdir = str((base / f"{job_name}").resolve())
            os.makedirs(workdir, exist_ok=True)
        else:
            home_base = Path.home() / ".kubeflow_trainer" / "localdocker"
            home_base.mkdir(parents=True, exist_ok=True)
            workdir = str((home_base / f"{job_name}").resolve())
            os.makedirs(workdir, exist_ok=True)
        _ = self._write_training_script(workdir, trainer)

        # Provision per-job network
        network_name = self._ensure_network(job_name)

        # Resolve image and pull if needed
        image = self._resolve_image(runtime)
        self._maybe_pull_image(image)

        # Build base environment
        env = dict(self.cfg.env or {})
        if trainer.env:
            env.update(trainer.env)

        # Construct pre-run command to install packages, if any, inside the container.
        pre_install_cmd = ""
        pkgs = trainer.packages_to_install or []
        if pkgs:
            # Respect pip_index_urls ordering (index-url then extra-index-urls)
            index_urls = trainer.pip_index_urls or list(constants.DEFAULT_PIP_INDEX_URLS)
            main_idx = index_urls[0]
            extras = " ".join(f"--extra-index-url {u}" for u in index_urls[1:])
            quoted = " ".join(f'"{p}"' for p in pkgs)
            pre_install_cmd = (
                "PIP_DISABLE_PIP_VERSION_CHECK=1 pip install --no-warn-script-location "
                f"--index-url {main_idx} {extras} {quoted} && "
            )

        # Create N containers (one per node)
        num_nodes = trainer.num_nodes or runtime.trainer.num_nodes or 1
        containers: list[_Node] = []

        for rank in range(num_nodes):
            container_name = f"{job_name}-node-{rank}"

            # torchrun rendezvous across nodes using network alias
            # We use a simple host:port rendezvous; rank 0 acts as master.
            master_addr = f"{job_name}-node-0"
            master_port = 29500

            # Prefer torchrun; fall back to python if torchrun is unavailable.
            # IMPORTANT: Only add torchrun flags when torchrun is present. If we fall back to
            # python, run the script directly without torchrun-specific arguments to avoid
            # immediate failure and short-lived containers.
            entry_cmd = (
                f"{pre_install_cmd}"
                "if command -v torchrun >/dev/null 2>&1; then "
                f"  torchrun --nproc_per_node=1 --nnodes={num_nodes} "
                f"  --node_rank={rank} --rdzv_backend=c10d "
                f"  --rdzv_endpoint={master_addr}:{master_port} "
                f"  {self.cfgd_workspace_path()}train.py; "
                "else "
                f"  python {self.cfgd_workspace_path()}train.py; "
                "fi"
            )

            full_cmd = ("bash", "-lc", entry_cmd)

            labels = {
                f"{DOCKER_LABEL_PREFIX}/trainjob-name": job_name,
                f"{DOCKER_LABEL_PREFIX}/step": f"node-{rank}",
            }

            # Bind mount the host workdir into the container working dir
            binds = {
                workdir: {
                    "bind": "/workspace",
                    "mode": "rw",
                }
            }

            container = self.client.containers.run(
                image=image,
                command=full_cmd,
                name=container_name,
                detach=True,
                working_dir="/workspace",
                network=network_name,
                environment=env,
                labels=labels,
                volumes=binds,
                auto_remove=False,  # we control removal in delete_job
            )

            containers.append(_Node(name=container_name, container=container))

        self._jobs[job_name] = _Job(
            name=job_name,
            created=datetime.now(),
            runtime=runtime,
            network_name=network_name,
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
                        status=self._container_status(node.container),
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
            status = self._container_status(node.container)
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
            logs = node.container.logs(stream=bool(follow), follow=bool(follow))
            if follow:
                for chunk in logs:
                    yield chunk.decode("utf-8", errors="ignore")
            else:
                yield logs.decode("utf-8", errors="ignore")

    def wait_for_job_status(
        self,
        name: str,
        status: set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        # Simple polling loop similar to Kubernetes backend
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

        # Stop containers, collect final logs if needed, and remove
        from contextlib import suppress

        for node in job.nodes:
            with suppress(Exception):
                node.container.stop(timeout=10)
            with suppress(Exception):
                node.container.remove(force=True)

        # Remove network (best-effort)
        try:
            net = self.client.networks.get(job.network_name)
            net.remove()
        except Exception:
            pass

        # Remove working directory if configured
        if self.cfg.auto_remove and os.path.isdir(job.workdir_host):
            shutil.rmtree(job.workdir_host, ignore_errors=True)

        del self._jobs[name]

    # ---- Helpers ----
    def _write_training_script(self, workdir: str, trainer: types.CustomTrainer) -> Path:
        script_path = Path(workdir) / "train.py"
        # Serialize function source and immediate call (mirrors localprocess utils)
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

    def _resolve_image(self, runtime: types.Runtime) -> str:
        # Prefer explicit image from config; else from runtime YAML. Runtime YAML
        # must include the image field; otherwise we fail with a clear message.
        if self.cfg.image:
            return self.cfg.image

        # Load from local runtime YAMLs directory
        # The loader already validated presence, but not the image; enforce here.
        # We reuse the torch-distributed default when name matches.
        # For extensibility, a mapping could be added in the YAML schema later.
        name = runtime.name
        # Best-effort: read the same YAML used for this runtime to fetch image.
        # The simple loader does not expose image directly, so we re-read YAML by matching name.
        from kubeflow.trainer.backends.docker.runtime_loader import LOCAL_RUNTIMES_DIR

        for f in sorted(LOCAL_RUNTIMES_DIR.glob("*.yaml")):
            try:
                import yaml

                data = yaml.safe_load(Path(f).read_text())
                # CRD-like structure only (ClusterTrainingRuntime or TrainingRuntime)
                if (
                    data.get("kind") in {"ClusterTrainingRuntime", "TrainingRuntime"}
                    and data.get("metadata", {}).get("name") == name
                ):
                    # locate the 'node' replicated job
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
            f"No image specified for runtime '{name}'. Provide DockerBackendConfig.image or "
            f"add an 'image' field to its YAML in {LOCAL_RUNTIMES_DIR}."
        )

    def _maybe_pull_image(self, image: str):
        policy = (self.cfg.pull_policy or "IfNotPresent").lower()
        try:
            if policy == "never":
                # Ensure image exists locally
                self.client.images.get(image)
                return
            if policy == "always":
                logger.debug(f"Pulling image (Always): {image}")
                self.client.images.pull(image)
                return
            # IfNotPresent
            try:
                self.client.images.get(image)
            except Exception:
                logger.debug(f"Pulling image (IfNotPresent): {image}")
                self.client.images.pull(image)
        except Exception as e:
            raise RuntimeError(f"Failed to ensure image '{image}': {e}") from e

    def _ensure_network(self, job_name: str) -> str:
        # Create a job-scoped network to allow name-based discovery between nodes
        network_name = f"{job_name}-net"
        try:
            self.client.networks.get(network_name)
            return network_name
        except Exception:
            pass
        # Create network with labels for filtering and management
        self.client.networks.create(
            name=network_name,
            check_duplicate=True,
            labels={
                "trainer.kubeflow.org/trainjob-name": job_name,
            },
        )
        return network_name

    def _container_status(self, container: Container) -> str:
        try:
            container.reload()
            status = container.status  # e.g., created, running, exited
            if status == "running":
                return constants.TRAINJOB_RUNNING
            if status == "created":
                return constants.TRAINJOB_CREATED
            if status == "exited":
                # Exit code 0 -> complete, else failed
                inspect = self.client.api.inspect_container(container.id)
                code = inspect.get("State", {}).get("ExitCode")
                return constants.TRAINJOB_COMPLETE if code == 0 else constants.TRAINJOB_FAILED
        except Exception:
            return constants.UNKNOWN
        return constants.UNKNOWN

    def _aggregate_status(self, job: _Job) -> str:
        statuses = [self._container_status(n.container) for n in job.nodes]
        if constants.TRAINJOB_FAILED in statuses:
            return constants.TRAINJOB_FAILED
        if constants.TRAINJOB_RUNNING in statuses:
            return constants.TRAINJOB_RUNNING
        if all(s == constants.TRAINJOB_COMPLETE for s in statuses if s != constants.UNKNOWN):
            return constants.TRAINJOB_COMPLETE
        if any(s == constants.TRAINJOB_CREATED for s in statuses):
            return constants.TRAINJOB_CREATED
        return constants.UNKNOWN

    def _run_oneoff_container(self, image: str, command: tuple[str, ...]) -> str:
        """
        Run a short-lived container and return its stdout as a string.

        Implementation detail:
        - Use detach=False and remove=True so Docker SDK runs the container to
          completion, returns combined logs as bytes, and cleans it up. This
          avoids races where auto-removed containers disappear before .wait/.logs.
        """
        try:
            output = self.client.containers.run(
                image=image,
                command=command,
                detach=False,
                remove=True,
            )
        except Exception as e:
            raise RuntimeError(f"One-off container failed to run: {e}") from e

        if isinstance(output, (bytes, bytearray)):
            return output.decode("utf-8", errors="ignore")
        # Fallback: ensure string return
        return str(output)

    def cfgd_workspace_path(self) -> str:
        # Location inside the container where the host workdir is mounted
        return "/workspace/"
