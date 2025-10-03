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
PodmanBackend
-------------

Local execution backend for `CustomTrainer` jobs using Podman containers.

Key behaviors:
- Multi-node jobs: containers are connected via a Podman network, giving each
  container its own network interface for proper distributed training with
  name-based DNS resolution (similar to Docker).
- Entry script generation: we serialize the user's training function to a small
  Python file and invoke it inside the container using `torchrun` (preferred) or
  `python` as a fallback.
- Runtimes: we use `config/local_runtimes` to define runtime images and
  characteristics (e.g., torch). Defaults to `torch-distributed` if no runtime
  is provided.
- Image pulling: controlled via `pull_policy` and performed automatically if
  needed.
- Logs and lifecycle: streaming logs and deletion semantics similar to the
  Docker backend.
- Networking: Uses Podman networks (not pods) so each container has its own
  network interface, enabling proper PyTorch distributed training communication.
"""

from __future__ import annotations

from collections.abc import Iterator
import logging
from pathlib import Path

try:
    import podman  # type: ignore
except Exception:  # pragma: no cover - optional dependency, validated at runtime
    podman = None  # type: ignore

from kubeflow.trainer.backends.container_base import ContainerBackend
from kubeflow.trainer.backends.podman.runtime_loader import (
    get_local_runtime,
    list_local_runtimes,
)
from kubeflow.trainer.backends.podman.types import LocalPodmanBackendConfig
from kubeflow.trainer.types import types

logger = logging.getLogger(__name__)


class LocalPodmanBackend(ContainerBackend):
    def __init__(self, cfg: LocalPodmanBackendConfig):
        if podman is None:
            raise ImportError(
                "The 'podman' Python package is not installed. Install with extras: "
                "pip install kubeflow[podman]"
            )

        super().__init__(label_prefix="trainer.kubeflow.org")

        # Initialize Podman client
        if cfg.podman_url:
            self.client = podman.PodmanClient(base_url=cfg.podman_url)
        else:
            # Use system default socket
            self.client = podman.PodmanClient()

        self.cfg = cfg

    # ---- Runtime APIs ----
    def list_runtimes(self) -> list[types.Runtime]:
        return list_local_runtimes()

    def get_runtime(self, name: str) -> types.Runtime:
        return get_local_runtime(name)

    # ---- Container backend implementation ----
    def _get_client(self):
        return self.client

    def _create_network(self, job_name: str) -> str:
        """
        Create a Podman network for the job to enable container networking.

        Unlike pods (which share network namespace), a Podman network gives each
        container its own network interface, enabling proper distributed training.
        """
        network_name = f"{job_name}-net"
        try:
            self.client.networks.get(network_name)
            return network_name
        except Exception:
            pass
        # Create network with DNS enabled for hostname resolution
        self.client.networks.create(
            name=network_name,
            driver="bridge",
            dns_enabled=True,
            labels={
                f"{self.label_prefix}/trainjob-name": job_name,
            },
        )
        return network_name

    def _delete_network(self, network_id: str):
        """Delete the Podman network."""
        try:
            net = self.client.networks.get(network_id)
            net.remove()
        except Exception:
            pass

    def _get_master_addr(self, job_name: str, rank: int) -> str:
        """Podman uses container names for DNS resolution within a network."""
        return f"{job_name}-node-0"

    def _get_master_port(self, rank: int) -> int:
        """Use a fixed port since containers are on different network endpoints."""
        return 29500

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
        """Create and start a Podman container on the network."""
        # Use containers.run() which creates and starts in one step
        # This is equivalent to: podman run --network <network_id> ...
        container = self.client.containers.run(
            image=image,
            command=command,
            name=name,
            network=network_id,
            working_dir=working_dir,
            environment=environment,
            labels=labels,
            volumes=volumes,
            detach=True,
            remove=False,  # Don't auto-remove, we control this in delete_job
        )

        return container.id

    def _get_container(self, container_id: str):
        """Get Podman container by ID."""
        return self.client.containers.get(container_id)

    def _container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from a Podman container."""
        container = self._get_container(container_id)
        logs = container.logs(stream=bool(follow), follow=bool(follow))
        if follow:
            for chunk in logs:
                if isinstance(chunk, bytes):
                    yield chunk.decode("utf-8", errors="ignore")
                else:
                    yield str(chunk)
        else:
            if isinstance(logs, bytes):
                yield logs.decode("utf-8", errors="ignore")
            else:
                yield str(logs)

    def _stop_container(self, container_id: str, timeout: int = 10):
        """Stop a Podman container."""
        container = self._get_container(container_id)
        container.stop(timeout=timeout)

    def _remove_container(self, container_id: str, force: bool = True):
        """Remove a Podman container."""
        container = self._get_container(container_id)
        container.remove(force=force)

    def _pull_image(self, image: str):
        """Pull a Podman image."""
        self.client.images.pull(image)

    def _image_exists(self, image: str) -> bool:
        """Check if Podman image exists locally."""
        try:
            self.client.images.get(image)
            return True
        except Exception:
            return False

    def _run_oneoff_container(self, image: str, command: list[str]) -> str:
        """Run a short-lived Podman container and return its logs."""
        try:
            container = self.client.containers.create(
                image=image,
                command=command,
                detach=False,
                remove=True,
            )
            container.start()
            container.wait()
            logs = container.logs()

            if isinstance(logs, (bytes, bytearray)):
                return logs.decode("utf-8", errors="ignore")
            return str(logs)
        except Exception as e:
            raise RuntimeError(f"One-off container failed to run: {e}") from e

    def _get_pull_policy(self) -> str:
        return self.cfg.pull_policy

    def _get_workdir_base(self) -> str | None:
        return self.cfg.workdir_base

    def _get_auto_remove(self) -> bool:
        return self.cfg.auto_remove

    def _build_environment(self, trainer: types.CustomTrainer) -> dict[str, str]:
        """Build environment variables for containers."""
        env = dict(self.cfg.env or {})
        if trainer.env:
            env.update(trainer.env)
        return env

    def _get_runtimes_dir(self) -> Path:
        """Get the path to the local runtimes directory."""
        from kubeflow.trainer.backends.podman.runtime_loader import LOCAL_RUNTIMES_DIR
        return LOCAL_RUNTIMES_DIR

    def _resolve_image(self, runtime: types.Runtime) -> str:
        """Resolve the Podman image for a runtime."""
        return self._resolve_image_from_runtime(runtime, self.cfg.image)
