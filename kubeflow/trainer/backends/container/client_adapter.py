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
Container client adapters for Docker and Podman.

This module implements the adapter pattern to abstract away differences between
Docker and Podman APIs, allowing the backend to work with either runtime through
a common interface.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator


class ContainerClientAdapter(abc.ABC):
    """
    Abstract adapter interface for container clients.

    This adapter abstracts the container runtime API, allowing the backend
    to work with Docker and Podman through a unified interface.
    """

    @abc.abstractmethod
    def ping(self):
        """Test the connection to the container runtime."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_network(
        self,
        name: str,
        labels: dict[str, str],
    ) -> str:
        """
        Create a container network.

        Args:
            name: Network name
            labels: Labels to attach to the network

        Returns:
            Network ID or name
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_network(self, network_id: str):
        """Delete a network."""
        raise NotImplementedError()

    @abc.abstractmethod
    def create_and_start_container(
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
        """
        Create and start a container.

        Args:
            image: Container image
            command: Command to run
            name: Container name
            network_id: Network to attach to
            environment: Environment variables
            labels: Container labels
            volumes: Volume mounts
            working_dir: Working directory

        Returns:
            Container ID
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_container(self, container_id: str):
        """Get container object by ID."""
        raise NotImplementedError()

    @abc.abstractmethod
    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def stop_container(self, container_id: str, timeout: int = 10):
        """Stop a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def remove_container(self, container_id: str, force: bool = True):
        """Remove a container."""
        raise NotImplementedError()

    @abc.abstractmethod
    def pull_image(self, image: str):
        """Pull an image."""
        raise NotImplementedError()

    @abc.abstractmethod
    def image_exists(self, image: str) -> bool:
        """Check if an image exists locally."""
        raise NotImplementedError()

    @abc.abstractmethod
    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        """
        Run a short-lived container and return its output.

        Args:
            image: Container image
            command: Command to run

        Returns:
            Container output as string
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def container_status(self, container_id: str) -> tuple[str, int | None]:
        """
        Get container status.

        Returns:
            Tuple of (status_string, exit_code)
            Status strings: "running", "created", "exited", etc.
            Exit code is None if container hasn't exited
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_container_ip(self, container_id: str, network_id: str) -> str | None:
        """
        Get container's IP address on a specific network.

        Args:
            container_id: Container ID
            network_id: Network name or ID

        Returns:
            IP address string or None if not found
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def list_containers(self, filters: dict[str, str] | None = None) -> list[dict]:
        """
        List containers, optionally filtered by labels.

        Args:
            filters: Dictionary of filters (e.g., {"label": ["key=value"]})

        Returns:
            List of container info dictionaries with keys:
            - id: Container ID
            - name: Container name
            - labels: Dictionary of labels
            - status: Container status
            - created: Creation timestamp
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_network(self, network_id: str) -> dict | None:
        """
        Get network information by ID or name.

        Args:
            network_id: Network ID or name

        Returns:
            Dictionary with network info including labels, or None if not found
        """
        raise NotImplementedError()


class DockerClientAdapter(ContainerClientAdapter):
    """Adapter for Docker client."""

    def __init__(self, host: str | None = None):
        """
        Initialize Docker client.

        Args:
            host: Docker host URL, or None to use environment defaults
        """
        try:
            import docker  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'docker' Python package is not installed. Install with extras: "
                "pip install kubeflow[docker]"
            ) from e

        if host:
            self.client = docker.DockerClient(base_url=host)
        else:
            self.client = docker.from_env()

        self._runtime_type = "docker"

    def ping(self):
        """Test connection to Docker daemon."""
        self.client.ping()

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        """Create a Docker network."""
        try:
            self.client.networks.get(name)
            return name
        except Exception:
            pass

        self.client.networks.create(
            name=name,
            check_duplicate=True,
            labels=labels,
        )
        return name

    def delete_network(self, network_id: str):
        """Delete Docker network."""
        try:
            net = self.client.networks.get(network_id)
            net.remove()
        except Exception:
            pass

    def create_and_start_container(
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
        """Create and start a Docker container."""
        container = self.client.containers.run(
            image=image,
            command=tuple(command),
            name=name,
            detach=True,
            working_dir=working_dir,
            network=network_id,
            environment=environment,
            labels=labels,
            volumes=volumes,
            auto_remove=False,
        )
        return container.id

    def get_container(self, container_id: str):
        """Get Docker container by ID."""
        return self.client.containers.get(container_id)

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from Docker container."""
        container = self.get_container(container_id)
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

    def stop_container(self, container_id: str, timeout: int = 10):
        """Stop Docker container."""
        container = self.get_container(container_id)
        container.stop(timeout=timeout)

    def remove_container(self, container_id: str, force: bool = True):
        """Remove Docker container."""
        container = self.get_container(container_id)
        container.remove(force=force)

    def pull_image(self, image: str):
        """Pull Docker image."""
        self.client.images.pull(image)

    def image_exists(self, image: str) -> bool:
        """Check if Docker image exists locally."""
        try:
            self.client.images.get(image)
            return True
        except Exception:
            return False

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        """Run a short-lived Docker container and return output."""
        try:
            output = self.client.containers.run(
                image=image,
                command=tuple(command),
                detach=False,
                remove=True,
            )
            if isinstance(output, (bytes, bytearray)):
                return output.decode("utf-8", errors="ignore")
            return str(output)
        except Exception as e:
            raise RuntimeError(f"One-off container failed to run: {e}") from e

    def container_status(self, container_id: str) -> tuple[str, int | None]:
        """Get Docker container status."""
        try:
            container = self.get_container(container_id)
            status = container.status
            # Get exit code if container has exited
            exit_code = None
            if status == "exited":
                inspect = container.attrs if hasattr(container, "attrs") else container.inspect()
                exit_code = inspect.get("State", {}).get("ExitCode")
            return (status, exit_code)
        except Exception:
            return ("unknown", None)

    def get_container_ip(self, container_id: str, network_id: str) -> str | None:
        """Get container's IP address on a specific network."""
        try:
            container = self.get_container(container_id)
            # Refresh container info
            container.reload()
            # Get network settings
            networks = container.attrs.get("NetworkSettings", {}).get("Networks", {})

            # Try to find the network by exact name or ID
            if network_id in networks:
                return networks[network_id].get("IPAddress")

            # Fallback: return first available IP
            for net_name, net_info in networks.items():
                ip = net_info.get("IPAddress")
                if ip:
                    return ip

            return None
        except Exception:
            return None

    def list_containers(self, filters: dict[str, str] | None = None) -> list[dict]:
        """List Docker containers with optional filters."""
        try:
            containers = self.client.containers.list(all=True, filters=filters)
            result = []
            for c in containers:
                result.append({
                    "id": c.id,
                    "name": c.name,
                    "labels": c.labels,
                    "status": c.status,
                    "created": c.attrs.get("Created", ""),
                })
            return result
        except Exception:
            return []

    def get_network(self, network_id: str) -> dict | None:
        """Get Docker network information."""
        try:
            network = self.client.networks.get(network_id)
            return {
                "id": network.id,
                "name": network.name,
                "labels": network.attrs.get("Labels", {}),
            }
        except Exception:
            return None


class PodmanClientAdapter(ContainerClientAdapter):
    """Adapter for Podman client."""

    def __init__(self, host: str | None = None):
        """
        Initialize Podman client.

        Args:
            host: Podman host URL, or None to use environment defaults
        """
        try:
            import podman  # type: ignore
        except ImportError as e:
            raise ImportError(
                "The 'podman' Python package is not installed. Install with extras: "
                "pip install kubeflow[podman]"
            ) from e

        if host:
            self.client = podman.PodmanClient(base_url=host)
        else:
            self.client = podman.PodmanClient()

        self._runtime_type = "podman"

    def ping(self):
        """Test connection to Podman."""
        self.client.ping()

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        """Create a Podman network with DNS enabled."""
        try:
            self.client.networks.get(name)
            return name
        except Exception:
            pass

        self.client.networks.create(
            name=name,
            driver="bridge",
            dns_enabled=True,
            labels=labels,
        )
        return name

    def delete_network(self, network_id: str):
        """Delete Podman network."""
        try:
            net = self.client.networks.get(network_id)
            net.remove()
        except Exception:
            pass

    def create_and_start_container(
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
        """Create and start a Podman container."""
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
            remove=False,
        )
        return container.id

    def get_container(self, container_id: str):
        """Get Podman container by ID."""
        return self.client.containers.get(container_id)

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        """Stream logs from Podman container."""
        container = self.get_container(container_id)
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

    def stop_container(self, container_id: str, timeout: int = 10):
        """Stop Podman container."""
        container = self.get_container(container_id)
        container.stop(timeout=timeout)

    def remove_container(self, container_id: str, force: bool = True):
        """Remove Podman container."""
        container = self.get_container(container_id)
        container.remove(force=force)

    def pull_image(self, image: str):
        """Pull Podman image."""
        self.client.images.pull(image)

    def image_exists(self, image: str) -> bool:
        """Check if Podman image exists locally."""
        try:
            self.client.images.get(image)
            return True
        except Exception:
            return False

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        """Run a short-lived Podman container and return output."""
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

    def container_status(self, container_id: str) -> tuple[str, int | None]:
        """Get Podman container status."""
        try:
            container = self.get_container(container_id)
            status = container.status
            # Get exit code if container has exited
            exit_code = None
            if status == "exited":
                inspect = container.attrs if hasattr(container, "attrs") else container.inspect()
                exit_code = inspect.get("State", {}).get("ExitCode")
            return (status, exit_code)
        except Exception:
            return ("unknown", None)

    def get_container_ip(self, container_id: str, network_id: str) -> str | None:
        """Get container's IP address on a specific network."""
        try:
            container = self.get_container(container_id)
            # Get container inspect data
            inspect = container.attrs if hasattr(container, "attrs") else container.inspect()

            # Get network settings - Podman structure is similar to Docker
            networks = inspect.get("NetworkSettings", {}).get("Networks", {})

            # Try to find the network by exact name or ID
            if network_id in networks:
                return networks[network_id].get("IPAddress")

            # Fallback: return first available IP
            for net_name, net_info in networks.items():
                ip = net_info.get("IPAddress")
                if ip:
                    return ip

            return None
        except Exception:
            return None

    def list_containers(self, filters: dict[str, str] | None = None) -> list[dict]:
        """List Podman containers with optional filters."""
        try:
            containers = self.client.containers.list(all=True, filters=filters)
            result = []
            for c in containers:
                inspect = c.attrs if hasattr(c, "attrs") else c.inspect()
                result.append({
                    "id": c.id,
                    "name": c.name,
                    "labels": c.labels if hasattr(c, "labels") else inspect.get("Config", {}).get("Labels", {}),
                    "status": c.status,
                    "created": inspect.get("Created", ""),
                })
            return result
        except Exception:
            return []

    def get_network(self, network_id: str) -> dict | None:
        """Get Podman network information."""
        try:
            network = self.client.networks.get(network_id)
            inspect = network.attrs if hasattr(network, "attrs") else network.inspect()
            return {
                "id": inspect.get("ID", network_id),
                "name": inspect.get("Name", network_id),
                "labels": inspect.get("Labels", {}),
            }
        except Exception:
            return None
