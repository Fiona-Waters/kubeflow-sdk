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
Unit tests for ContainerBackend.

Tests the ContainerBackend class with mocked container adapters.
"""

from collections.abc import Iterator
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.backends.container.backend import ContainerBackend
from kubeflow.trainer.backends.container.client_adapter import ContainerClientAdapter
from kubeflow.trainer.backends.container.types import ContainerBackendConfig
from kubeflow.trainer.constants import constants
from kubeflow.trainer.test.common import FAILED, SUCCESS, TestCase
from kubeflow.trainer.types import types


# Mock Container Adapter
class MockContainerAdapter(ContainerClientAdapter):
    """Mock adapter for testing ContainerBackend without Docker/Podman."""

    def __init__(self):
        self._runtime_type = "mock"
        self.networks_created = []
        self.containers_created = []
        self.containers_stopped = []
        self.containers_removed = []
        self.networks_deleted = []
        self.images_pulled = []
        self.ping_called = False

    def ping(self):
        self.ping_called = True

    def create_network(self, name: str, labels: dict[str, str]) -> str:
        network_id = f"net-{name}"
        self.networks_created.append({"id": network_id, "name": name, "labels": labels})
        return network_id

    def delete_network(self, network_id: str):
        self.networks_deleted.append(network_id)

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
        container_id = f"container-{len(self.containers_created)}"
        self.containers_created.append(
            {
                "id": container_id,
                "name": name,
                "image": image,
                "command": command,
                "network": network_id,
                "environment": environment,
                "labels": labels,
                "volumes": volumes,
                "working_dir": working_dir,
                "status": "running",
                "exit_code": None,
            }
        )
        return container_id

    def get_container(self, container_id: str):
        for container in self.containers_created:
            if container["id"] == container_id:
                return Mock(id=container_id, status=container["status"])
        return None

    def container_logs(self, container_id: str, follow: bool) -> Iterator[str]:
        if follow:
            yield f"Log line 1 from {container_id}\n"
            yield f"Log line 2 from {container_id}\n"
        else:
            yield f"Complete log from {container_id}\n"

    def stop_container(self, container_id: str, timeout: int = 10):
        self.containers_stopped.append(container_id)
        for container in self.containers_created:
            if container["id"] == container_id:
                container["status"] = "exited"
                container["exit_code"] = 0

    def remove_container(self, container_id: str, force: bool = True):
        self.containers_removed.append(container_id)

    def pull_image(self, image: str):
        self.images_pulled.append(image)

    def image_exists(self, image: str) -> bool:
        return "local" in image or image in self.images_pulled

    def run_oneoff_container(self, image: str, command: list[str]) -> str:
        return "Python 3.9.0\npip 21.0.1\nnvidia-smi not found\n"

    def container_status(self, container_id: str) -> tuple[str, int | None]:
        for container in self.containers_created:
            if container["id"] == container_id:
                return (container["status"], container.get("exit_code"))
        return ("unknown", None)

    def set_container_status(self, container_id: str, status: str, exit_code: int | None = None):
        """Helper method to set container status for testing."""
        for container in self.containers_created:
            if container["id"] == container_id:
                container["status"] = status
                container["exit_code"] = exit_code


# Fixtures
@pytest.fixture
def container_backend():
    """Provide ContainerBackend with mocked adapter."""
    backend = ContainerBackend(ContainerBackendConfig())
    backend._adapter = MockContainerAdapter()
    return backend


@pytest.fixture
def temp_workdir():
    """Provide a temporary working directory."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)


# Helper Function
def simple_train_func():
    """Simple training function for tests."""
    print("Training")


# Tests
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="auto-detect docker first",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="auto-detect falls back to podman",
            expected_status=SUCCESS,
        ),
        TestCase(
            name="both unavailable raises error",
            expected_status=FAILED,
            expected_error=RuntimeError,
        ),
    ],
)
def test_backend_initialization(test_case):
    """Test ContainerBackend initialization and adapter creation."""
    print("Executing test:", test_case.name)
    try:
        if test_case.name == "auto-detect docker first":
            with (
                patch(
                    "kubeflow.trainer.backends.container.backend.DockerClientAdapter"
                ) as mock_docker,
                patch(
                    "kubeflow.trainer.backends.container.backend.PodmanClientAdapter"
                ) as mock_podman,
            ):
                mock_docker_instance = Mock()
                mock_docker.return_value = mock_docker_instance

                _ = ContainerBackend(ContainerBackendConfig())

                mock_docker.assert_called_once_with(None)
                mock_docker_instance.ping.assert_called_once()
                mock_podman.assert_not_called()
                assert test_case.expected_status == SUCCESS

        elif test_case.name == "auto-detect falls back to podman":
            with (
                patch(
                    "kubeflow.trainer.backends.container.backend.DockerClientAdapter"
                ) as mock_docker,
                patch(
                    "kubeflow.trainer.backends.container.backend.PodmanClientAdapter"
                ) as mock_podman,
            ):
                mock_docker_instance = Mock()
                mock_docker_instance.ping.side_effect = Exception("Docker not available")
                mock_docker.return_value = mock_docker_instance

                mock_podman_instance = Mock()
                mock_podman.return_value = mock_podman_instance

                _ = ContainerBackend(ContainerBackendConfig())

                mock_docker.assert_called_once()
                mock_podman.assert_called_once_with(None)
                mock_podman_instance.ping.assert_called_once()
                assert test_case.expected_status == SUCCESS

        elif test_case.name == "both unavailable raises error":
            with (
                patch(
                    "kubeflow.trainer.backends.container.backend.DockerClientAdapter"
                ) as mock_docker,
                patch(
                    "kubeflow.trainer.backends.container.backend.PodmanClientAdapter"
                ) as mock_podman,
            ):
                mock_docker_instance = Mock()
                mock_docker_instance.ping.side_effect = Exception("Docker not available")
                mock_docker.return_value = mock_docker_instance

                mock_podman_instance = Mock()
                mock_podman_instance.ping.side_effect = Exception("Podman not available")
                mock_podman.return_value = mock_podman_instance

                ContainerBackend(ContainerBackendConfig())

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_list_runtimes(container_backend):
    """Test listing available local runtimes."""
    print("Executing test: list_runtimes")
    runtimes = container_backend.list_runtimes()

    assert isinstance(runtimes, list)
    assert len(runtimes) > 0
    runtime_names = [r.name for r in runtimes]
    assert "torch-distributed" in runtime_names
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get valid runtime",
            expected_status=SUCCESS,
            config={"name": "torch-distributed"},
        ),
        TestCase(
            name="get invalid runtime",
            expected_status=FAILED,
            config={"name": "nonexistent-runtime"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime(container_backend, test_case):
    """Test getting a specific runtime."""
    print("Executing test:", test_case.name)
    try:
        runtime = container_backend.get_runtime(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(runtime, types.Runtime)
        assert runtime.name == test_case.config["name"]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


def test_get_runtime_packages(container_backend):
    """Test getting runtime packages."""
    print("Executing test: get_runtime_packages")
    runtime = container_backend.get_runtime("torch-distributed")
    container_backend.get_runtime_packages(runtime)

    assert len(
        container_backend._adapter.images_pulled
    ) > 0 or container_backend._adapter.image_exists(runtime.trainer.image)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="train single node",
            expected_status=SUCCESS,
            config={"num_nodes": 1, "expected_containers": 1},
        ),
        TestCase(
            name="train multi-node",
            expected_status=SUCCESS,
            config={"num_nodes": 3, "expected_containers": 3},
        ),
        TestCase(
            name="train with custom env",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "env": {"MY_VAR": "my_value", "DEBUG": "true"},
                "expected_containers": 1,
            },
        ),
        TestCase(
            name="train with packages",
            expected_status=SUCCESS,
            config={
                "num_nodes": 1,
                "packages": ["numpy", "pandas"],
                "expected_containers": 1,
            },
        ),
    ],
)
def test_train(container_backend, test_case):
    """Test training job creation."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(
            func=simple_train_func,
            num_nodes=test_case.config.get("num_nodes", 1),
            env=test_case.config.get("env"),
            packages_to_install=test_case.config.get("packages"),
        )
        runtime = container_backend.get_runtime("torch-distributed")

        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        assert test_case.expected_status == SUCCESS
        assert job_name is not None
        assert len(job_name) == 12
        assert (
            len(container_backend._adapter.containers_created)
            == test_case.config["expected_containers"]
        )
        assert len(container_backend._adapter.networks_created) == 1

        # Check environment if specified
        if "env" in test_case.config:
            container = container_backend._adapter.containers_created[0]
            for key, value in test_case.config["env"].items():
                assert container["environment"][key] == value

        # Check packages if specified
        if "packages" in test_case.config:
            container = container_backend._adapter.containers_created[0]
            command_str = " ".join(container["command"])
            assert "pip install" in command_str
            for package in test_case.config["packages"]:
                assert package in command_str

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list all jobs",
            expected_status=SUCCESS,
            config={"num_jobs": 2},
        ),
        TestCase(
            name="list empty jobs",
            expected_status=SUCCESS,
            config={"num_jobs": 0},
        ),
    ],
)
def test_list_jobs(container_backend, test_case):
    """Test listing training jobs."""
    print("Executing test:", test_case.name)
    try:
        runtime = container_backend.get_runtime("torch-distributed")
        created_jobs = []

        for _ in range(test_case.config["num_jobs"]):
            trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
            job_name = container_backend.train(runtime=runtime, trainer=trainer)
            created_jobs.append(job_name)

        jobs = container_backend.list_jobs()

        assert test_case.expected_status == SUCCESS
        assert len(jobs) == test_case.config["num_jobs"]
        if test_case.config["num_jobs"] > 0:
            job_names = [job.name for job in jobs]
            for created_job in created_jobs:
                assert created_job in job_names

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get existing job",
            expected_status=SUCCESS,
            config={"num_nodes": 2},
        ),
        TestCase(
            name="get nonexistent job",
            expected_status=FAILED,
            config={"job_name": "nonexistent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_job(container_backend, test_case):
    """Test getting a specific job."""
    print("Executing test:", test_case.name)
    try:
        if test_case.name == "get existing job":
            trainer = types.CustomTrainer(
                func=simple_train_func, num_nodes=test_case.config["num_nodes"]
            )
            runtime = container_backend.get_runtime("torch-distributed")
            job_name = container_backend.train(runtime=runtime, trainer=trainer)

            job = container_backend.get_job(job_name)

            assert test_case.expected_status == SUCCESS
            assert job.name == job_name
            assert job.num_nodes == test_case.config["num_nodes"]
            assert len(job.steps) == test_case.config["num_nodes"]

        elif test_case.name == "get nonexistent job":
            container_backend.get_job(test_case.config["job_name"])

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get logs no follow",
            expected_status=SUCCESS,
            config={"follow": False},
        ),
        TestCase(
            name="get logs with follow",
            expected_status=SUCCESS,
            config={"follow": True},
        ),
    ],
)
def test_get_job_logs(container_backend, test_case):
    """Test getting job logs."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
        runtime = container_backend.get_runtime("torch-distributed")
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        logs = list(container_backend.get_job_logs(job_name, follow=test_case.config["follow"]))

        assert test_case.expected_status == SUCCESS
        assert len(logs) > 0
        if test_case.config["follow"]:
            assert any("Log line" in log for log in logs)
        else:
            assert any("Complete log" in log for log in logs)

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait for complete",
            expected_status=SUCCESS,
            config={"wait_status": constants.TRAINJOB_COMPLETE, "container_exit_code": 0},
        ),
        TestCase(
            name="wait timeout",
            expected_status=FAILED,
            config={"wait_status": constants.TRAINJOB_COMPLETE, "timeout": 2},
            expected_error=TimeoutError,
        ),
        TestCase(
            name="job fails",
            expected_status=FAILED,
            config={"wait_status": constants.TRAINJOB_COMPLETE, "container_exit_code": 1},
            expected_error=RuntimeError,
        ),
    ],
)
def test_wait_for_job_status(container_backend, test_case):
    """Test waiting for job status."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
        runtime = container_backend.get_runtime("torch-distributed")
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        if test_case.name == "wait for complete":
            container_id = container_backend._adapter.containers_created[0]["id"]
            container_backend._adapter.set_container_status(
                container_id, "exited", test_case.config["container_exit_code"]
            )

            completed_job = container_backend.wait_for_job_status(
                job_name, status={test_case.config["wait_status"]}, timeout=5, polling_interval=1
            )

            assert test_case.expected_status == SUCCESS
            assert completed_job.status == constants.TRAINJOB_COMPLETE

        elif test_case.name == "wait timeout":
            container_backend.wait_for_job_status(
                job_name,
                status={test_case.config["wait_status"]},
                timeout=test_case.config["timeout"],
                polling_interval=1,
            )

        elif test_case.name == "job fails":
            container_id = container_backend._adapter.containers_created[0]["id"]
            container_backend._adapter.set_container_status(
                container_id, "exited", test_case.config["container_exit_code"]
            )

            container_backend.wait_for_job_status(
                job_name, status={test_case.config["wait_status"]}, timeout=5, polling_interval=1
            )

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete with auto_remove true",
            expected_status=SUCCESS,
            config={"auto_remove": True, "num_nodes": 2},
        ),
        TestCase(
            name="delete with auto_remove false",
            expected_status=SUCCESS,
            config={"auto_remove": False, "num_nodes": 2},
        ),
    ],
)
def test_delete_job(container_backend, temp_workdir, test_case):
    """Test deleting a job."""
    print("Executing test:", test_case.name)
    try:
        container_backend.cfg.workdir_base = temp_workdir
        container_backend.cfg.auto_remove = test_case.config["auto_remove"]

        trainer = types.CustomTrainer(
            func=simple_train_func, num_nodes=test_case.config["num_nodes"]
        )
        runtime = container_backend.get_runtime("torch-distributed")
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        job_workdir = Path(temp_workdir) / job_name
        assert job_workdir.exists()

        container_backend.delete_job(job_name)

        assert test_case.expected_status == SUCCESS
        assert len(container_backend._adapter.containers_stopped) == test_case.config["num_nodes"]
        assert len(container_backend._adapter.containers_removed) == test_case.config["num_nodes"]
        assert len(container_backend._adapter.networks_deleted) == 1

        if test_case.config["auto_remove"]:
            assert not job_workdir.exists()
        else:
            assert job_workdir.exists()

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="running container",
            expected_status=SUCCESS,
            config={
                "container_status": "running",
                "exit_code": None,
                "expected_job_status": constants.TRAINJOB_RUNNING,
            },
        ),
        TestCase(
            name="exited success",
            expected_status=SUCCESS,
            config={
                "container_status": "exited",
                "exit_code": 0,
                "expected_job_status": constants.TRAINJOB_COMPLETE,
            },
        ),
        TestCase(
            name="exited failure",
            expected_status=SUCCESS,
            config={
                "container_status": "exited",
                "exit_code": 1,
                "expected_job_status": constants.TRAINJOB_FAILED,
            },
        ),
    ],
)
def test_container_status_mapping(container_backend, test_case):
    """Test container status mapping to TrainJob status."""
    print("Executing test:", test_case.name)
    try:
        trainer = types.CustomTrainer(func=simple_train_func, num_nodes=1)
        runtime = container_backend.get_runtime("torch-distributed")
        job_name = container_backend.train(runtime=runtime, trainer=trainer)

        container_id = container_backend._adapter.containers_created[0]["id"]
        container_backend._adapter.set_container_status(
            container_id, test_case.config["container_status"], test_case.config["exit_code"]
        )

        job = container_backend.get_job(job_name)

        assert test_case.expected_status == SUCCESS
        assert job.status == test_case.config["expected_job_status"]

    except Exception as e:
        assert type(e) is test_case.expected_error
    print("test execution complete")
