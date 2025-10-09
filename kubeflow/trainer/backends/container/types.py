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
Types and configuration for the unified Container backend.

This backend automatically detects and uses either Docker or Podman.
It provides a single interface for container-based execution regardless
of the underlying runtime.

Configuration options:
 - image: Optional explicit image. If omitted, use the image referenced by the
   selected runtime (e.g., torch_distributed) from `config/local_runtimes`.
 - pull_policy: Controls image pulling. Supported values: "IfNotPresent",
   "Always", "Never". The default is "IfNotPresent".
 - auto_remove: Whether to remove containers and networks when jobs are deleted.
   Defaults to True.
 - gpus: GPU support (implementation varies between Docker and Podman).
   Defaults to None.
 - env: Optional global environment variables applied to all containers.
 - container_host: Optional override for connecting to a remote/local container
   daemon. By default, auto-detects from environment or uses system defaults.
   For Docker: uses DOCKER_HOST or default socket.
   For Podman: uses CONTAINER_HOST or default socket.
 - workdir_base: Base directory on the host to place per-job working dirs that
   are bind-mounted into containers as /workspace. Defaults to a path under the
   user's home directory for compatibility.
 - runtime: Force use of a specific container runtime ("docker" or "podman").
   If not set, auto-detects based on availability (tries Docker first, then Podman).
"""

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field


class ContainerBackendConfig(BaseModel):
    image: Optional[str] = Field(default=None)
    pull_policy: str = Field(default="IfNotPresent")
    auto_remove: bool = Field(default=True)
    gpus: Optional[Union[int, bool]] = Field(default=None)
    env: Optional[dict[str, str]] = Field(default=None)
    container_host: Optional[str] = Field(default=None)
    workdir_base: Optional[str] = Field(default=None)
    runtime: Optional[Literal["docker", "podman"]] = Field(default=None)
