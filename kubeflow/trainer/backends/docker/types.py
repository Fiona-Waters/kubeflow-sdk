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
Types and configuration for the Docker backend.

We keep the configuration surface area intentionally small for v1:
 - image: Optional explicit image. If omitted, use the image referenced by the
   selected runtime (e.g., torch_distributed) from `config/local_runtimes`.
 - network: Optional explicit Docker network name. If omitted, the backend will
   create a per-job ephemeral network to interconnect all node containers.
 - pull_policy: Controls image pulling. Supported values: "IfNotPresent",
   "Always", "Never". The default is "IfNotPresent".
 - auto_remove: Whether to remove containers and ephemeral network when jobs
   are deleted. Defaults to True.
 - gpus: GPU support is not implemented for v1 (kept for future extensibility).
 - env: Optional global environment variables applied to all containers.
 - docker_host: Optional override for connecting to a remote/local Docker
   daemon; by default the Docker SDK resolves from environment and local
   configuration.
"""

from typing import Optional, Union

from pydantic import BaseModel, Field


class LocalDockerBackendConfig(BaseModel):
    image: Optional[str] = Field(default=None)
    network: Optional[str] = Field(default=None)
    pull_policy: str = Field(default="IfNotPresent")
    auto_remove: bool = Field(default=True)
    # In Python 3.9, avoid PEP 604 unions; use typing.Union/Optional instead.
    # Define the type at module scope so Pydantic doesn't treat it as a field.
    gpus: Optional[Union[int, bool]] = Field(default=None)
    env: Optional[dict[str, str]] = Field(default=None)
    docker_host: Optional[str] = Field(default=None)
    # Base directory on the host to place per-job working dirs that are bind-mounted
    # into containers as /workspace. Defaults to a path under the user's home to
    # maximize compatibility with Docker Desktop file sharing on macOS/Windows.
    workdir_base: Optional[str] = Field(default=None)
