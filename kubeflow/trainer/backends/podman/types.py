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
Types and configuration for the Podman backend.

Configuration options for Podman backend:
 - image: Optional explicit image. If omitted, use the image referenced by the
   selected runtime (e.g., torch_distributed) from `config/local_runtimes`.
 - pull_policy: Controls image pulling. Supported values: "IfNotPresent",
   "Always", "Never". The default is "IfNotPresent".
 - auto_remove: Whether to remove containers and networks when jobs are deleted.
   Defaults to True.
 - gpus: GPU support using NVIDIA CDI or other device plugins. Defaults to None.
 - env: Optional global environment variables applied to all containers.
 - podman_url: Optional override for connecting to a remote/local Podman socket.
   By default, the Podman client resolves from environment variables or uses
   the system default socket (e.g., unix:///run/user/1000/podman/podman.sock
   for rootless, unix:///run/podman/podman.sock for rootful).
 - workdir_base: Base directory on the host to place per-job working dirs that
   are bind-mounted into containers as /workspace. Defaults to a path under the
   user's home directory for compatibility.
"""

from typing import Optional, Union

from pydantic import BaseModel, Field


class LocalPodmanBackendConfig(BaseModel):
    image: Optional[str] = Field(default=None)
    pull_policy: str = Field(default="IfNotPresent")
    auto_remove: bool = Field(default=True)
    gpus: Optional[Union[int, bool]] = Field(default=None)
    env: Optional[dict[str, str]] = Field(default=None)
    podman_url: Optional[str] = Field(default=None)
    workdir_base: Optional[str] = Field(default=None)
