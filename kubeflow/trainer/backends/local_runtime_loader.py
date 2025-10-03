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
Shared runtime loader for local container backends (Docker, Podman).

We support loading local runtime definitions from
`kubeflow/trainer/config/local_runtimes/` (YAML files). The schema mirrors the
essential fields from the upstream CRD manifest but is tailored for local
container backends to capture the container image and trainer characteristics
needed to construct a `types.Runtime` object.

We ship a built-in `torch_distributed.yaml` as the default runtime. Users can
add additional YAML files to the same directory to define custom local runtimes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kubeflow.trainer.types import types as base_types

LOCAL_RUNTIMES_DIR = Path(__file__).parents[1] / "config" / "local_runtimes"


def _load_runtime_from_yaml(path: Path) -> dict[str, Any]:
    with open(path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def list_local_runtimes() -> list[base_types.Runtime]:
    runtimes: list[base_types.Runtime] = []
    if not LOCAL_RUNTIMES_DIR.exists():
        return runtimes

    for f in sorted(LOCAL_RUNTIMES_DIR.glob("*.yaml")):
        data = _load_runtime_from_yaml(f)

        # Require CRD-like schema strictly. Accept both ClusterTrainingRuntime
        # and TrainingRuntime kinds.
        if not (
            data.get("kind") in {"ClusterTrainingRuntime", "TrainingRuntime"}
            and data.get("metadata")
        ):
            raise ValueError(
                f"Runtime YAML {f} must be a ClusterTrainingRuntime CRD-shaped document"
            )

        name = data["metadata"].get("name")
        if not name:
            raise ValueError(f"Runtime YAML {f} missing metadata.name")

        labels = data["metadata"].get("labels", {})
        framework = labels.get("trainer.kubeflow.org/framework")
        if not framework:
            raise ValueError(
                f"Runtime {name} must set metadata.labels['trainer.kubeflow.org/framework']"
            )

        spec = data.get("spec", {})
        ml_policy = spec.get("mlPolicy", {})
        num_nodes = int(ml_policy.get("numNodes", 1))

        # Validate presence of a 'node' replicated job with a container image
        templ = spec.get("template", {}).get("spec", {})
        replicated = templ.get("replicatedJobs", [])
        node_jobs = [j for j in replicated if j.get("name") == "node"]
        if not node_jobs:
            raise ValueError(f"Runtime {name} must define replicatedJobs with a 'node' entry")
        node_spec = (
            node_jobs[0].get("template", {}).get("spec", {}).get("template", {}).get("spec", {})
        )
        containers = node_spec.get("containers", [])
        if not containers or not containers[0].get("image"):
            raise ValueError(f"Runtime {name} 'node' must specify containers[0].image")

        runtimes.append(
            base_types.Runtime(
                name=name,
                trainer=base_types.RuntimeTrainer(
                    trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                    framework=framework,
                    num_nodes=num_nodes,
                ),
                pretrained_model=None,
            )
        )
    return runtimes


def get_local_runtime(name: str) -> base_types.Runtime:
    for rt in list_local_runtimes():
        if rt.name == name:
            return rt
    raise ValueError(f"Runtime '{name}' not found in {LOCAL_RUNTIMES_DIR}")
