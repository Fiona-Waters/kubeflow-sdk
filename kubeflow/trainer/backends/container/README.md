# ContainerBackend

The unified container backend for Kubeflow Trainer that automatically detects and uses either Docker or Podman.

## Overview

This backend provides a single, unified interface for container-based training execution, automatically detecting which container runtime is available on your system.

The implementation uses the **adapter pattern** to abstract away differences between Docker and Podman APIs, providing clean separation between runtime detection logic and container operations.

## Usage

### Basic usage (auto-detection)

```python
from kubeflow.trainer import TrainerClient, ContainerBackendConfig

# Auto-detects Docker or Podman
config = ContainerBackendConfig()
client = TrainerClient(backend_config=config)
```

### Force specific runtime

```python
# Force Docker
config = ContainerBackendConfig(runtime="docker")
client = TrainerClient(backend_config=config)

# Force Podman
config = ContainerBackendConfig(runtime="podman")
client = TrainerClient(backend_config=config)
```

### Configuration options

```python
config = ContainerBackendConfig(
    # Optional: force specific runtime ("docker" or "podman")
    runtime=None,

    # Optional: explicit image override
    image="my-custom-image:latest",

    # Image pull policy: "IfNotPresent", "Always", or "Never"
    pull_policy="IfNotPresent",

    # Auto-remove containers and networks on job deletion
    auto_remove=True,

    # GPU support (varies by runtime)
    gpus=None,

    # Environment variables for all containers
    env={"MY_VAR": "value"},

    # Container daemon URL override (required for Colima/Podman Machine on macOS)
    container_host=None,

    # Base directory for job workspaces
    workdir_base=None,
)
```

### macOS-specific configuration

On macOS, you may need to specify `container_host` depending on your container runtime:

**Docker with Colima:**
```python
import os
config = ContainerBackendConfig(
    container_host=f"unix://{os.path.expanduser('~')}/.colima/default/docker.sock"
)
```

**Podman Machine:**
```python
import os
config = ContainerBackendConfig(
    container_host=f"unix://{os.path.expanduser('~')}/.local/share/containers/podman/machine/podman.sock"
)
```

**Docker Desktop:**
```python
# Usually works without specifying container_host
config = ContainerBackendConfig()
```

Alternatively, set environment variables before running:
```bash
# For Colima
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"

# For Podman Machine
export CONTAINER_HOST="unix://$HOME/.local/share/containers/podman/machine/podman.sock"
```

### How it works

The backend initialization follows this logic:

1. If `runtime` is specified in config, use that runtime exclusively
2. Otherwise, try to initialize Docker client adapter
3. If Docker fails, try to initialize Podman client adapter
4. If both fail, raise a RuntimeError

If you don't have Docker or Podman installed, use `LocalProcessBackendConfig` instead, which runs training as local subprocesses.

All container operations are delegated to the adapter, eliminating code duplication.

## Installation

Install with Docker support:
```bash
pip install kubeflow[docker]
```

Install with Podman support:
```bash
pip install kubeflow[podman]
```

Install with both:
```bash
pip install kubeflow[docker,podman]
```

## Example: Training Job

```python
from kubeflow.trainer import TrainerClient, ContainerBackendConfig, CustomTrainer

# Define your training function
def train():
    import torch
    print(f"Training with PyTorch {torch.__version__}")
    # Your training code here

# Create trainer
trainer = CustomTrainer(
    func=train,
    packages_to_install=["torch"],
)

# Initialize client (auto-detects runtime)
config = ContainerBackendConfig()
client = TrainerClient(backend_config=config)

# Run training
job_name = client.train(trainer=trainer)
print(f"Training job started: {job_name}")

# Get logs
for log in client.get_job_logs(job_name, follow=True):
    print(log, end='')
```

## See also

- [Example notebook](TBA) - Complete working example to be added
