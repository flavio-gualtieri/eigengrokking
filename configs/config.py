from dataclasses import dataclass
import math
import torch
import os

@dataclass(frozen=True)
class ExperimentConfig:
    # Admin
    run_spectral: bool = True
    hpc: bool = False  # Whether we're running on HPC (forces CPU device)
    test_mode: bool = False  # If True, saves to a TEST subdir and uses a simplified config for quick runs

    # Data
    dataset: str = "MNIST"
    modulus: int = 10  # Only used for MODULAR dataset
    download_directory: str = "."
    train_points: int = 100
    batch_size: int = 100

    # Model
    input_dim: int = 784
    output_dim: int = 10
    depth: int = 3
    width: int = 200
    activation: str = "ReLU"
    initialization_scale: float = 1.0

    # Optimization
    optimization_steps: int = 200_000
    optimizer: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0
    loss_function: str = "MSE"

    # Adversarial eval
    fgsm_epsilon: float = 0.1
    fgsm_refresh_step: int = 20_000

    # Logging/saving
    log_points: int = 150
    save_every: int = 1000
    spectral_every: int = 1000
    output_dir: str = os.environ.get("OUTPUT_DIR", "./runs")
    seed: int = 0
    dtype: torch.dtype = torch.float32

    # Spectral analysis
    spectral_every: int = 1000          # run spectral snapshot every N steps
    spectral_m: int = 90                # Lanczos iterations
    spectral_k: int = 100               # number of probe vectors
    spectral_sigma: float = 0.01        # smoothing parameter
    spectral_t_min: float = -1.0        # spectrum grid lower bound
    spectral_t_max: float = 15.0        # spectrum grid upper bound
    spectral_t_points: int = 400        # number of grid points
    spectral_probe_batch_size: int = 200

    @property
    def log_freq(self) -> int:
        return math.ceil(self.optimization_steps / self.log_points)