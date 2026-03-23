from dataclasses import dataclass
import math
import torch
import os

@dataclass(frozen=True)
class ExperimentConfig:
    # Admin
    run_spectral: bool = True
    test_mode: bool = False  # If True, saves to a TEST subdir and uses a simplified config for quick runs
    toy: bool = False  # If True, uses a toy MLP architecture for quick runs

    # Data
    dataset: str = "MNIST"
    download_directory: str =  "/gpfs/scratch/qp252676/globus/grokking/data"

    train_points: int = 100
    batch_size: int = 100

    # Model
    input_dim: int = 784
    output_dim: int = 10
    depth: int = 3
    width: int = 200
    activation: str = "ReLU"
    initialization_scale: float = 1.0

    # Spectral analysis
    spectral_every: int = 1000
    spectral_m: int = 90
    spectral_k: int = 100
    spectral_sigma: float = 0.01

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
    log_every: int = 150
    eval_every: int = 1000
    output_dir: str = os.environ.get("OUTPUT_DIR", "./runs")
    seed: int = 0
    dtype: torch.dtype = torch.float32

    @property
    def log_freq(self) -> int:
        return math.ceil(self.optimization_steps / self.log_points)