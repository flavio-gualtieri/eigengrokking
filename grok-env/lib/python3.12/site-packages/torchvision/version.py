__version__ = '0.25.0+cu128'
git_version = '8ac84ee75afb1c327902156b5336f56ad63b7e2f'
from torchvision.extension import _check_cuda_version
if _check_cuda_version() > 0:
    cuda = _check_cuda_version()
