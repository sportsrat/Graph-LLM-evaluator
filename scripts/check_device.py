"""
Hardware and PyTorch device diagnostic script for Windows / Cross-platform.
"""

import sys
import torch


def main() -> None:
    print("=" * 60)
    print("PyTorch Environment & Hardware Diagnostic")
    print("=" * 60)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available : {cuda_available}")
    if cuda_available:
        print(f"CUDA device count : {torch.cuda.device_count()}")
        print(f"Current device name: {torch.cuda.get_device_name(0)}")
        dev = torch.device("cuda:0")
        x = torch.randn(2, 2, device=dev)
        y = torch.mm(x, x)
        print("CUDA Tensor Test   : SUCCESS")
    else:
        print("Running on     : CPU")
        x = torch.randn(2, 2)
        y = torch.mm(x, x)
        print("CPU Tensor Test: SUCCESS")

    print("=" * 60)


if __name__ == "__main__":
    main()
