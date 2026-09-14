"""Lab 1 environment check - mirrors the script from Lab1_VSCode.pdf (slide 23).

Run it against the lab01 environment and compare the output line by line with
the course reference machine:

    docker compose exec llm conda run -n lab01 python /workspace/lab01/check_env.py
"""

import platform
import sys

import accelerate
import datasets
import torch
import transformers


def main() -> None:
    print("=" * 60)
    print("LLM Development Environment")
    print("=" * 60)

    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {sys.version.split()[0]}")
    print(f"Operating system  : {platform.platform()}")
    print(f"PyTorch           : {torch.__version__}")
    print(f"Transformers      : {transformers.__version__}")
    print(f"Datasets          : {datasets.__version__}")
    print(f"Accelerate        : {accelerate.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")
    print(f"PyTorch CUDA      : {torch.version.cuda}")

    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU               : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory        : {properties.total_memory / 1024**3:.2f} GB")

        x = torch.rand((1024, 1024), device="cuda")
        y = x @ x
        print(f"GPU tensor test   : OK, {tuple(y.shape)}")
    else:
        print("GPU tensor test   : FAILED")

    print("=" * 60)


if __name__ == "__main__":
    main()
