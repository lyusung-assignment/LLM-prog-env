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
    print(f"Pythonexecutable : {sys.executable}")
    print(f"Pythonversion : {sys.version.split()[0]}")
    print(f"Operatingsystem : {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Datasets: {datasets.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"CUDAavailable : {torch.cuda.is_available()}")
    print(f"PyTorchCUDA : {torch.version.cuda}")

    if torch.cuda.is_available():
        gpu_name= torch.cuda.get_device_name(0)
        properties = torch.cuda.get_device_properties(0)
        memory_gb= properties.total_memory/ 1024**3

        print(f"GPU: {gpu_name}")
        print(f"GPUmemory : {memory_gb:.2f} GB")

        x = torch.rand((1024, 1024), device="cuda")
        y = x @ x

        print(f"GPUtensor test : OK, {tuple(y.shape)}")
    else:
        print("GPU tensor test : FAILED")
        print("=" * 60)

if __name__ == "__main__":
    main()