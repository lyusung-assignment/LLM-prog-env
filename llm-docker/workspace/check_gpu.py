import torch, platform, transformers
print("python      :", platform.python_version())
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("cuda avail  :", torch.cuda.is_available())
print("cuda ver    :", torch.version.cuda)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}")
