import torch

print("=== PyTorch & CUDA Environment Check ===")

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version (compiled):", torch.version.cuda)
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU index:", torch.cuda.current_device())
    print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Simple CUDA tensor test
    x = torch.randn(2, 3).cuda()
    y = x * 2
    print("CUDA tensor test: OK")
else:
    print("CUDA not available, using CPU")
