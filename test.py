import torch

# Print PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check if GPU acceleration is available
print(f"CUDA Available: {torch.cuda.is_available()}")

# Create a sample tensor
x = torch.rand(2, 3)
print("Sample Tensor:\n", x)
