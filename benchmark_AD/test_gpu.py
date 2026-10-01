import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # run a small tensor op on GPU to confirm it works
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Tensor device: {x.device}")
    print("GPU is working correctly")
else:
    print("Running on CPU")