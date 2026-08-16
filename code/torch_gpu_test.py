import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
else:
    print("No CUDA/ROCm device available. Maybe torch was installed wrong.")
