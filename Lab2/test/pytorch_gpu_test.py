import torch

# Kontrollera versionen
print(torch.__version__)

# Kontrollera om ROCm är tillgängligt
print(torch.cuda.is_available())

# Räkna antalet ROCm-enheter (GPU:er)
print(torch.cuda.device_count())