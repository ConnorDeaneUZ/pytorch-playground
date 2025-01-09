import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms



## check device for GPU else uses CPU.
device = torch.device('cud' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
