import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms



## check device for GPU else uses CPU.
device = torch.device('cud' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


## loading and transforming MNIST dataset
transform_mnist = transforms.Compose([
    transforms.ToTensor(),   ## converts images to PyTorch tensors, where pixel values go from [0,255] to [0,1].
    transforms.Normalize((0.1307,), (0.3081,)) ## (mean, std) standardises pixel values. For MNIST, mean ≈ 0.1307 and std ≈ 0.3081. This is optional but helps in training stability.
])

train_dataset_mnist = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform_mnist
)

test_datasetmnist = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform_mnist
)
