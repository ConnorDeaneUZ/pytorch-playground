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

test_dataset_mnist = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform_mnist
)


## data laoders

## batch size = number of images passed at once
## shuffle = model sees data in random order for better training.


train_loader_mnist = torch.utils.data.DataLoader(
    train_dataset_mnist,
    batch_size=64,
    shuffle=True
)

test_loader_mnist = torch.utils.data.DataLoader(
    test_dataset_mnist,
    batch_size=1000,
    shuffle=False
)


## MNIST is 28x28 - 784 input features
class SimpleMNISTModel(nn.Module):
    def __init__(self):
        super(SimpleMNISTModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

## Flatten image batch (N, 1, 28, 28) into (N, 784)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) ## final linear layer
        return x



## training loop 

model = SimpleMNISTModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


## Epoch means one pass through the dataset

## training section
num_epochs = 5

for epoch in range(num_epochs):
    model.train() # set model to training mode
    running_loss = 0.0

    for images, labels in train_loader_mnist:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss =+ loss.item()

    epoch_loss = running_loss / len(train_loader_mnist)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")


## switches to evaluation mode. this then runs the same test on the dataset to gather the accuracy.
model.eval() 

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader_mnist:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")



# Using device: cpu
# Epoch [1/5], Loss: 0.0022
# Epoch [2/5], Loss: 0.0013
# Epoch [3/5], Loss: 0.0007
# Epoch [4/5], Loss: 0.0006
# Epoch [5/5], Loss: 0.0008
# Accuracy on test set: 86.91%


## we are using a cpu as we are on a macbook pro.
## The tests ran for 5 full rotations of the data 
## the 'loss' decrased over time which means the model is learning to classify images more accurately.

