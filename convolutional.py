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


### Convolutional Neural Network

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



## training loop 

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)


## Epoch means one pass through the dataset

## training section
num_epochs = 20

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
# Epoch [1/20], Loss: 0.0008
# Epoch [2/20], Loss: 0.0003
# Epoch [3/20], Loss: 0.0004
# Epoch [4/20], Loss: 0.0005
# Epoch [5/20], Loss: 0.0003
# Epoch [6/20], Loss: 0.0003
# Epoch [7/20], Loss: 0.0001
# Epoch [8/20], Loss: 0.0002
# Epoch [9/20], Loss: 0.0002
# Epoch [10/20], Loss: 0.0001
# Epoch [11/20], Loss: 0.0001
# Epoch [12/20], Loss: 0.0000
# Epoch [13/20], Loss: 0.0002
# Epoch [14/20], Loss: 0.0002
# Epoch [15/20], Loss: 0.0000
# Epoch [16/20], Loss: 0.0000
# Epoch [17/20], Loss: 0.0001
# Epoch [18/20], Loss: 0.0001
# Epoch [19/20], Loss: 0.0001
# Epoch [20/20], Loss: 0.0001


## Now using a convolutioanl neural network with more epochs
## CNN better for classifying images.
