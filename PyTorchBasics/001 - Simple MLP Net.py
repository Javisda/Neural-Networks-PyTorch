# Imports
import torch
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
import torchvision.datasets as datasets  # Standard datasets, like MNIST dataset
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation


#############################################################################################################

# Fully Connected Network
class FCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 50)
        self.fc3 = nn.Linear(50, num_classes)
        self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data
# Ponemos una ruta en la que va a descargar el set de datos y ponemos que dicho set se va a usar para el entrenamiento
train_dataset = datasets.MNIST(root='/dataset', train=True, transform=transforms.ToTensor(), download=True)
# Carga el set indicado y lo prepara en batches para poder ser procesados por el programa
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='/dataset', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model = FCN(input_size=input_size, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):

    # Batch index es devuelto gracias al enumerate, y básicamente nos devuelve el indice del batch que estamos recorriendo
    for data, targets in iter(train_loader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        # print(data.shape) # [64, 1, 28, 28]

        # Get correct shape
        data = data.reshape(data.shape[0], -1)
        # print(data.shape) # [64, 728]

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient Descent
        optimizer.step()


# Check Accuracy
def check_accuracy(loader, model):
    if(loader.dataset.train):
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # Queremos los valores máximos de la dimension 2º [64, 10]
            _, predictions = scores.max(1)
            # Sumará un numero correcto aquel cuya prediccion concuerde con y (valor real)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)