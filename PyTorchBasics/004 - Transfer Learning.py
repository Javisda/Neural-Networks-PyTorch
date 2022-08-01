# Imports
import torch
import torchvision.models
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
import torchvision.datasets as datasets  # Standard datasets, like MNIST dataset
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation


#############################################################################################################

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Hyperparameters
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5

# Load pretrain model & modify it. Vamos a quitarle la avgpool que hay entre las features y el classifier. Además, la salida será de 10 en vez de 1000
# Creamos una clase de apoyo para poder sustituir la avgpool
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

# Pretrained = True está desfasado. En cambio, el compilador nos recomienda weights='VGG16_Weights.IMAGENET1K_V1'
model = torchvision.models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1')
# Congela todos los pesos de la red.
for param in model.parameters():
    param.requires_grad = False

# Pero a partir de aqui, modificamos una parte de la red, el clasificador, por lo que los pesos congelados solo se aplican a las capas no modificadas, y nuestro clasificador
# personalizado si se entrenará
print(model)
model.avgpool = Identity()
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
# De la manera de arriba hemos sustituido al completo el clasificador. En cambio, si solo queremos cambiar una capa en concreto, podemos hacer lo siguiente
#model.classifier[6] = nn.Linear(4096, 10)
print(model)

model.to(device)



# Load Data
# Ponemos una ruta en la que va a descargar el set de datos y ponemos que dicho set se va a usar para el entrenamiento
train_dataset = datasets.CIFAR10(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
# Carga el set indicado y lo prepara en batches para poder ser procesados por el programa
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
def training():
    for epoch in range(num_epochs):

        # Batch index es devuelto gracias al enumerate, y básicamente nos devuelve el indice del batch que estamos recorriendo
        for data, targets in iter(train_loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

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

            scores = model(x)
            # Queremos los valores máximos de la dimension 2º [64, 10]
            _, predictions = scores.max(1)
            # Sumará un numero correcto aquel cuya prediccion concuerde con y (valor real)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

training()
check_accuracy(train_loader, model)