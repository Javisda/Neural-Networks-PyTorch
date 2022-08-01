# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        # Si hacemos print(model) vemos que las capas que necesitamos son las de los numeros que hemos puesto en el array de abajo
        self.chosen_features = ["0", "5", "10", "19", "28"]  # Conv 1-1, Conv 2-1, Conv 3-1, Conv 4-1, Conv 5-1.
        self.model = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features[:29] # Nos quedamos con la parte de la arquitectura que necesitamos (hasta la 28)

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            x = layer(x)

            if str(layer_num) in self.chosen_features:
                features.append(x)

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0) # Adds an aditional dimention for the batch size
    return image.to(device=device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
image_size = 356

loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[], std=[])
])

# TODO: Solucionar lo expuesto abajo
# DE MOMENTO FUNCIONA SOLO CON IMAGENES CON UNA PROFUNDIDAD DE BITS DE 24
original_img = load_image("javiPlayero.jpg")
style_img = load_image("hokusai.jpg")

# generated = torch.randn(original_img.shape, device=device, requires_grad=True)
generated = original_img.clone().requires_grad_(True)


# Hyperparameters
total_steps = 5000
learning_rate = 0.005
alpha = 1
beta = 1e-4
optimizer = optim.Adam([generated], lr=learning_rate) # De normal pasariamos model.parameters(),
# pero la imagen generada es lo que el optimizador tiene que tratar de optimizar en este caso

model = VGG().to(device=device)
model.eval() # Freeze weights

for step in range(total_steps + 1):
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    # Loss is 0 initially
    style_loss = content_loss = 0

    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
    ):
        #print(gen_feature.shape)
        batch_size, channel, height, width = gen_feature.shape # Estas dimensiones cambiarán dependiendo de que bloque de convolucion mire

        # Content Loss
        content_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Style Loss
        # gram matrix
        G = gen_feature.view(channel, height*width).mm(
            gen_feature.view(channel, height*width).t()
        )

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t()
        )

        style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha*content_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 1000 == 0 and step != 0:
        print(total_loss)
        print("Current step: " + str(step))
        save_image(generated, 'generated'+ str(step)+'.png')
