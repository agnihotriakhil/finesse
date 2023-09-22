import torch
import torch.nn as nn
import torch.nn.functional as F


class CVAE(nn.Module):
    def __init__(self, image_dim, text_dim, latent_dim, image_channels):
        super(CVAE, self).__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_flattened_dim = self.image_channels*self.image_dim*self.image_dim

        # Encoder
        self.fc1 = nn.Linear(self.image_flattened_dim + self.text_dim, 512)
        self.fc_mu = nn.Linear(512, self.latent_dim)
        self.fc_logvar = nn.Linear(512, self.latent_dim)

        # Decoder
        self.fc2 = nn.Linear(self.latent_dim + self.text_dim, 512)
        self.fc3 = nn.Linear(512, self.image_flattened_dim)

    def encode(self, image, text):

        image = image.view(-1, self.image_channels*self.image_dim*self.image_dim)
        text  = text.view(-1, self.text_dim)

        x = torch.cat([image, text], dim=1)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, text):
        text = text.squeeze(dim=1) 
        x = torch.cat([z, text], dim=1)
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

    def forward(self, image, text):
        mu, logvar = self.encode(image, text)
        z = self.reparameterize(mu, logvar)
        reconstructed_image = self.decode(z, text)
        return reconstructed_image, mu, logvar
