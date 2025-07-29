import torch.nn as nn
import torch
from typing import Annotated,Tuple

device = torch.device('cuda' if torch.cuda.isavailable() else 'cpu')
print(f"Device set to {device}")

class Encoder(nn.Module):
    def __init__(self, batch_size: int,
                 image_size:Annotated[int, "the image should be same width and height"],
                 latent_space:int):
        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size 
        self.latent_space = latent_space
        self.image_size = (self.image_size - 2) / 4 # After convolution
        
        # last channel size * height * width
        self.ln_infeature = int(64 * image_size * image_size)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 62, kernel_size=(3,3), stride=2)
        self.batch_norm2 = nn.BatchNorm2d(62)
        
        # (b, channel, height, width)
        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.LeakyReLU()
        
        self.latent_mean = nn.Linear(self.ln_infeature, out_features=self.latent_space)
        self.latent_log_var = nn.Linear(self.ln_infeature, out_features=self.latent_space)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.flatten(x)
        latent_mean = self.latent_mean(x)
        latent_log_var = self.latent_log_var(x)
        sampling(latent_mean, latent_log_var)
        
        return latent_mean, latent_log_var
        
def sampling(latent_mean, latent_log_var):
    batch = latent_mean.shape[0]
    dim = latent_mean.shape[1]
    epilson = torch.randn(size=(batch, dim), device=device)
    std = torch.exp(0.5 * latent_log_var)
    return latent_mean + std * epilson
        

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose_1 = nn.ConvTranspose2d(62, 32, kernel_size=(3,3), stride=2)
        self.conv_transpose_2 = nn.ConvTranspose2d(32, 3, kernel_size=(3,3), stride=2)
        

class VariationalAutoEncoder(nn.Module):
    pass
    
    