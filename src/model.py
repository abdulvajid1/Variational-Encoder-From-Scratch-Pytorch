import torch.nn as nn
import torch
from typing import Annotated,Tuple
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device set to {device}")

class Encoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.encoder_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )
        
        self.infeature_latent = 33856 # precomputed before
        
        self.latent_mean = nn.Linear(self.infeature_latent, self.latent_dim)
        self.latent_log_var = nn.Linear(self.infeature_latent, self.latent_dim)
        
    def forward(self, x):
        x = self.encoder_block(x)
        latent_mean = self.latent_mean(x)
        latent_log_var = self.latent_log_var(x)
        return latent_mean, latent_log_var
    
    
        
def sampling(latent_mean, latent_log_var):
    epilson = torch.randn_like(latent_mean, device=device)
    std = torch.exp(0.5 * latent_log_var)
    return latent_mean + std * epilson
        

class Decoder(nn.Module):
    def __init__(self, latent_dim:int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_proj_dim = 33856
        self.latent_proj = nn.Linear(in_features=self.latent_dim, out_features=self.latent_proj_dim)
        
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=(3,3), stride=2, output_padding=1),
            nn.Sigmoid()
        )   
        
    def forward(self, z: torch.Tensor):
        print(z.shape)
        z = self.latent_proj(z)
        print(z.shape)
        z = z.contiguous().view(-1, 64, 23, 23) # precomputed
        z = self.decoder_block(z)
        return z
        

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self. decoder = decoder 
    
    def forward(self, x, original_x):
        latent_mean, latent_log_var = self.encoder(x)
        z = sampling(latent_mean, latent_log_var)
        z = self.decoder(z)
        z_flatten = torch.flatten(z)
        original_x_flatten = torch.flattent(original_x)
        loss = F.binary_cross_entropy(z_flatten, original_x_flatten)
        
        return z, loss
