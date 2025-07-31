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
    latent_std = torch.exp(0.5 * latent_log_var)
    sampled_z = latent_mean + latent_std * epilson
    return sampled_z, latent_std, latent_mean
        

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
        z = self.latent_proj(z)
        z = z.contiguous().view(-1, 64, 23, 23) # precomputed
        z = self.decoder_block(z)
        return z
        

class VariationalAutoEncoder(nn.Module):
    def __init__(self,latent_dim:int, kl_div_beta: float):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self. decoder = Decoder(latent_dim=latent_dim)
        self.kl_div_beta = kl_div_beta
    
    def forward(self, x):
        original_x = x
        latent_mean, latent_log_var = self.encoder(x)
        z_sampled, latent_std, latent_mean = sampling(latent_mean, latent_log_var)
        reconstructed_x = self.decoder(z_sampled)
        assert original_x.shape != reconstructed_x, "shape mismatch between orginal vs reconstructed images"
        reconstruction_loss = F.binary_cross_entropy(reconstructed_x, original_x, reduction='sum')
        kl_div_loss = - torch.sum(1+ torch.log(latent_std) - torch.pow(latent_mean, 2) - torch.pow(latent_std, 2))
        loss = reconstruction_loss + self.kl_div_beta * kl_div_loss
        reconstructed_x = reconstructed_x.view(-1, 3, 96, 96)        
        return reconstructed_x, loss
