import torch.nn as nn
import torch
from typing import Annotated,Tuple
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f"Device set to {device}")


# def precompute_flatten_dim(func, device):
#     with torch.no_grad():
#         dummy_input = torch.rand(size=(1, 3, 96, 96)).to(device)
#         func.to(device)
#         x = func(dummy_input)
#         x = torch.flatten(x, start_dim=1)
#         return x.shape[-1]


class Encoder(nn.Module):
    def __init__(self, latent_dim: int, device):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.device = device
        
        self.encoder_block = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten(start_dim=1)
        )
        
        self.infeature_latent = self.precompute_flatten_dim(device) # precomputed before
        
        self.latent_mean = nn.Linear(self.infeature_latent, self.latent_dim)
        self.latent_log_var = nn.Linear(self.infeature_latent, self.latent_dim)
        
    def forward(self, x: torch.Tensor):
        x = self.encoder_block(x)
        latent_mean = self.latent_mean(x)
        latent_log_var = self.latent_log_var(x)
        return latent_mean, latent_log_var
    
    def precompute_flatten_dim(self, device):
        with torch.no_grad():
            dummy_input = torch.rand(size=(1, 1, 28, 28)).to(device)
            self.encoder_block.to(device)
            x = self.encoder_block(dummy_input)
            x = torch.flatten(x, start_dim=1)
            return x.shape[-1]
    
        

class Decoder(nn.Module):
    def __init__(self, latent_dim:int, latent_proj_dim:int, device):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        self.decoder_block = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=(3,3), stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(3,3), stride=2, output_padding=1),
            nn.Sigmoid()
        )
        
        self.latent_proj_dim = latent_proj_dim
        self.latent_proj = nn.Linear(in_features=self.latent_dim, out_features=self.latent_proj_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.Linear(1024, latent_dim)
            )   
        
    def forward(self, z: torch.Tensor):
        z = self.proj(z)
        z = self.latent_proj(z)
        
        z = z.contiguous().view(-1, 32, 6, 6) # precomputed
        z = self.decoder_block(z)
        return z
        

class VariationalAutoEncoder(nn.Module):
    def __init__(self,latent_dim:int, kl_div_beta: float, device):
        super().__init__()
        self.encoder = Encoder(latent_dim, device=device)
        self.latent_proj_dim = self.encoder.infeature_latent # latent_dim -> "falatten_size" -> back to img
        self.decoder = Decoder(latent_dim, self.latent_proj_dim, device=device)
        self.kl_div_beta = kl_div_beta
        self.device = device
        
    
    def forward(self, x):
        original_x = x
        latent_mean, latent_log_var = self.encoder(x)
        z_sampled, latent_std, latent_mean = self.sampling(latent_mean, latent_log_var)
        reconstructed_x = self.decoder(z_sampled)
        assert original_x.shape == reconstructed_x.shape, "shape mismatch between orginal vs reconstructed images"
        
        
        reconstruction_loss = F.binary_cross_entropy(reconstructed_x, original_x, reduction='sum')
        kl_div_loss = - torch.sum(1+ torch.log(latent_std) - torch.pow(latent_mean, 2) - torch.pow(latent_std, 2))
        loss = reconstruction_loss + self.kl_div_beta * kl_div_loss
        reconstructed_x = reconstructed_x.view(-1, 1, 28, 28)        
        return reconstructed_x, loss
        
    
    def sampling(self, latent_mean, latent_log_var):
        epilson = torch.randn_like(latent_mean, device=self.device)
        latent_std = torch.exp(0.5 * latent_log_var)
        sampled_z = latent_mean + latent_std * epilson
        return sampled_z, latent_std, latent_mean