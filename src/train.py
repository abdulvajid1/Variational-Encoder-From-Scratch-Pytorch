import torch
import torch.nn as nn
import torch.nn.functional as F
from model import VariationalAutoEncoder
from torchvision import transforms
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from torchvision.datasets import STL10

def get_dataloader(batch_size):
    dataset = STL10('data', split='train', transform=transforms.ToTensor(), download=False)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_optimizer(model: VariationalAutoEncoder, lr:float, weight_decay: float):
  
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)
    return optimizer
    

def train(latent_dim:int, epochs: int, batch_size:int):
    device = torch.device('cuda' if torch.cuda.is_available())
    model = VariationalAutoEncoder(latent_dim=latent_dim)
    dataloader = get_dataloader(batch_size)
    optimizer  = get_optimizer(model)
    
    for epoch in range(epochs):
        
        for step, input_img in enumerate(dataloader):
           input_img = input_img.to(device)
           reconstructed_x, loss = model(input_img)
           optimizer.zero_grad()
           loss.backwards()
           optimizer.step()
           
            
        
    
    