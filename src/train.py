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
    train_dataset = torch.load("D:\Machine Learning\PROJECTS\Variational-Encoder-From-Scratch-Pytorch\\notebooks\data\\test_data.pt", weights_only=False)
    test_dataset = torch.load("D:\Machine Learning\PROJECTS\Variational-Encoder-From-Scratch-Pytorch\\notebooks\data\\train_data.pt", weights_only=False)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader

def get_optimizer(model: VariationalAutoEncoder, lr:float):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    return optimizer

def evaluate(model, test_dataloader ,num_limit_step: int, device):
    model.to(device)
    with torch.no_grad():
        total_loss = 0
        for step, (inp_img, labels)  in enumerate(test_dataloader):
            inp_img = inp_img.to(device)
            _, loss = model(inp_img)
            total_loss += loss.item()
            
            if step == num_limit_step:
                break
    
    avg_loss = total_loss / num_limit_step
    return avg_loss
    
    
def train(latent_dim:int, epochs: int, batch_size:int, evaluate_step: int, evaluation_step_limit: int=20, lr: float=1e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VariationalAutoEncoder(latent_dim=latent_dim, kl_div_beta=0.90).to(device)
    train_dataloader, test_dataloader = get_dataloader(batch_size)
    optimizer  = get_optimizer(model, lr=lr)
    
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm.tqdm(train_dataloader, dynamic_ncols=True)
        progress_bar.set_description(f'Training Epoch: {epoch}')
        total_epoch_loss = 0
        
        for step, (input_img, labels) in enumerate(progress_bar):
           input_img = input_img.to(device)
           reconstructed_x, loss = model(input_img)
           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
           
           total_epoch_loss += loss.item()
           
           progress_bar.set_postfix({"step": step, "loss": loss.item()})
           if step % evaluate_step == 0:
               model.eval()
               loss = evaluate(model, test_dataloader, num_limit_step=evaluation_step_limit, device=device)
               tqdm.tqdm.write(f"Evaluation Loss : {loss}")
               model.train()
        
        tqdm.tqdm.write(f'Epoch {epoch} Loss: {total_epoch_loss / len(train_dataloader)}')
               

if __name__=="__main__":
    config ={
        "latent_dim": 512,
        "epochs":10,
        "batch_size":4,
        "evaluate_step": 100,
        "evaluation_step_limit":20,
        "lr":1e-5
    }
    
    train(**config)
            
            
               
            
            
           
            
        
    
    