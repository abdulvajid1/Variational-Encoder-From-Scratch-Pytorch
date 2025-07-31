import torch
import torch.nn as nn
import torch.nn.functional as F


def build_model(latent_dim:int, hidden_dim:int)