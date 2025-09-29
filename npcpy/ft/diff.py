# finetuning diffuser models
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset
    from transformers import CLIPTextModel, CLIPTokenizer
except:
    torch = None
    nn = None    
    F = None
    DataLoader = None
    TorchDataset = None
    CLIPTextModel = None
    CLIPTokenizer = None
import math
from dataclasses import dataclass, field
from typing import List, Optional, Callable
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import gc


@dataclass
class DiffusionConfig:
    image_size: int = 128
    channels: int = 256
    time_emb_dim: int = 128
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_epochs: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-5
    checkpoint_frequency: int = 1000
    output_dir: str = "diffusion_model"
    use_clip: bool = True
    num_channels: int = 1


class SinusoidalPositionEmbeddings(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings
        )
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat(
            (embeddings.sin(), embeddings.cos()),
            dim=-1
        )
        return embeddings


class SimpleUNet(nn.Module):
    
    def __init__(
        self,
        image_size=128,
        channels=256,
        time_emb_dim=128,
        num_channels=1
    ):
        super().__init__()
        
        self.image_size = image_size
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, channels),
        )
        
        self.text_mlp = nn.Sequential(
            nn.Linear(768, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, channels),
        )
        
        self.conv_in = nn.Conv2d(num_channels, channels, 1, padding=0)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 4, 2, 1),
            nn.GroupNorm(8, channels * 2),
            nn.GELU(),
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1),
            nn.GroupNorm(8, channels * 4),
            nn.GELU(),
        )
        
        self.down3 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 8, 4, 2, 1),
            nn.GroupNorm(8, channels * 8),
            nn.GELU(),
        )