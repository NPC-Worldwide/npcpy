try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset as TorchDataset

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None    
    F = None
    DataLoader = None
    TorchDataset = None
    CLIPTextModel = None
    CLIPTokenizer = None
    TORCH_AVAILABLE = False

import math
from dataclasses import dataclass, field
from typing import List, Optional
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
    checkpoint_frequency: int = 10
    output_model_path: str = "diffusion_model"
    use_clip: bool = False
    num_channels: int = 3


if TORCH_AVAILABLE:
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
        def __init__(self, image_size=128, channels=256, 
                     time_emb_dim=128, num_channels=3):
            super().__init__()
            self.image_size = image_size
            
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim * 4),
                nn.GELU(),
                nn.Linear(time_emb_dim * 4, channels),
            )
            
            self.conv_in = nn.Conv2d(num_channels, channels, 3, padding=1)
            
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
            
            self.mid = nn.Sequential(
                nn.Conv2d(channels * 4, channels * 4, 3, 1, 1),
                nn.GroupNorm(8, channels * 4),
                nn.GELU(),
            )
            
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1),
                nn.GroupNorm(8, channels * 2),
                nn.GELU(),
            )
            
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(channels * 4, channels, 4, 2, 1),
                nn.GroupNorm(8, channels),
                nn.GELU(),
            )
            
            self.conv_out = nn.Conv2d(channels * 2, num_channels, 3, padding=1)

        def forward(self, x, t):
            t_emb = self.time_mlp(t)
            
            x = self.conv_in(x)
            h1 = x + t_emb[:, :, None, None]
            
            h2 = self.down1(h1)
            h3 = self.down2(h2)
            
            h3 = self.mid(h3)
            
            h = self.up1(h3)
            h = torch.cat([h, h2], dim=1)
            h = self.up2(h)
            h = torch.cat([h, h1], dim=1)
            
            return self.conv_out(h)

    class ImageDataset(TorchDataset):
        def __init__(self, image_paths, captions, image_size=128):
            self.image_paths = image_paths
            self.captions = captions if captions else [''] * len(image_paths)
            self.image_size = image_size

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img).astype(np.float32) / 255.0
            img = (img - 0.5) * 2.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            caption = self.captions[idx] if idx < len(self.captions) else ''
            return img, caption

    class DiffusionTrainer:
        def __init__(self, config):
            self.config = config
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            self.model = SimpleUNet(
                image_size=config.image_size,
                channels=config.channels,
                time_emb_dim=config.time_emb_dim,
                num_channels=config.num_channels
            ).to(self.device)
            
            self.betas = torch.linspace(
                config.beta_start, 
                config.beta_end, 
                config.timesteps
            ).to(self.device)
            self.alphas = 1.0 - self.betas
            self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
            self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
            self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
                1.0 - self.alphas_cumprod
            )

        def add_noise(self, x, t):
            sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
            sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][
                :, None, None, None
            ]
            noise = torch.randn_like(x)
            return sqrt_alpha * x + sqrt_one_minus * noise, noise

        def train(self, dataloader):
            optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.learning_rate
            )
            
            os.makedirs(self.config.output_model_path, exist_ok=True)
            checkpoint_dir = os.path.join(
                self.config.output_model_path, 
                'checkpoints'
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            global_step = 0
            
            for epoch in range(self.config.num_epochs):
                self.model.train()
                epoch_loss = 0.0
                
                pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
                for batch_idx, (images, captions) in enumerate(pbar):
                    images = images.to(self.device)
                    batch_size = images.shape[0]
                    
                    t = torch.randint(
                        0, 
                        self.config.timesteps, 
                        (batch_size,), 
                        device=self.device
                    ).long()
                    
                    noisy_images, noise = self.add_noise(images, t)
                    
                    predicted_noise = self.model(noisy_images, t)
                    
                    loss = F.mse_loss(predicted_noise, noise)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    global_step += 1
                    
                    pbar.set_postfix({'loss': loss.item()})
                    
                    if global_step % self.config.checkpoint_frequency == 0:
                        ckpt_path = os.path.join(
                            checkpoint_dir,
                            f'checkpoint-epoch{epoch+1}-step{global_step}.pt'
                        )
                        torch.save({
                            'epoch': epoch,
                            'step': global_step,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': loss.item(),
                        }, ckpt_path)
                
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch {epoch+1} avg loss: {avg_loss:.6f}')
            
            final_path = os.path.join(
                self.config.output_model_path, 
                'model_final.pt'
            )
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
            }, final_path)
            
            return self.config.output_model_path

        @torch.no_grad()
        def sample(self, num_samples=1):
            self.model.eval()
            
            x = torch.randn(
                num_samples, 
                self.config.num_channels,
                self.config.image_size, 
                self.config.image_size,
                device=self.device
            )
            
            for t in reversed(range(self.config.timesteps)):
                t_batch = torch.full(
                    (num_samples,), 
                    t, 
                    device=self.device, 
                    dtype=torch.long
                )
                
                predicted_noise = self.model(x, t_batch)
                
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                beta = self.betas[t]
                
                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha)) * (
                    x - (beta / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
                ) + torch.sqrt(beta) * noise
            
            x = (x + 1) / 2
            x = torch.clamp(x, 0, 1)
            
            return x

else:
    SinusoidalPositionEmbeddings = None
    SimpleUNet = None
    ImageDataset = None
    DiffusionTrainer = None


def train_diffusion(image_paths, captions=None, config=None, 
                    resume_from=None):
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not available. Install: pip install torch torchvision"
        )
    
    if config is None:
        config = DiffusionConfig()
    
    if captions is None:
        captions = [''] * len(image_paths)
    
    dataset = ImageDataset(image_paths, captions, config.image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    trainer = DiffusionTrainer(config)
    
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Resumed from {resume_from}')
    
    output_path = trainer.train(dataloader)
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_path


def generate_image(model_path, prompt=None, num_samples=1, image_size=128):
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not available. Install: pip install torch torchvision"
        )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Fix: Load with weights_only=False for your custom checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = DiffusionConfig(image_size=image_size)
    
    trainer = DiffusionTrainer(config)
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    samples = trainer.sample(num_samples)
    
    images = []
    for i in range(num_samples):
        img_tensor = samples[i].cpu()
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        images.append(img)
    
    if num_samples == 1:
        return images[0]
    return images