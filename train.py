import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import os
import math # 引入math模块

from dataset import get_mnist_dataloaders 
from model import DiT 

# 超参数
epochs = 100
batch_size = 64
learning_rate = 1e-4
img_size = 28 
img_channels = 1 
patch_size = 4 
embed_dim = 256 
num_heads = 4 
num_layers = 6 # 在DiT初始化时会用 depth=num_layers
device = "cuda" if torch.cuda.is_available() else "cpu"

# 扩散模型相关超参数
num_timesteps = 1000 # 总的扩散步数
beta_start = 0.0001
beta_end = 0.02

# 结果保存路径
output_dir = "outputs"
sample_dir = os.path.join(output_dir, "samples") # 存放采样图片的子目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

# --- 扩散过程辅助函数 --- 
def linear_beta_schedule(timesteps, start=beta_start, end=beta_end):
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(num_timesteps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]]) # p(x_{t-1}|x_t, x_0)会用到

# 计算 q(x_t | x_0) 所需的均值和方差系数
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 前向加噪过程 q(x_t | x_0)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) # (B, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) # (B, 1, 1, 1)
    
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image
# --- 扩散过程辅助函数结束 --- 

def train():
    train_dataloader, _ = get_mnist_dataloaders(batch_size=batch_size, data_dir='./mnist')

    model = DiT(
        in_channels=img_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=num_layers, 
        num_classes=10,  # MNIST有10个类别
        mlp_ratio=4.0,
        num_timesteps=num_timesteps
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print(f"Starting training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (images, labels) in enumerate(train_dataloader):  
            images = images.to(device)
            labels = labels.to(device)  # 将标签移到设备上
            B = images.shape[0]
            
            # 1. 随机采样时间步
            t = torch.randint(0, num_timesteps, (B,), device=device).long()
            
            # 2. 生成真实噪声
            noise = torch.randn_like(images)
            
            # 3. 计算带噪图像
            noisy_images = q_sample(x_start=images, t=t, noise=noise)
            
            # 4. 模型预测噪声 - 传入类别标签
            predicted_noise = model(noisy_images, t, y=labels)
            
            # 5. 计算损失
            loss = criterion(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}")

    print("Training finished.")
    torch.save(model.state_dict(), os.path.join(output_dir, "dit_mnist_final.pth"))
    print(f"Model saved to {os.path.join(output_dir, 'dit_mnist_final.pth')}")

if __name__ == "__main__":
    train()