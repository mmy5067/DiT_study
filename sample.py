import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
import os
import math

# 假设 model.py 在同一目录下
from model import DiT 

# 和train.py中一致的扩散超参数
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]]) # p_sample会用到

# 计算 p_sample 所需的均值和方差系数
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# 超参数 (需要与训练时一致或根据需求调整)
img_size = 28
img_channels = 1
patch_size = 4
embed_dim = 256
num_heads = 4
num_layers = 6 # DiT的深度
device = "cuda" if torch.cuda.is_available() else "cpu"

output_dir = "./outputs/generated_samples"
os.makedirs(output_dir, exist_ok=True)
num_samples = 20 # 修改为20个样本 (0-9每个数字2次)
sample_batch_size = 2 # 每次生成2个样本

# 模型路径
model_path = "./outputs/dit_mnist_final.pth" 

# 辅助函数：从 alphas_cumprod 中提取特定时间步 t 的值，并重塑以进行广播
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

@torch.no_grad()
def p_sample(model, x, t, t_index, class_labels=None):
    """
    单步去噪函数 (DDPM的反向过程中的一步)
    model: 训练好的DiT模型
    x: 当前的噪声图像 (B, C, H, W)
    t: 当前的时间步张量 (B,)
    t_index: 当前时间步的索引 (整数)
    class_labels: 类别标签 (B,) 可选
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        torch.sqrt(1. - alphas_cumprod), t, x.shape
    )
    sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / alphas), t, x.shape)
    
    # 使用模型预测噪声 - 传入类别标签
    predicted_noise = model(x, t, y=class_labels) 
    
    # 计算均值
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, num_timesteps, class_labels=None):
    """
    完整的采样循环
    model: 训练好的DiT模型
    shape: 要生成的图像的形状 (B, C, H, W)
    num_timesteps: 总的扩散步数
    class_labels: 类别标签 (B,) 可选
    """
    device = next(model.parameters()).device
    b = shape[0]
    # 从纯噪声开始
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, num_timesteps)):
        # 创建时间步张量 (B,)
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, class_labels)
    imgs.append(img.cpu()) # 保存最后生成的图像
    return imgs

def sample():
    print(f"Loading model from {model_path}...")
    model = DiT(
        in_channels=img_channels,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=num_layers, 
        num_classes=10, # 添加类别数支持
        mlp_ratio=4.0,
        num_timesteps=num_timesteps
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model weights not found at {model_path}. Please train the model first and save it as dit_mnist_final.pth")
        return
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.eval()
    print(f"Model loaded. Generating {num_samples} samples on {device}...")

    all_samples = []
    
    # 为每个数字生成2次
    for digit in range(10):  # 0-9
        for repeat in range(2):  # 每个数字生成2次
            current_batch_size = sample_batch_size
            shape = (current_batch_size, img_channels, img_size, img_size)
            
            # 创建类别标签 - 当前数字
            class_labels = torch.full((current_batch_size,), digit, device=device, dtype=torch.long)
            
            print(f"Generating digit {digit} (attempt {repeat + 1}/2)...")
            generated_images_sequence = p_sample_loop(model, shape, num_timesteps, class_labels)
            generated_images = generated_images_sequence[-1] # 取最后一步的结果
            all_samples.append(generated_images)
    
    all_samples_tensor = torch.cat(all_samples, dim=0)

    # 将图像值从[-1, 1]转换回[0, 1]以便保存
    all_samples_tensor = (all_samples_tensor + 1) / 2 
    all_samples_tensor = torch.clamp(all_samples_tensor, 0, 1)

    # 创建网格显示，每行显示4个图像
    grid = make_grid(all_samples_tensor, nrow=4)
    save_path = os.path.join(output_dir, f"generated_samples_0_9_twice.png")
    save_image(grid, save_path)
    print(f"Saved {num_samples} generated samples (0-9 each twice) to {save_path}")


@torch.no_grad()
def p_sample(model, x, t, t_index, class_labels=None):
    """
    单步去噪函数 (DDPM的反向过程中的一步)
    model: 训练好的DiT模型
    x: 当前的噪声图像 (B, C, H, W)
    t: 当前的时间步张量 (B,)
    t_index: 当前时间步的索引 (整数)
    class_labels: 类别标签 (B,) 可选
    """
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        torch.sqrt(1. - alphas_cumprod), t, x.shape
    )
    sqrt_recip_alphas_t = extract(torch.sqrt(1.0 / alphas), t, x.shape)
    
    # 使用模型预测噪声 - 传入类别标签
    predicted_noise = model(x, t, y=class_labels) 
    
    # 计算均值
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, num_timesteps, class_labels=None):
    """
    完整的采样循环
    model: 训练好的DiT模型
    shape: 要生成的图像的形状 (B, C, H, W)
    num_timesteps: 总的扩散步数
    class_labels: 类别标签 (B,) 可选
    """
    device = next(model.parameters()).device
    b = shape[0]
    # 从纯噪声开始
    img = torch.randn(shape, device=device)
    imgs = []

    for i in reversed(range(0, num_timesteps)):
        # 创建时间步张量 (B,)
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, class_labels)
    imgs.append(img.cpu()) # 保存最后生成的图像
    return imgs

if __name__ == '__main__':
    sample()