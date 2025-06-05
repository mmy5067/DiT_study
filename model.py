import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """正弦位置编码。"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe) # 将pe注册为buffer，这样模型保存时会一同保存，但不会被视为模型参数参与梯度更新

    def forward(self, x):
        """ x: 输入张量，形状为 (seq_len, batch_size, d_model) """
        x = x + self.pe[:x.size(0), :] # 将位置编码加到输入张量上
        return self.dropout(x)

class DiTBlock(nn.Module):
    """一个DiT块，使用自适应层归一化 (adaLN-Zero)。"""
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) # 注意力模块
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio) # MLP的隐藏层维度
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(), # GELU激活函数
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # adaLN-Zero 调制参数生成模块
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), # SiLU激活函数
            nn.Linear(hidden_size, 6 * hidden_size, bias=True) # 输出6倍hidden_size用于生成缩放和平移参数
        )

    def forward(self, x, c):
        # c 是条件嵌入 (例如时间嵌入)，形状为 (B, hidden_size)
        # 生成调制参数: shift_msa, scale_msa, gate_msa 分别对应多头注意力后的调整
        # shift_mlp, scale_mlp, gate_mlp 分别对应MLP后的调整
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # 注意力块
        x_norm1 = self.norm1(x) # 层归一化
        # adaLN: x_norm * (1 + scale) + shift
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1) # 多头自注意力
        # gate_msa 控制残差连接的强度
        x = x + gate_msa.unsqueeze(1) * attn_output # 残差连接
        
        # MLP 块
        x_norm2 = self.norm2(x) # 层归一化
        # adaLN: x_norm * (1 + scale) + shift
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2) # MLP前馈网络
        # gate_mlp 控制残差连接的强度
        x = x + gate_mlp.unsqueeze(1) * mlp_output # 残差连接
        return x

class DiT(nn.Module):
    """
    Diffusion Transformer (DiT) 模型，适用于MNIST数据集。
    """
    def __init__(self, img_size=28, patch_size=4, in_channels=1, embed_dim=256, 
                 depth=6, num_heads=4, mlp_ratio=4.0, num_classes=10, num_timesteps=1000):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2 # patch的数量
        self.num_classes = num_classes

        # 1. Patch嵌入: 将输入图像转换为一系列patch嵌入
        # 使用卷积层实现，kernel_size和stride都等于patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # 2. 位置嵌入: 为每个patch添加位置信息
        # 这是一个可学习的参数，形状为 (1, num_patches, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3. 时间嵌入: 将时间步t编码为向量
        # 首先使用PositionalEncoding（尽管它通常用于序列），然后通过MLP处理
        time_mlp_hidden_dim = int(embed_dim * mlp_ratio) # <--- 修改这里，确保是整数
        self.time_embed = nn.Sequential(
            PositionalEncoding(embed_dim, max_len=num_timesteps), # 对时间步进行编码
            nn.Linear(embed_dim, time_mlp_hidden_dim), # 线性层 <--- 使用整数维度
            nn.SiLU(), # SiLU激活函数
            nn.Linear(time_mlp_hidden_dim, embed_dim), # 线性层 <--- 使用整数维度
        )

        # 4. 类别嵌入 
        self.class_embed = nn.Embedding(num_classes, embed_dim)
        
        # 5. Transformer 块: DiT的核心部分
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        
        # 6. 输出层: 将Transformer的输出映射回patch的维度，用于预测噪声
        self.norm_out = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6) # 输出前的层归一化
        self.final_layer = nn.Linear(embed_dim, patch_size * patch_size * in_channels) # 线性层预测每个patch的像素值
        self.initialize_weights() # 初始化权重

    def initialize_weights(self):
        # 初始化位置嵌入和patch嵌入的权重
        nn.init.normal_(self.pos_embed, std=.02) # 正态分布初始化位置嵌入
        # 初始化类别嵌入
        nn.init.normal_(self.class_embed.weight, std=.02)

    def unpatchify(self, x):
        """
        将patch序列转换回图像形状。
        x: 输入张量，形状为 (B, N, P*P*C)，其中N是patch数量, P是patch_size, C是通道数
        输出: 图像张量，形状为 (B, C, H, W)
        """
        B = x.shape[0] # 批量大小
        P = self.patch_size # patch 边长
        # H_patch 和 W_patch 是在高度和宽度维度上的patch数量
        H_patch = W_patch = int(self.num_patches**0.5)
        assert H_patch * W_patch == self.num_patches, "num_patches开方后不是整数，图像尺寸或patch尺寸设置有误"
        
        # x: (B, N, P*P*C) -> (B, H_patch, W_patch, P, P, C_patch)
        # C_patch 通常等于 in_channels
        x = x.reshape(B, H_patch, W_patch, P, P, -1) 
        # permute: (B, C_patch, H_patch, P_h, W_patch, P_w)
        x = x.permute(0, 5, 1, 3, 2, 4) 
        # reshape: (B, C_patch, H_patch*P_h, W_patch*P_w) -> (B, C, H, W)
        images = x.reshape(B, -1, H_patch * P, W_patch * P) 
        return images

    def forward(self, x, t, y=None): # x: (B, C, H, W), t: (B,)
        B, C, H, W = x.shape
        x = self.patch_embed(x)  # (B, embed_dim, H_patch, W_patch)
        # flatten(2) 将最后两个维度展平, transpose(1,2) 交换维度以匹配Transformer输入 (B, SeqLen, Dim)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        x = x + self.pos_embed # 添加位置嵌入
        
        # 时间嵌入
        # t 的期望形状是 (B,), 需要将其转换为 (B, embed_dim) 以便输入到 time_embed
        # PositionalEncoding 期望的输入是 (seq_len, batch, features) 或 (seq_len, features)
        # 对于时间步 't', 每个批次元素对应一个时间步值。
        # 我们需要调整时间嵌入的获取方式。常见做法是直接对t使用正弦编码。
        
        # 当前简化的时间嵌入 (可以使用更复杂的正弦特征进行改进)
        # 假设 t 是 (B,) 形状的整数时间步张量
        # self.time_embed[0] 是 PositionalEncoding 实例
        # self.time_embed[0].pe 是 (max_len, d_model) 的位置编码表
        # pe[t] 会选取对应时间步的编码, squeeze(1) 如果pe[t]是 (B,1,D) 则变为 (B,D)
        # 注意：PositionalEncoding的forward方法是为序列设计的，这里直接取用其pe buffer
        time_encoded_vector = self.time_embed[0].pe[t] # (B, 1, embed_dim) if pe is (max_len, 1, dim) or (B, embed_dim) if pe is (max_len, dim)
        if time_encoded_vector.dim() == 3 and time_encoded_vector.shape[1] == 1:
            time_encoded_vector = time_encoded_vector.squeeze(1) # (B, embed_dim)
        
        # 通过后续的线性层和激活函数处理时间编码
        time_emb = self.time_embed[1:](time_encoded_vector) # (B, embed_dim)

        # 条件嵌入：结合时间和类别信息
        if y is not None:
            # 类别条件生成
            class_emb = self.class_embed(y)  # (B, embed_dim)
            c = time_emb + class_emb  # 将时间嵌入和类别嵌入相加
        else:
            # 无条件生成
            c = time_emb

        # 通过一系列DiT块处理
        for block in self.blocks:
            x = block(x, c)
        
        x = self.norm_out(x) # 输出前的层归一化
        x = self.final_layer(x) # (B, num_patches, patch_size*patch_size*in_channels)
        
        # 将patch序列转换回图像形状以得到预测的噪声
        predicted_noise = self.unpatchify(x)
        return predicted_noise

if __name__ == '__main__':
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 28
    patch_size = 4 # 28能被4整除 -> 7x7 个patches
    model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_channels=1,      # MNIST是单通道灰度图
        embed_dim=128,      # 嵌入维度 (为MNIST减小)
        depth=4,            # Transformer块的深度 (为MNIST减小)
        num_heads=4,        # 多头注意力的头数
        num_timesteps=1000  # 扩散过程的总时间步数
    ).to(device)

    # 创建一个虚拟输入批次
    dummy_x = torch.randn(4, 1, img_size, img_size).to(device) # 4张图像的批次
    dummy_t = torch.randint(0, 1000, (4,)).to(device)    # 4个时间步的批次

    print("输入图像形状:", dummy_x.shape)
    predicted_noise = model(dummy_x, dummy_t)
    print("预测噪声形状:", predicted_noise.shape)

    # 计算模型的可训练参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数数量: {num_params/1e6:.2f}M")

