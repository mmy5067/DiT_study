import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataloaders(batch_size=64, data_dir='./mnist'):

    transform = transforms.Compose([
        transforms.ToTensor(),  # 将PIL图像或numpy.ndarray转换为tensor，并将像素值从[0, 255]缩放到[0, 1]
        transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1]范围，这对于生成模型通常有益
    ])

    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader

if __name__ == '__main__':

    mnist_data_root_dir = 'g:\\paper\\DiT_study\\mnist' 
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64, data_dir=mnist_data_root_dir)

    print(f"训练集批次数: {len(train_loader)}")
    print(f"测试集批次数: {len(test_loader)}")

    # 显示一个批次的图像形状和标签
    for images, labels in train_loader:
        print("图像批次形状:", images.shape) # torch.Size([64, 1, 28, 28])
        print("标签批次形状:", labels.shape) # torch.Size([64])
        break