# 一个AIGC初学者的DiT复现

## 简介

- 以学习为主要目的，简化了DiT模型，缩减了参数，使用MNIST手写数字数据集进行训练。

## 环境依赖

- python 3.9.18
- torch = 2.7.0+cu128
- torchvision = 0.22.0+cu128

## 文件结构

```directory
DiT_study/
├── mnist/
│   └── ...
├── dataset.py
├── model.py
├── train.py
├── sample.py
└── README.md
```

## 学习过程

- 参考[Scale Diffusion Models with Transformer](github.com/facebookresearch/DiT)
- 代码由**Claude-3.7-sonnet**辅助修改完成

