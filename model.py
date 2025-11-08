import torch
import torch.nn as nn
from torchvision import models


class ViT_Tiny(nn.Module):
    def __init__(self, num_classes=200):
        super(ViT_Tiny, self).__init__()
        # 使用torchvision提供的ViT模型，设置为16x16的patch，适配64x64图像
        self.model = models.vit_b_16(pretrained=False)  # 不使用预训练权重

        # 获取ViT的分类器部分的输入特征数
        in_features = self.model.heads[0].in_features  # 获取ViT的输入特征数

        # 修改头部（分类器）
        self.model.heads = nn.Sequential(  # 替换ViT的头部为自定义的全连接层
            nn.Linear(in_features, 512),  # 隐藏层
            nn.ReLU(),  # 激活函数
            nn.Linear(512, num_classes)  # 输出层，适应Tiny-ImageNet 200个类别
        )

    def forward(self, x):
        return self.model(x)

# 初始化ViT-Tiny模型
model = ViT_Tiny(num_classes=200)  # Tiny-ImageNet包含200个类别