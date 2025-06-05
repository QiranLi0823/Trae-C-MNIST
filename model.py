import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, config['feature_dim'])
        )
        
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config['feature_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # MNIST有10个类别
        )
    
    def forward(self, x):
        return self.classifier(x)

class ContrastiveLearning(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.temperature = config['temperature']
        
    def forward(self, x1, x2=None):
        if self.training and x2 is not None:
            # 对比学习阶段
            z1 = self.encoder(x1)
            z2 = self.encoder(x2)
            return z1, z2
        else:
            # 分类阶段
            z = self.encoder(x1)
            return self.decoder(z)

    def contrastive_loss(self, z1, z2):
        # 归一化特征向量
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 正样本对的标签（对角线上的元素）
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # 计算对比学习损失（交叉熵）
        loss = F.cross_entropy(similarity_matrix, labels) + \
               F.cross_entropy(similarity_matrix.T, labels)
        
        return loss / 2