import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 下采样路径 (编码器部分)
        # 第一层
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)  # 28x28 -> 14x14
        
        # 第二层
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)  # 14x14 -> 7x7
        
        # 瓶颈层
        self.bottleneck_conv = nn.Conv2d(64, 128, 3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(128)
        
        # 上采样路径 (解码器部分)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 7x7 -> 14x14
        self.upbn1 = nn.BatchNorm2d(64)
        self.upconv1_2 = nn.Conv2d(128, 64, 3, padding=1)  # 合并后的通道数减半
        self.upbn1_2 = nn.BatchNorm2d(64)
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # 14x14 -> 28x28
        self.upbn2 = nn.BatchNorm2d(32)
        self.upconv2_2 = nn.Conv2d(64, 32, 3, padding=1)  # 合并后的通道数减半
        self.upbn2_2 = nn.BatchNorm2d(32)
        
        # 最终特征提取
        self.final_conv = nn.Conv2d(32, 32, 3, padding=1)
        self.final_bn = nn.BatchNorm2d(32)
        
        # 特征压缩
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, config['feature_dim'])
        
    def forward(self, x):
        # 下采样路径
        x1 = F.relu(self.bn1(self.conv1(x)))  # 第一层特征
        p1 = self.pool1(x1)
        
        x2 = F.relu(self.bn2(self.conv2(p1)))  # 第二层特征
        p2 = self.pool2(x2)
        
        # 瓶颈层
        bottleneck = F.relu(self.bottleneck_bn(self.bottleneck_conv(p2)))
        
        # 上采样路径 (带跳跃连接)
        up1 = self.upbn1(self.upconv1(bottleneck))
        # 连接特征图 (确保尺寸匹配)
        concat1 = torch.cat([up1, x2], dim=1)  # 沿通道维度连接
        up1_2 = F.relu(self.upbn1_2(self.upconv1_2(concat1)))
        
        up2 = self.upbn2(self.upconv2(up1_2))
        # 连接特征图
        concat2 = torch.cat([up2, x1], dim=1)  # 沿通道维度连接
        up2_2 = F.relu(self.upbn2_2(self.upconv2_2(concat2)))
        
        # 最终特征提取
        final = F.relu(self.final_bn(self.final_conv(up2_2)))
        
        # 特征压缩
        pooled = self.avgpool(final)
        flattened = self.flatten(pooled)
        features = self.fc(flattened)
        
        return features

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 增加中间层维度
        self.fc1 = nn.Linear(config['feature_dim'], 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 10)  # MNIST有10个类别
        
        # 残差连接用的映射层
        self.shortcut = nn.Linear(config['feature_dim'], 128)
        
    def forward(self, x):
        # 第一个残差块
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out = F.relu(out + identity)  # 残差连接
        
        # 第二层
        out = F.relu(self.bn3(self.fc3(out)))
        out = self.fc4(out)
        
        return out

# 预训练阶段模型 - 只包含编码器和对比损失
class PretrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.temperature = config['temperature']
        
    def forward(self, x1, x2):
        # 预训练阶段 - 只使用编码器
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return z1, z2
    
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

# 微调阶段模型 - 包含预训练的编码器（冻结）和可训练的解码器
class FinetuneModel(nn.Module):
    def __init__(self, pretrained_encoder=None):
        super().__init__()
        if pretrained_encoder is not None:
            self.encoder = pretrained_encoder
        else:
            self.encoder = Encoder()
            
        # 冻结编码器参数
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.decoder = Decoder()
        
    def forward(self, x):
        # 微调阶段 - 使用冻结的编码器和可训练的解码器
        z = self.encoder(x)
        return self.decoder(z)

# 保留原始类以保持兼容性
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