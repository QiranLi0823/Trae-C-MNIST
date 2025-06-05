import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from config import config

class ContrastiveMNIST(Dataset):
    def __init__(self, data_path, train=True, transform=None):
        if train:
            self.x = np.load(f"{data_path}/mnist_train_x.npy")
            self.y = np.load(f"{data_path}/mnist_train_y.npy")
        else:
            self.x = np.load(f"{data_path}/mnist_test_x.npy")
            self.y = np.load(f"{data_path}/mnist_test_y.npy")
        
        self.x = torch.FloatTensor(self.x) / 255.0  # 归一化
        self.x = self.x.reshape(-1, 1, 28, 28)  # 调整维度为(N, C, H, W)
        self.y = torch.LongTensor(self.y)
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        
        if self.transform:
            img1 = self.transform(img)
            img2 = self.transform(img)  # 对同一图像应用两次不同的随机增强
            return img1, img2, label
        return img, img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(config['rotation_range']),
        transforms.RandomAffine(
            degrees=0,
            translate=(config['shift_range'], config['shift_range']),
            scale=config['scale_range']
        )
    ])

    return train_transform, None

def get_dataloaders():
    train_transform, test_transform = get_transforms()

    train_dataset = ContrastiveMNIST(
        data_path=config['data_path'],
        train=True,
        transform=train_transform
    )

    test_dataset = ContrastiveMNIST(
        data_path=config['data_path'],
        train=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    return train_loader, test_loader