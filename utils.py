import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def visualize_samples(images, labels, num_samples=16):
    """可视化一批样本"""
    # 选择指定数量的样本
    images = images[:num_samples]
    labels = labels[:num_samples]
    
    # 创建图像网格
    grid = make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.title('Sample Images with Labels: ' + 
              ' '.join(str(label.item()) for label in labels))
    plt.show()

def plot_training_curve(train_losses, test_losses, save_path=None):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Curves')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def load_checkpoint(model, checkpoint_path):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint