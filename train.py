import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import ContrastiveLearning
from dataset import get_dataloaders
from config import config
import os
from datetime import datetime
from tqdm import tqdm

def train_contrastive(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f'Train Epoch: {epoch}')
    for batch_idx, (x1, x2, _) in enumerate(pbar):
        x1, x2 = x1.to(config['device']), x2.to(config['device'])
        
        optimizer.zero_grad()
        z1, z2 = model(x1, x2)
        loss = model.contrastive_loss(z1, z2)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / len(train_loader)

def train_classifier(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total_samples = len(train_loader.dataset)
    
    pbar = tqdm(train_loader, desc=f'Train Epoch: {epoch}')
    for batch_idx, (x1, _, labels) in enumerate(pbar):
        x1, labels = x1.to(config['device']), labels.to(config['device'])
        
        optimizer.zero_grad()
        outputs = model(x1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.6f}',
            'acc': f'{100. * correct / total_samples:.2f}%'
        })
    
    accuracy = 100. * correct / total_samples
    return total_loss / len(train_loader), accuracy

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_samples = len(test_loader.dataset)
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(test_loader, desc='Test')
    with torch.no_grad():
        for x1, _, labels in pbar:
            x1, labels = x1.to(config['device']), labels.to(config['device'])
            outputs = model(x1)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{test_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total_samples:.2f}%'
            })
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total_samples
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{total_samples} '
          f'({accuracy:.2f}%)\n')
    
    return test_loss, accuracy

def train_model(args):
    # 创建保存目录
    if not os.path.exists(config['checkpoint_dir']):
        os.makedirs(config['checkpoint_dir'])
    
    # 创建日期时间子文件夹
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    save_dir = os.path.join(config['checkpoint_dir'], timestamp)
    os.makedirs(save_dir)
    
    # 获取数据加载器
    train_loader, test_loader = get_dataloaders()
    
    # 创建模型
    model = ContrastiveLearning().to(config['device'])
    
    if args.stage == 'pretrain':
        # 预训练阶段
        print("Stage: Contrastive Learning Pre-training")
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs_pretrain'])
        
        best_loss = float('inf')
        best_model_path = None  # 添加变量跟踪最佳模型路径
        
        for epoch in range(1, config['num_epochs_pretrain'] + 1):
            loss = train_contrastive(model, train_loader, optimizer, epoch)
            scheduler.step()
            
            # 保存最好的模型
            if loss < best_loss:
                best_loss = loss
                # 如果存在之前的最佳模型，删除它
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # 保存新的最佳模型
                best_model_path = f"{save_dir}/best_pretrain_model_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, best_model_path)
                print(f"Saved best model at epoch {epoch} with loss: {loss:.6f}")

    elif args.stage == 'finetune':
        # 加载预训练模型
        if args.resume is None:
            raise ValueError("Must provide --resume checkpoint path for fine-tuning")
        
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained model from {args.resume}")
        
        # 微调阶段
        print("Stage: Classifier Fine-tuning")
        # 冻结编码器参数
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        optimizer = optim.Adam(model.decoder.parameters(), lr=config['learning_rate'],
                              weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs_finetune'])
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        best_model_path = None  # 添加变量跟踪最佳模型路径
        
        for epoch in range(1, config['num_epochs_finetune'] + 1):
            train_loss, train_acc = train_classifier(model, train_loader, optimizer,
                                                    criterion, epoch)
            test_loss, test_acc = test(model, test_loader)
            scheduler.step()
            
            if test_acc > best_acc:
                best_acc = test_acc
                # 如果存在之前的最佳模型，删除它
                if best_model_path is not None and os.path.exists(best_model_path):
                    os.remove(best_model_path)
                
                # 保存新的最佳模型
                best_model_path = f"{save_dir}/best_finetune_model_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                }, best_model_path)
                print(f"Saved best model at epoch {epoch} with test accuracy: {test_acc:.4f}")