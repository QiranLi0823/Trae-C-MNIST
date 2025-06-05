import argparse
from train import train_model, test
from utils import set_seed
from model import ContrastiveLearning  # 添加导入
from config import config  # 添加导入
import torch
from dataset import get_dataloaders  # 添加导入

def parse_args():
    parser = argparse.ArgumentParser(description='MNIST Contrastive Learning')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'],
                        help='mode: train or test')
    parser.add_argument('--stage', type=str, choices=['pretrain', 'finetune'],
                        help='training stage: pretrain or finetune (required for train mode)')
    parser.add_argument('--resume', type=str, default=None,
                        help='path to checkpoint to resume from')
    parser.add_argument('--checkpoint', type=str,
                        help='path to model checkpoint for testing')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    if args.mode == 'train':
        if args.stage is None:
            raise ValueError("--stage argument is required for train mode")
        train_model(args)
    else:  # test mode
        if args.checkpoint is None:
            raise ValueError("--checkpoint argument is required for test mode")
        # 加载模型和数据
        model = ContrastiveLearning().to(config['device'])
        # 修改加载方式，从字典中提取模型状态
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        _, test_loader = get_dataloaders()
        # 执行测试
        test(model, test_loader)

if __name__ == '__main__':
    main()