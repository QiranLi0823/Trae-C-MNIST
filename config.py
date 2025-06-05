import torch

config = {
    # 数据集配置
    'data_path': '/home/qiranli/src/datasets/MNIST',  # 确保路径指向包含.npy文件的目录
    'batch_size': 256,
    'num_workers': 4,
    
    # 模型配置
    'feature_dim': 128,  # 特征维度
    'temperature': 0.07,  # 对比学习温度参数
    
    # 训练配置
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs_pretrain': 50,  # 对比学习预训练轮数
    'num_epochs_finetune': 30,   # 分类器微调轮数
    
    # 数据增强配置
    'rotation_range': 15,    # 随机旋转角度范围
    'shift_range': 0.1,      # 随机平移范围
    'scale_range': (0.9, 1.1),  # 随机缩放范围
    
    # 保存配置
    'checkpoint_dir': 'checkpoints',
    'log_interval': 100,     # 日志打印间隔
}