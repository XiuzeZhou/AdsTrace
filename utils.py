import os
import torch
import csv
import numpy as np
from datetime import datetime

def prepare_model_for_warmup(model):
    for param in model.parameters():
        param.requires_grad = False

    trainable_keywords = ['hme', 'taa', 'bilstm', 'head', 'proj', 'cross', 'norm', 'fusion']
    for name, param in model.named_parameters():
        if any(x in name for x in trainable_keywords):
            param.requires_grad = True
    print("Warm-up mode active: Custom layers trainable.")

def unfreeze_all_layers(model, base_lr=1e-5):
    for param in model.parameters():
        param.requires_grad = True
    
    backbone_params = list(model.visual_enc.parameters()) + \
                      list(model.bert.parameters()) + \
                      list(model.wav2vec2.parameters())
    backbone_ids = set(id(p) for p in backbone_params)

    custom_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    
    param_groups = [
        {'params': model.visual_enc.parameters(), 'lr': base_lr * 0.1},
        {'params': model.bert.parameters(), 'lr': base_lr * 0.1},
        {'params': model.wav2vec2.parameters(), 'lr': base_lr * 0.1},
        {'params': custom_params, 'lr': base_lr * 10}
    ]
    
    optimizer = torch.optim.AdamW(param_groups)

    return optimizer

class EarlyStopping:
    def __init__(self, patience=3, checkpoint_path='best_taman.pth'):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        dir_name = os.path.dirname(checkpoint_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.checkpoint_path)
        self.val_loss_min = val_loss


def log_experiment_results(args, metrics, file_path='outputs/summary_results.csv'):
    """
    将实验配置和最终测试结果保存到 CSV 文件中
    args: 命令行参数对象
    metrics: 字典，包含 AUC, MAE 等指标
    """
    # 准备表头和数据
    header = [
        'timestamp', 'exp_name', 'lr', 'lambda_loss', 'hidden_size', 
        'num_layers', 'batch_size', 'test_auc', 'test_f1', 
        'roi_mae', 'roi_rmse', 'cvr_mae', 'cvr_rmse'
    ]
    
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'exp_name': args.exp_name,
        'lr': args.lr,
        'lambda_loss': args.lambda_loss,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'test_auc': f"{metrics['auc']:.4f}",
        'test_f1': f"{metrics['f1']:.4f}",
        'roi_mae': f"{metrics['mae_roi']:.4f}",
        'roi_rmse': f"{metrics['rmse_roi']:.4f}",
        'cvr_mae': f"{metrics['mae_cvr']:.4f}",
        'cvr_rmse': f"{metrics['rmse_cvr']:.4f}"
    }

    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    
    print(f">>> Results successfully logged to {file_path}")