import argparse
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from dataset import AdsTraceDataset, custom_collate_fn
from model import TAMAN
from utils import prepare_model_for_warmup, unfreeze_all_layers, EarlyStopping, log_experiment_results
from engine import train_one_epoch, evaluate
import os
import json

def main():
    parser = argparse.ArgumentParser(description="TAMAN Training Script")
    parser.add_argument('--data_path', type=str, default='./datasets/AdsTrace', help='path for loading data')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--lambda_loss', type=float, default=0.5, help='Weight for iCTR loss')
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of TAA layers')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--exp_name', type=str, default='default_exp', help='Name for Tensorboard and Checkpoint')
    parser.add_argument('--bert_path', type=str, default='./pretrained_models/bert-base-chinese', help='path of BERT')
    parser.add_argument('--wav2vec_path', type=str, default='./pretrained_models/wav2vec2-large-xlsr-53-chinese-zh-cn', help='path of Wav2Vec2')
    parser.add_argument('--swin_path', type=str, default='local-dir:./pretrained_models/swin_base_patch4_window7_224', help='path of Swin-Base')
    
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = AdsTraceDataset(root_dir=args.data_path, transform=transform, bert_path=args.bert_path)
    
    split_path = os.path.join(args.data_path, "split.json")
    with open(split_path, "r") as f:
        splits = json.load(f)
    
    id_to_idx = {vid: i for i, vid in enumerate(dataset.video_ids)}
    train_idx = [id_to_idx[v] for v in splits['train'] if v in id_to_idx]
    val_idx = [id_to_idx[v] for v in splits['val'] if v in id_to_idx]
    test_idx = [id_to_idx[v] for v in splits['test'] if v in id_to_idx]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=16, pin_memory=True)

    # --- Model Training---
    model = TAMAN(hidden_size=args.hidden_size, 
                  num_layers=args.num_layers,
                  swin_path=args.swin_path,
                  bert_path=args.bert_path,
                  wav2vec_path=args.wav2vec_path
                  ).cuda()
    
    train(train_loader, val_loader, model, args) 
    print("\n" + "="*30)
    print(">>> Training Finished. Starting Final Test...")
    print("="*30)
    
    # Evaluate model
    checkpoint_path = f'outputs/{args.exp_name}/best_model.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    t_loss, t_ictr, t_glob, t_auc, t_f1, t_mae_r, t_rmse_r, t_mae_c, t_rmse_c = evaluate(
        model, test_loader, args.lambda_loss, threshold=0.004439
    )

    test_metrics = {
        'auc': t_auc, 'f1': t_f1,
        'mae_roi': t_mae_r, 'rmse_roi': t_rmse_r,
        'mae_cvr': t_mae_c, 'rmse_cvr': t_rmse_c
    }
    log_experiment_results(args, test_metrics)
    print(f"[FINAL TEST RESULT]")
    print(f"iCTR AUC: {t_auc:.4f} | F1: {t_f1:.4f}")
    print(f"ROI MAE:  {t_mae_r:.4f} | RMSE: {t_rmse_r:.4f}")
    print(f"CVR MAE:  {t_mae_c:.4f} | RMSE: {t_rmse_c:.4f}")


def train(train_loader, val_loader, model, args):
    writer = SummaryWriter(log_dir=f'runs/{args.exp_name}')
    checkpoint_path = f'outputs/{args.exp_name}/best_model.pth'
    early_stopping = EarlyStopping(patience=args.patience, checkpoint_path=checkpoint_path)
    
    # Warm-up
    prepare_model_for_warmup(model)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    def get_lr_scheduler(opt):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=3, min_lr=1e-7
        )
    scheduler = get_lr_scheduler(optimizer)
    
    is_full_finetuning = False
    for epoch in range(args.epochs):
        # Unfreeze all parameters
        if epoch >= args.warmup_epochs and not is_full_finetuning:
            optimizer = unfreeze_all_layers(model, base_lr=args.lr)
            scheduler = get_lr_scheduler(optimizer)
            is_full_finetuning = True
            print(f">>> Epoch {epoch}: Switched to Full Fine-tuning mode.")

        train_l, train_ictr, train_glob = train_one_epoch(
            model, train_loader, optimizer, args.lambda_loss, writer, epoch
        )

        # Evaluate (threshold=0.004439)
        v_l, v_ictr, v_glob, v_auc, v_f1, v_mae_r, v_rmse_r, v_mae_c, v_rmse_c = evaluate(
            model, val_loader, args.lambda_loss, threshold=0.004439
        )

        scheduler.step(v_auc)

        # Keep log to Tensorboard
        writer.add_scalar('Loss_Detail/Train_iCTR', train_ictr, epoch)
        writer.add_scalar('Loss_Detail/Train_Global', train_glob, epoch)
        writer.add_scalar('Loss_Detail/Val_iCTR', v_ictr, epoch)
        writer.add_scalar('Loss_Detail/Val_Global', v_glob, epoch)
        writer.add_scalar('Metrics/iCTR_AUC', v_auc, epoch)
        writer.add_scalar('Metrics/ROI_MAE', v_mae_r, epoch)
        writer.add_scalar('Metrics/CVR_MAE', v_mae_c, epoch)

        print(f"[*] Epoch {epoch+1} Summary:")
        print(f"    Train Loss: {train_l:.4f} | Val Loss: {v_l:.4f}")
        print(f"    Train_iCTR Loss: {train_ictr:.4f} | Val Loss: {v_ictr:.4f}")
        print(f"    Train_Global Loss: {train_glob:.4f} | Val Loss: {v_glob:.4f}")
        print(f"    iCTR AUC: {v_auc:.4f} | ROI MAE: {v_mae_r:.4f} | CVR MAE: {v_mae_c:.4f}")

        early_stopping(-v_auc, model)
        if early_stopping.early_stop:
            print("!!! Early stopping triggered. Training finished.")
            break

    writer.close()
    print(f"Training of {args.exp_name} completed.")

if __name__ == "__main__":
    main()