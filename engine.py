import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error, mean_squared_error
from torch.cuda.amp import autocast, GradScaler

def train_one_epoch(model, loader, optimizer, lambda_loss, writer, epoch):
    model.train()
    scaler = GradScaler()
    total_l, ictr_l, glob_l = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")
    for i, batch in enumerate(pbar):
        f = batch['frames'].cuda()
        s = batch['speech'].cuda()
        sm = batch['speech_mask'].cuda()
        tid = batch['text_ids'].cuda()
        tm = batch['text_mask'].cuda()
        target_ictr, target_global = batch['ictr'].cuda(), batch['roi_cvr'].cuda()
        
        optimizer.zero_grad()

        with autocast():
            p_ictr, p_global = model(f, s, sm, tid, tm, seq_lens=batch['T'])

            batch_ictr_loss = torch.tensor(0.0, device=f.device)
            batch_glob_loss = torch.tensor(0.0, device=f.device)

            if p_ictr is not None:
                bce_raw = F.binary_cross_entropy_with_logits(p_ictr, target_ictr, reduction='none')
                mask = torch.zeros_like(p_ictr)
                for idx, t in enumerate(batch['T']):
                    mask[idx, :t] = 1.0
                batch_ictr_loss = (bce_raw * mask).sum() / (mask.sum() + 1e-8)
            
            if p_global is not None:
                pred_roi, pred_cvr = p_global[:, 0], p_global[:, 1]
                target_roi, target_cvr = target_global[:, 0], target_global[:, 1]
                loss_roi = F.mse_loss(pred_roi, target_roi)
                loss_cvr = F.mse_loss(pred_cvr, target_cvr)
                batch_glob_loss = 0.01 * loss_roi + loss_cvr
            
            loss = batch_glob_loss + lambda_loss * batch_ictr_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_l += loss.item()
        ictr_l += batch_ictr_loss.item()
        glob_l += batch_glob_loss.item()
        
        pbar.set_postfix({
            'L': f"{loss.item():.4f}", 
            'iCTR': f"{batch_ictr_loss.item():.4f}", 
            'ROI': f"{batch_glob_loss.item():.4f}"
        })

    num_batches = len(loader)
    return total_l / num_batches, ictr_l / num_batches, glob_l / num_batches

def evaluate(model, loader, lambda_loss, threshold):
    model.eval()
    val_loss, val_ictr_l, val_glob_l = 0.0, 0.0, 0.0
    p_ictr_all, t_ictr_all = [], []
    p_roi, t_roi = [], []
    p_cvr, t_cvr = [], []

    with torch.no_grad():
        for batch in loader:
            f = batch['frames'].cuda()
            s = batch['speech'].cuda()
            sm = batch['speech_mask'].cuda()
            tid = batch['text_ids'].cuda()
            tm = batch['text_mask'].cuda()
            target_ictr, target_global = batch['ictr'].cuda(), batch['roi_cvr'].cuda()
            p_ictr, p_global = model(f, s, sm, tid, tm, seq_lens=batch['T'])
            
            l_ictr_batch = torch.tensor(0.0).cuda()
            if p_ictr is not None:
                # iCTR Loss (with Mask)
                bce_raw = F.binary_cross_entropy_with_logits(p_ictr, target_ictr, reduction='none')
                v_mask = torch.zeros_like(p_ictr).cuda()
                for idx, t in enumerate(batch['T']):
                    v_mask[idx, :t] = 1.0
                l_ictr_batch = (bce_raw * v_mask).sum() / (v_mask.sum() + 1e-9)

                pi_prob = torch.sigmoid(p_ictr)
                for b_idx, t in enumerate(batch['T']):
                    p_ictr_all.extend(pi_prob[b_idx, :t].cpu().numpy())
                    t_ictr_all.extend(target_ictr[b_idx, :t].cpu().numpy())
            
            l_glob_batch = torch.tensor(0.0).cuda()
            if p_global is not None:
                # Global Loss (ROI + CVR*100)
                pred_roi, pred_cvr = p_global[:, 0], p_global[:, 1]
                target_roi, target_cvr = target_global[:, 0], target_global[:, 1]
                loss_roi = F.mse_loss(pred_roi, target_roi)
                loss_cvr = F.mse_loss(pred_cvr, target_cvr)
                l_glob_batch = 0.01 * loss_roi + loss_cvr
                
                p_roi.extend(pred_roi.cpu().numpy())
                t_roi.extend(target_roi.cpu().numpy())
                p_cvr.extend(pred_cvr.cpu().numpy())
                t_cvr.extend(target_cvr.cpu().numpy())

            val_loss += (l_glob_batch + lambda_loss * l_ictr_batch).item()
            val_ictr_l += l_ictr_batch.item()
            val_glob_l += l_glob_batch.item()

    num_batches = len(loader)
    
    # Metric Calculation
    auc, f1, mae_roi, rmse_roi, mae_cvr, rmse_cvr = 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
    if len(p_ictr_all) > 0:
        p_ictr_all = np.nan_to_num(p_ictr_all, nan=0.5)
        t_bin = (np.array(t_ictr_all) > threshold).astype(int)
        if len(np.unique(t_bin)) > 1:
            auc = roc_auc_score(t_bin, p_ictr_all)
            p_bin = (np.array(p_ictr_all) > threshold).astype(int)
            f1 = f1_score(t_bin, p_bin, zero_division=0)

    if len(p_roi) > 0:
        p_roi_real = np.expm1(np.array(p_roi)) # exp(x) - 1
        t_roi_real = np.expm1(np.array(t_roi))
        p_cvr_real = np.expm1(np.array(p_cvr)) # exp(x) - 1
        t_cvr_real = np.expm1(np.array(t_cvr))
        mae_roi = mean_absolute_error(t_roi_real, p_roi_real)
        rmse_roi = np.sqrt(mean_squared_error(t_roi_real, p_roi_real))
        mae_cvr = mean_absolute_error(t_cvr_real, p_cvr_real)
        rmse_cvr = np.sqrt(mean_squared_error(t_cvr_real, p_cvr_real))
    
    return val_loss/num_batches, val_ictr_l/num_batches, val_glob_l/num_batches, auc, f1, mae_roi, rmse_roi, mae_cvr, rmse_cvr