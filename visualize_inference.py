import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import jieba
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from transformers import BertTokenizer
from dataset import AdsTraceDataset, custom_collate_fn
from model import TAMAN
import json
import seaborn as sns

# Set font for Chinese
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def aggregate_weights_to_words(full_text_string, bert_tokens, bert_weights):
    # Filtering special tokens of BERT
    special_tokens = {'[CLS]', '[SEP]', '[PAD]', '[MASK]'}
    
    # Word segmentation
    words = list(jieba.cut(full_text_string))
    
    word_weights = []
    token_ptr = 0
    
    for word in words:
        word = word.strip()
        if not word or word in special_tokens: 
            continue
        
        matched_weights = []
        for char in word.lower():
            for i in range(token_ptr, len(bert_tokens)):
                token = bert_tokens[i].replace("##", "").lower()
                if token == char:
                    matched_weights.append(bert_weights[i])
                    token_ptr = i + 1
                    break
        
        if matched_weights:
            avg_score = sum(matched_weights) / len(matched_weights)
            word_weights.append((word, avg_score))
            
    return word_weights


def plot_case_study_multidim(frame, ictr_val, word_weights, video_id, save_path):
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # Show the key frame
    ax0 = fig.add_subplot(gs[0])
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img = frame.permute(1, 2, 0).cpu().numpy()
    img = np.clip(std * img + mean, 0, 1)
    ax0.imshow(img)
    ax0.set_title(f"ID: {video_id}\n iCTR: {ictr_val:.4f}", fontsize=14)
    ax0.axis('off')

    # Bar of Words
    ax1 = fig.add_subplot(gs[1])
    if not word_weights:
        ax1.text(0.5, 0.5, "No words matched", ha='center')
    else:
        final_list = [(w, s) for w, s in word_weights if len(w) > 1 or ('\u4e00' <= w <= '\u9fff')]
        final_list.sort(key=lambda x: x[1])
        
        top_k = min(15, len(final_list))
        plot_data = final_list[-top_k:]
        
        ax1.barh([x[0] for x in plot_data], [x[1] for x in plot_data], color='#2ca02c')
        ax1.set_title("Cross-modal Semantic Alignment (What: Text + Audio)", fontsize=14)
        ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()

def plot_attention_evolution(video_id, ictr_probs, attn_matrix, tokens, save_path):
    T = len(ictr_probs)
    L = len(tokens)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, L*0.2), 10), 
                                   gridspec_kw={'height_ratios': [1, 3]})

    # 1. iCTR Evolution
    ax1.plot(range(T), ictr_probs, marker='o', color='#d62728', linewidth=2)
    ax1.fill_between(range(T), ictr_probs, alpha=0.1, color='#d62728')
    ax1.set_title(f"Video {video_id}: Predict iCTR Evolution", fontsize=15)
    ax1.set_ylabel("Click Probability")
    ax1.set_xticks(range(T))
    ax1.grid(True, alpha=0.3)

    # Attention Map
    clean_tokens = [t.replace("##", "") for t in tokens]
    # Normlize attention scores
    norm_attn = attn_matrix / (attn_matrix.max(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(norm_attn, ax=ax2, cmap="YlGnBu", cbar_kws={'label': 'Attention Weight'},
                xticklabels=clean_tokens, yticklabels=[f"{i}s" for i in range(T)])
    
    ax2.set_title("Temporal offset (Acoustic-Textual Alignment)", fontsize=15)
    ax2.set_xlabel("Fused Token (Text + Audio Context)")
    ax2.set_ylabel("Time (second)")
    
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000)
    plt.close()

def run_evolution_visualization(args, num_cases=5):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    dataset = AdsTraceDataset(root_dir=args.data_path, transform=transform, bert_path=args.bert_path)
    
    with open(os.path.join(args.data_path, "split.json"), "r") as f:
        splits = json.load(f)
    
    id_to_idx = {vid: i for i, vid in enumerate(dataset.video_ids)}
    test_idx = [id_to_idx[v] for v in splits['test'] if v in id_to_idx]
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=1, collate_fn=custom_collate_fn)

    model = TAMAN(hidden_size=args.hidden_size, swin_path=args.swin_path).cuda()
    model.load_state_dict(torch.load(f'outputs/{args.exp_name}/best_model.pth'))
    model.eval()

    os.makedirs("visualizations", exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_cases: break
            
            video_id = batch['video_ids'][0]
            p_ictr, _, attn_weights = model(
                batch['frames'].cuda(), batch['speech'].cuda(), batch['speech_mask'].cuda(),
                batch['text_ids'].cuda(), batch['text_mask'].cuda(),
                seq_lens=batch['T'], return_attn=True
            )
            
            probs = torch.sigmoid(p_ictr).squeeze(0).cpu().numpy()
            t_len = batch['T'][0].item()
            valid_text_len = batch['text_mask'][0].sum().item()
            
            full_attn = attn_weights.squeeze(0).cpu().numpy() # [T, L_ctx]
            time_text_attn = full_attn[:t_len, :valid_text_len]
            
            # Evolution Plot
            raw_ids = batch['text_ids'][0][:valid_text_len]
            tokens = tokenizer.convert_ids_to_tokens(raw_ids)
            evo_path = f"visualizations/{video_id}_evolution.png"
            plot_attention_evolution(video_id, probs[:t_len], time_text_attn, tokens, evo_path)

            # Case Study: choose the peak frame of iCTR
            t_peak = np.argmax(probs[:t_len])
            peak_weights = time_text_attn[t_peak]

            raw_ids = batch['text_ids'][0][:valid_text_len]
            tokens = tokenizer.convert_ids_to_tokens(raw_ids)

            full_sentence = tokenizer.decode(raw_ids, skip_special_tokens=True).replace(" ", "")

            word_level_weights = aggregate_weights_to_words(
                full_sentence, 
                tokens, 
                peak_weights.tolist()
            )

            case_path = f"visualizations/{video_id}_peak_analysis.png"
            plot_case_study_multidim(
                batch['frames'][0][t_peak], 
                probs[t_peak], 
                word_level_weights, 
                video_id, 
                case_path
            )
            print(f"Visualization completed for video: {video_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='TAMAN_Final')
    parser.add_argument('--data_path', type=str, default='./datasets/AdsTrace')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--bert_path', type=str, default='./pretrained_models/bert-base-chinese')
    parser.add_argument('--swin_path', type=str, default='local-dir:./pretrained_models/swin_base_patch4_window7_224')
    args = parser.parse_args()

    run_evolution_visualization(args, num_cases=2)