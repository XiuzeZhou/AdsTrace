import os
import json
import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer

def custom_collate_fn(batch):
    max_frames = max([item['frames'].shape[0] for item in batch])
    max_speech = max([item['speech'].shape[0] for item in batch])
    
    batched_data = {k: [] for k in batch[0].keys()}
    batched_data['speech_mask'] = []
    
    for item in batch:
        # Padding frames
        pad_f = max_frames - item['frames'].shape[0]
        frames = F.pad(item['frames'], (0, 0, 0, 0, 0, 0, 0, pad_f)) if pad_f > 0 else item['frames']
            
        # Padding speech & mask
        pad_s = max_speech - item['speech'].shape[0]
        speech = F.pad(item['speech'], (0, pad_s)) if pad_s > 0 else item['speech']
        s_mask = torch.ones(item['speech'].shape[0], dtype=torch.long)
        speech_mask = F.pad(s_mask, (0, pad_s), value=0) if pad_s > 0 else s_mask
            
        # Padding ictr
        frame_len = item['frames'].shape[0]
        ictr_aligned = item['ictr'][:frame_len] if frame_len > 0 else torch.empty(0, dtype=item['ictr'].dtype)
        pad_i = max_frames - ictr_aligned.shape[0]
        ictr = F.pad(ictr_aligned, (0, pad_i)) if pad_i > 0 else ictr_aligned
            
        batched_data['frames'].append(frames)
        batched_data['speech'].append(speech)
        batched_data['speech_mask'].append(speech_mask)
        batched_data['text_ids'].append(item['text_ids'])
        batched_data['text_mask'].append(item['text_mask'])
        batched_data['ictr'].append(ictr)
        batched_data['roi_cvr'].append(item['roi_cvr'])
        batched_data['T'].append(torch.tensor(item['T']))
        
    return {k: torch.stack(v) for k, v in batched_data.items()}

class AdsTraceDataset(Dataset):
    def __init__(self, root_dir, transform=None, bert_path='bert-base-chinese'):
        self.root_dir = root_dir
        self.transform = transform
        
        self.tags_df = pd.read_csv(os.path.join(root_dir, "tags_cn.csv"), low_memory=False).fillna("")
        self.tags_df['ID'] = pd.to_numeric(self.tags_df['ID'], errors='coerce').fillna(-1).astype(int).astype(str)
        self.product_info = json.load(open(os.path.join(root_dir, "products_cn.json"), encoding='utf-8'))
        self.product_lookup = {p['name']: f"{p['intro']} {p['selling_points']}" for p in self.product_info}
        
        self.video_ids = [f.split('.')[0] for f in os.listdir(os.path.join(root_dir, "ictr"))]
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)

        self.audio_cache = {}
        max_len = 46*16000  # 46s * 16k
        print("Pre-loading audios to Memory...")
        for vid in tqdm(self.video_ids):
            audio_path = os.path.join(self.root_dir, f"audios_16k/{vid}.wav")
            speech, _ = torchaudio.load(audio_path)
            if speech.shape[1] > max_len:
                speech = speech[:, :max_len]

            self.audio_cache[vid] = speech.squeeze(0)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        
        ictr_df = pd.read_csv(os.path.join(self.root_dir, f"ictr/{vid}.csv"))
        ictr_all = torch.tensor(ictr_df['ictr'].values, dtype=torch.float32)
        
        # The length of ictr_all may be greater than the number of extracted frames.
        frames_dir = os.path.join(self.root_dir, f"frames/{vid}")
        existing_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
        T = len(existing_frames)
        ictr_seq = ictr_all[:T]

        target_row = self.tags_df[self.tags_df['ID'] == vid].iloc[0]
        #roi_cvr = torch.tensor([target_row['ROI'], target_row['CVR']], dtype=torch.float32)
        roi_log = np.log1p(target_row['ROI'])
        cvr_log = np.log1p(target_row['CVR']) 
        roi_cvr = torch.tensor([roi_log, cvr_log], dtype=torch.float32)

        frames = []
        for i in range(1, T + 1):
            # 01.jpg, 02.jpg, ...
            img_path = os.path.join(frames_dir, f"{i:02d}.jpg")
            img = Image.open(img_path).convert('RGB')
            frames.append(self.transform(img))
        frames = torch.stack(frames)

        speech = self.audio_cache[vid]
        
        # video type: 视频特点, streamer: 主播, editing techniques: 剪辑技术, promotional mechanisms: 促销活动
        meta = f"{target_row['视频特点']} {target_row['主播']} {target_row['剪辑技术']} {target_row['促销活动']}" # "video type" "streamer" "editing techniques" "promotional mechanisms"
        transcript = json.load(open(os.path.join(self.root_dir, f"transcripts/{vid}.json"), encoding='utf-8'))['full_text']
        full_text = f"Meta: {meta} [SEP] Transcript: {transcript} [SEP] Description: {self.product_lookup.get(target_row['商品名字'], '')}" # product name: 商品名字
        text_tokens = self.tokenizer(full_text, padding='max_length', max_length=512, truncation=True, return_tensors="pt")

        return {
            "frames": frames, "speech": speech,
            "text_ids": text_tokens['input_ids'].squeeze(0),
            "text_mask": text_tokens['attention_mask'].squeeze(0),
            "ictr": ictr_seq, "roi_cvr": roi_cvr, "T": T
        }