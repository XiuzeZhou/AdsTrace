import torch
import torch.nn as nn
import timm
from transformers import BertModel, Wav2Vec2Model

class TAMAN(nn.Module):
    def __init__(self, 
                 hidden_size=768, 
                 num_layers=2, 
                 lstm_layers=2, 
                 dropout=0.5,
                 swin_path='local-dir:./pretrained_models/swin_base_patch4_window7_224',
                 bert_path='./pretrained_models/bert-base-chinese', 
                 wav2vec_path="./pretrained_models/wav2vec2-large-xlsr-53-chinese-zh-cn"
                 ):
        super().__init__()
        self.visual_enc = timm.create_model(swin_path, pretrained=True, num_classes=0)
        self.bert = BertModel.from_pretrained(bert_path)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec_path)
        
        self.vis_proj = nn.Linear(1024, hidden_size)
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_size)
        
        self.audio_proj = nn.Linear(1024, hidden_size)
        self.hme_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.hme_feat = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.ReLU())
        self.context_fusion = nn.Linear(hidden_size * 2, hidden_size)
        
        self.taa_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True) 
            for _ in range(num_layers)
        ])
        self.taa_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.bilstm = nn.LSTM(hidden_size, hidden_size // 2, num_layers=lstm_layers, bidirectional=True, batch_first=True)
        
        # iCTR Head: [Linear -> ReLU -> Dropout -> Linear]
        self.ictr_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # ROI/CVR Head: [Linear -> ReLU -> Dropout -> Linear]
        self.global_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

        backbone_names = ['visual_enc', 'bert', 'wav2vec2']
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if not any(bb in name for bb in backbone_names):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, frames, speech, speech_mask, text_ids, text_mask, seq_lens=None, return_attn=False):
        B, T, C, H, W = frames.shape

        z_vis = self.visual_enc(frames.view(B*T, C, H, W)).view(B, T, -1)
        z_vis = self.vis_proj(z_vis) # [B, T, d]

        e_text_seq = self.bert(text_ids, attention_mask=text_mask).last_hidden_state
        e_text_seq = self.text_proj(e_text_seq) # [B, L_text, d]
        
        e_audio_seq = self.wav2vec2(speech, attention_mask=speech_mask).last_hidden_state
        e_audio_seq = self.audio_proj(e_audio_seq) # [B, L_audio, d]

        audio_out = self.wav2vec2(speech, attention_mask=speech_mask).last_hidden_state
        input_lengths = speech_mask.sum(dim=-1)
        out_lengths = self.wav2vec2._get_feat_extract_output_lengths(input_lengths)
        audio_mask = torch.arange(audio_out.size(1), device=audio_out.device).expand(B, audio_out.size(1)) < out_lengths.unsqueeze(1)
        e_audio = (audio_out * audio_mask.unsqueeze(-1)).sum(dim=1) / (out_lengths.unsqueeze(1) + 1e-8)
        e_audio = self.audio_proj(e_audio) # [B, hidden_size]
        combined_context = torch.cat([e_text_seq, e_audio.unsqueeze(1).expand(-1, e_text_seq.size(1), -1)], dim=-1)
        h_context = self.context_fusion(combined_context)

        key_padding_mask = (text_mask == 0)
        
        attn_weights_list = []
        x = z_vis
        for attn, norm in zip(self.taa_layers, self.taa_norms):
            res, weights = attn(
                x, h_context, h_context, 
                key_padding_mask=key_padding_mask,
                average_attn_weights=True
            )
            x = norm(x + res)
            if return_attn:
                attn_weights_list.append(weights)

        h_seq, _ = self.bilstm(x)
        
        # Multi-task learning
        pred_ictr = self.ictr_head(h_seq).squeeze(-1)
        
        if seq_lens is not None:
            last_step_indices = (seq_lens - 1).to(h_seq.device) # [B]
            gather_indices = last_step_indices.view(B, 1, 1).expand(B, 1, h_seq.size(2))
            last_hidden = h_seq.gather(1, gather_indices).squeeze(1) # [B, hidden_size]
            pred_roi_cvr = self.global_head(last_hidden)
        else:
            pred_roi_cvr = self.global_head(h_seq[:, -1, :])
            
        if return_attn:
            return pred_ictr, pred_roi_cvr, attn_weights_list[-1]
        return pred_ictr, pred_roi_cvr