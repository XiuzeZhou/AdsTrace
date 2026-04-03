# AdsTrace: A Multimodal Dataset for Second-by-Second CTR and Advertising Effectiveness Prediction in Short-video Ads

[![Dataset](https://img.shields.io/badge/Dataset-AdsTrace-blue)](https://huggingface.co/datasets/Xiuze/AdsTrace)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

This repository contains the dataset tools for **AdsTrace** benchmark dataset, and the implementation of **TAMAN**. TAMAN aligns hierarchical marketing metadata (Text + Audio) with temporal visual tokens to predict second-by-second engagement (iCTR) and global business metrics (ROI/CVR).

## 🛠️ Installation
```bash
# Clone the repository
git clone [https://github.com/XiuzeZhou/AdsTrace.git](https://github.com/XiuzeZhou/AdsTrace.git)
cd AdsTrace

# Install dependencies
pip install -r requirements.txt
```

## 📊 Quick Start
### 1. Data Preparation
Place the pre-trained models (Swin-B, BERT-base-chinese, Wav2Vec 2.0) in ./pretrained_models/ and organize your dataset:
