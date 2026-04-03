# AdsTrace: A Multimodal Dataset for Second-by-Second CTR and Advertising Effectiveness Prediction in Short-video Ads

[![Dataset](https://img.shields.io/badge/Dataset-AdsTrace-blue)](https://huggingface.co/datasets/Xiuze/AdsTrace)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

This repository contains the dataset tools for **AdsTrace** benchmark dataset, and the implementation of **TAMAN**. TAMAN aligns hierarchical marketing metadata (Text + Audio) with temporal visual tokens to predict second-by-second engagement (iCTR) and global business metrics (ROI/CVR).

## 💎AdsTrace Dataset
<p align="center">
<img src="figures/annotation.png" width="700">
</p>

## 🧠TAMAN Framework
<p align="center">
<img src="figures/framework.png" width="700">
</p>

## 🛠️ Installation
```bash
# Clone the repository
git clone https://github.com/XiuzeZhou/AdsTrace.git
cd AdsTrace

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Data Preparation

#### Dataset
1) Download the **AdsTrace** dataset from [Hugging Face](https://huggingface.co/): https://huggingface.co/datasets/Xiuze/AdsTrace

2) Unzip the .zip files

3) Organize the data as follows:
```
├── datasets/AdsTrace/
|   ├── audios_16k/
│   ├── frames/
│   ├── ictr/
│   ├── transcripts/
│   ├── products_cn.json
│   ├── products_en.json
│   ├── tags_cn.csv
│   └── split.json
```

#### Pre-trained Models

1) Download the pretrained models (Swin, BERT, Wav2Vec) [Hugging Face](https://huggingface.co/)

- **Swin-B**: [timm/swin_base_patch4_window7_224.ms_in1k](https://huggingface.co/timm/swin_base_patch4_window7_224.ms_in1k)
- **BERT-base-chinese**: [google-bert/bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese)
- **Wav2Vec**: [jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn)

2) Organize the data as follows:
```
├── pretrained_models/
|   ├── bert-base-chinese/
│   ├── swin_base_patch4_window7_224/
│   └── wav2vec2-large-xlsr-53-chinese-zh-cn/
```

### 2. Training
To train TAMAN with the optimal parameters:
```bash

```

### 3. Visualization
Generate case studies (Acoustic-Textual Alignment) for the test set:
```bash
python visualize_inference.py --exp_name TAMAN_Final --num_cases 5
```

## 📝 Citation
If you find our work useful in your research, please consider citing:
```

```

## 📄 License

- **Dataset**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

- **Code**: [Apache 2.0](https://www.google.com/search?q=LICENSE)

