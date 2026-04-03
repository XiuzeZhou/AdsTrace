#!/bin/bash

DATA_PATH="./datasets/AdsTrace"
BERT_PATH="./pretrained_models/bert-base-chinese"
WAV2VEC_PATH="./pretrained_models/wav2vec2-large-xlsr-53-chinese-zh-cn"
SWIN_PATH="./pretrained_models/swin_base_patch4_window7_224"

HIDDEN_SIZE=768
BATCH_SIZE=8
WARMUP=5
EPOCHS=100
PATIENCE=7
LR=1e-5
LAMBDA=10.0
LAYERS=2

mkdir -p outputs
mkdir -p runs


EXP_NAME="TAMAN_L${LAYERS}_Lam${LAMBDA}_LR${LR}"

echo "================================================================"
echo "Starting Experiment: $EXP_NAME"
echo "Params: Layers=$LAYERS, Lambda=$LAMBDA, Base_LR=$LR"
echo "================================================================"

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/bin/python -u ./main.py \
    --data_path "$DATA_PATH" \
    --bert_path "$BERT_PATH" \
    --wav2vec_path "$WAV2VEC_PATH" \
    --swin_path "$SWIN_PATH" \
    --num_layers "$LAYERS" \
    --lambda_loss "$LAMBDA" \
    --hidden_size "$HIDDEN_SIZE" \
    --exp_name "$EXP_NAME" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --warmup_epochs "$WARMUP" \
    --lr "$LR" \
    --patience "$PATIENCE"
    
echo ">>> Finished $EXP_NAME"
echo ""

echo "All experiments completed! Check outputs/summary_results.csv for details."