#!/bin/bash
set -e

echo "[TRAIN] 开始训练..."

python ./stock_pred/src/train.py \
    --config ./stock_pred/src/config/ensemble.yaml \
    --mode final_train

echo "[TRAIN] 训练完成."
