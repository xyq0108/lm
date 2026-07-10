#!/bin/bash
set -e

echo "[TEST] 开始预测..."

python ./stock_pred/src/predict.py \
    --config ./stock_pred/src/config/ensemble.yaml \
    --model_dir ./model/final_ensemble \
    --output ./output/result.csv

echo "[TEST] 预测完成."
echo "[TEST] 结果文件: ./output/result.csv"

# 检查结果文件
if [ -f "./output/result.csv" ]; then
    echo "[TEST] 结果文件已生成 ✓"
    echo "---"
    cat ./output/result.csv
    echo "---"
else
    echo "[ERROR] 结果文件未生成！"
    exit 1
fi
