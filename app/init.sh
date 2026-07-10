#!/bin/bash
set -e

echo "[INIT] 初始化项目目录..."

mkdir -p ./output
mkdir -p ./temp/features_cache
mkdir -p ./temp/validation_cache
mkdir -p ./temp/predictions_cache
mkdir -p ./reports
mkdir -p ./model

echo "[INIT] 项目目录检查完成."

# 检查必要文件
if [ ! -f "./data/train.csv" ]; then
    echo "[WARNING] data/train.csv 不存在！请将训练数据放在 data/ 目录。"
fi

echo "[INIT] 初始化完成."
