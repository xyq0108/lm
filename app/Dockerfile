# 2026 大数据挑战赛 — Docker 复现环境
# 基于 Python 3.10 轻量镜像

FROM python:3.10-slim

LABEL maintainer="challenge2026"
LABEL description="2026 大数据挑战赛 — 股票收益预测"

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码和数据
COPY . .

# 初始化目录
RUN chmod +x init.sh && ./init.sh

# 训练入口
CMD ["bash", "train.sh"]
