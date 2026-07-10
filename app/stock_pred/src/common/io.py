"""I/O 工具函数：读写 YAML / JSON / CSV / Pickle"""

import json
import pickle
import csv
from pathlib import Path
from typing import Any, Dict, List

import yaml


def read_yaml(path: str) -> dict:
    """读取 YAML 配置文件"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, path: str) -> None:
    """写入 YAML 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def save_json(data: Any, path: str) -> None:
    """保存为 JSON 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """加载 JSON 文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(data: Any, path: str) -> None:
    """保存为 Pickle 文件"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def write_csv(data: List[Dict], path: str) -> None:
    """写入 CSV 文件 (列表字典 -> CSV)"""
    if not data:
        raise ValueError("data 列表为空，无法写入 CSV")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
