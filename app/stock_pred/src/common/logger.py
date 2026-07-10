"""结构化日志管理"""
import sys
import logging
from pathlib import Path


def setup_logger(name: str = "stock_pred", log_file: str = None,
                 level: str = "INFO") -> logging.Logger:
    """配置并返回 logger 实例"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "stock_pred") -> logging.Logger:
    """获取已存在的 logger"""
    return logging.getLogger(name)
