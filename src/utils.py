# src/utils.py
import json
import logging
from datetime import datetime
from pathlib import Path


def setup_logger():
    """配置日志显示"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    return logging.getLogger("BiShe_Logger")


def save_json(data, filename, folder_path):
    """保存处理后的数据到JSON"""
    # 生成带时间戳的文件名，防止覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = Path(folder_path) / f"{filename.stem}_processed_{timestamp}.json"

    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    return full_path