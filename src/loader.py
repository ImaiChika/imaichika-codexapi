# src/loader.py
import json
from pathlib import Path


def load_json_data(filepath):
    """
    加载原始JSON数据
    :param filepath: 文件的完整路径
    :return: 包含字典的列表
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"数据文件未找到: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # 简单的格式校验
            if isinstance(data, list):
                return data
            else:
                raise ValueError("JSON格式错误: 根节点应该是一个列表 []")
        except json.JSONDecodeError:
            raise ValueError(f"文件损坏或非标准JSON: {filepath}")