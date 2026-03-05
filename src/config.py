# src/config.py
import os
from pathlib import Path

# === 路径配置 ===
# 动态获取项目根目录 (E:\imaichika_whu)
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
MODEL_CACHE_DIR = str(ROOT_DIR / "models_cache") # 模型将下载到 E:\imaichika_whu\models_cache

# 确保输出目录存在
os.makedirs(DATA_PROC_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# === 镜像站配置 ===
HF_MIRROR_URL = "https://hf-mirror.com"
# 模型名称
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
# LLM_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # 这是目前最接近0.6B的SOTA小模型
# === 正则表达式模式 (PPT提及的PII扫描) ===
# 简单的手机号、USDT地址、银行卡号匹配
REGEX_PATTERNS = {
    # 手机号：强制匹配 11 或 12 位（部分国际前缀或特殊号段），前后绝不能有数字
    "mobile_cn": r"(?<!\d)1[3-9]\d{10}(?!\d)|(?<!\d)1[3-9]\d{9}(?!\d)", 
    
    # 银行卡：覆盖 13 到 19 位所有可能，优先匹配长的，防止被截断
    "bank_card": r"(?<!\d)[1-9]\d{15,18}(?!\d)|(?<!\d)[1-9]\d{12,14}(?!\d)",
    
    # 身份证：中国 18 位身份证
    "id_card": r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
    
    # 其他保持不变
    "usdt_address": r"(?<![A-Za-z0-9])T[A-Za-z1-9]{33}(?![A-Za-z0-9])",
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
}

# === 关键词字典 (用于初步打标) ===
# 参考PPT中的Context Rules
KEYWORD_RULES = {
    "fraud": ["跑分", "车队", "赔付", "通道", "U出", "点位", "码商", "骗子"],
    "gambling": ["注单", "包赢", "回血", "上岸", "下注", "特码"],
    "trade": ["收", "出", "求购", "低价", "实名", "汇率", "CVV"],
    "social": ["互粉", "小姐姐", "约", "同城", "兼职"]
}
# === NLP 配置 ===
# 简单的中文停用词表（也可以加载外部txt文件）
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
    "会", "着", "没有", "看", "好", "自己", "这", "那", "如何", "这个", "那个", "吗", "吧", "被", "让", "给", "但是", "还是",
    "UTC+8", "Bot", "System", "加入", "频道", "发言", "惩罚", "禁言"  # 针对你数据中的系统消息添加的过滤词
}

# 系统消息特征词（用于过滤脏数据）
SYSTEM_MSG_KEYWORDS = ["请先关注如下频道", "加入群组后才能发言", "禁言到", "已被管理员"]
# === 模型路径配置 ===


