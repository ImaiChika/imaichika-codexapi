import os
from pathlib import Path

# === Path Config ===
ROOT_DIR = Path(__file__).parent.parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"
DATA_DB_DIR = ROOT_DIR / "data" / "db"
ACCOUNT_DB_DIR = DATA_DB_DIR / "accounts"
MODEL_CACHE_DIR = str(ROOT_DIR / "models_cache")

os.makedirs(DATA_PROC_DIR, exist_ok=True)
os.makedirs(DATA_DB_DIR, exist_ok=True)
os.makedirs(ACCOUNT_DB_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# === Model Config ===
HF_MIRROR_URL = "https://hf-mirror.com"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# === Runtime / Cost Control ===
LLM_MIN_L1_SCORE = 45
LLM_MIN_L2_SCORE = 30

# === RAG / Vector Store ===
VECTOR_DB_DIR = DATA_DB_DIR / "chroma"
VECTOR_COLLECTION = "chat_messages"
RAG_TOP_K = 12
RAG_MAX_DOC_CHARS = 240

# === Lightweight Agent / Reflection ===
AGENT_ENABLE_LIGHT_REACT = True
AGENT_ENABLE_REFLECTION = True
AGENT_MAX_INTENT_CHARS = 24

# === Report Rendering (tunable) ===
# You can tune report scale here:
# - REPORT_CORE_USER_TOP_K controls "核心人物包括" total length.
# - REPORT_PROFILE_LINE_TOP_K controls detailed bullet lines in section 2.
# - REPORT_CLUE_CHAIN_TOP_K controls how many clue chains are printed in report tail.
REPORT_CORE_USER_TOP_K = 16
REPORT_PROFILE_LINE_TOP_K = 24
REPORT_CLUE_CHAIN_TOP_K = 20

# Group name hints used for role disambiguation.
SUSPECT_GROUP_HINTS = ["suspect", "internal", "ops", "分赃", "内部分工", "内部操作", "核心"]
VICTIM_GROUP_HINTS = ["victim", "family", "related", "受害", "亲友", "投诉", "维权"]

# === Regex Patterns (PII / sensitive evidence) ===
# NOTE:
# - name_cn / address_cn are context-triggered to reduce false positives.
REGEX_PATTERNS = {
    "mobile_cn": r"(?<!\d)1[3-9]\d{9}(?!\d)",
    "mobile_cn_masked": r"(?<!\d)1[3-9]\d(?:[\d\*Xx]){5,7}\d{2,4}(?!\d)",
    "bank_card": r"(?<!\d)[1-9]\d{15,18}(?!\d)",
    "id_card": r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)",
    "id_card_masked": r"(?<!\d)[1-9]\d{5}(?:[\d\*Xx]){8,12}\d{2,4}(?!\d)",
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "usdt_address": r"(?<![A-Za-z0-9])T[A-Za-z1-9]{33}(?![A-Za-z0-9])",
    "payment_address": r"(?:(?:收款|充值|付款|打款|入金)(?:地址|账号)?|usdt(?:地址)?|钱包地址)\s*[:：]?\s*([A-Za-z0-9]{20,64})",
    "qq_number": r"(?:QQ|qq|Q号|q号|企鹅号)\s*[:：]?\s*([1-9]\d{4,11})",
    "name_cn": r"(?:^|[\s\n,，;；/|])(?:户名|姓名|机主|我叫|我是)\s*[:：]?\s*([\u4e00-\u9fa5·]{2,12})",
    "name_en": r"(?:机主|查(?:档)?|查询|目标|资料)\s*[:：]?\s*([A-Z][a-z]+(?:[A-Z][a-z]+){1,3})",
    "address_cn": r"(?:住址|地址|接头地点|见面地点|见面地址|到)\s*[:：]?\s*([^\n，。]{6,80})",
}

# === Topic / role keywords ===
KEYWORD_RULES = {
    "fraud": [
        "跑分", "车队", "通道", "码商", "骗子", "黑卡", "保证金", "下发",
        "拆开卖", "重复卖", "统一口径", "假地址", "别发全", "全量", "半套", "模糊定位",
    ],
    "gambling": ["赌", "注单", "包赢", "上岸", "返水"],
    "trade": [
        "收U", "出U", "汇率", "实名", "CVV", "银行卡", "户名", "收款地址", "钱包地址",
        "三要素", "机主", "户籍", "家谱", "近照", "尾号", "开头", "结尾", "尾四",
    ],
    "social": ["兼职", "同城", "私聊", "群友"],
}

# === NLP Stopwords / system message filter ===
STOPWORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一个", "也", "很", "到", "要", "去", "你",
    "他", "她", "它", "我们", "你们", "他们", "这", "那", "这个", "那个", "吗", "呢", "啊", "吧", "被", "让", "给",
    "系统", "加入", "群组", "频道", "发言", "管理", "禁言", "用户", "消息",
}

SYSTEM_MSG_KEYWORDS = [
    "已加入群组",
    "已被移出群组",
    "请先关注",
    "加入后才能发言",
    "系统提示",
]

# Used by low-cost heuristic gate in main.py
HIGH_SIGNAL_KEYWORDS = [
    "卡号", "户名", "下发", "保证金", "身份证", "手机号", "QQ", "收款地址", "钱包地址",
    "报警", "被骗", "拉黑", "没到账", "住址", "接头地点", "三要素", "户籍", "家谱", "近照",
    "尾号", "开头", "结尾", "前六", "尾四", "半套", "全量", "别发全", "别全名", "模糊定位",
    "统一口径", "重复卖", "拆开卖", "主推", "后补", "假地址", "只发到县", "不发到门牌",
    "机主", "备注", "家里人号", "售后",
]

IMPLICIT_SIGNAL_KEYWORDS = [
    "尾号", "开头", "结尾", "前六", "尾四", "半套", "全量", "别发全", "别全名", "模糊定位",
    "统一口径", "重复卖", "拆开卖", "主推", "后补", "假地址", "只发到县", "不发到门牌",
    "机主", "家里人号", "备注写", "按口径", "别上群", "别写真名", "只提", "老地址",
]
