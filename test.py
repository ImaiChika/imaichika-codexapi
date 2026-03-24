import json

# 替换成你的文件路径
path = r"E:\imaichika_whu\data\raw\nchannel_hc8668.json"
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 检查前 10 个包含 @ 的消息
found = 0
for msg in data:
    if "@" in msg.get("text", ""):
        print(f"找到提及消息: {msg['text']}")
        found += 1
    if found > 5: break

if found == 0:
    print("结论：你的数据中确实没有 @ 符号，必须使用‘上下文邻近’逻辑来构建图。")