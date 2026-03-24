import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# === 配置部分 ===
# 你的本地缓存绝对路径
CACHE_DIR = r"e:/imaichika_whu/models_cache"
# 模型的 HuggingFace ID (必须与缓存文件夹结构匹配)
# MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    print(f"🔄 正在从本地加载模型: {MODEL_ID} ...")
    print(f"📂 缓存路径: {CACHE_DIR}")

    try:
        # 1. 加载分词器 (Tokenizer)
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            local_files_only=True,  # 关键：强制只读取本地文件，不联网
            trust_remote_code=True
        )

        # 2. 加载模型 (Model)
        # device_map="auto" 会自动检测是否有显卡，有则用 GPU，无则用 CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR,
            local_files_only=True,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        print("请检查：\n1. 路径是否正确\n2. 缓存文件是否完整\n3. 是否安装了 transformers 和 torch")
        return

    print("\n✅ 模型加载成功！开始对话 (输入 'exit' 或 '退出' 结束)")
    print("-" * 50)

    # 3. 初始化对话历史
    history = [
        {"role": "system", "content": "你是一只可爱的猫娘，我是你的主人，你需要用可爱和带“喵”的语言回复我。"}
    ]

    while True:
        # 获取用户输入
        try:
            query = input("\n👤 你: ").strip()
        except KeyboardInterrupt:
            break
            
        if not query:
            continue
        if query.lower() in ["exit", "quit", "退出"]:
            print("👋 再见！")
            break

        # 添加用户消息到历史
        history.append({"role": "user", "content": query})

        # 4. 构建模型输入
        text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # 5. 生成回复
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,  # 最大回复长度
                temperature=0.7,      # 随机性 (0-1)，越低越保守
                top_p=0.9
            )

        # 6. 解码输出
        # 去掉输入部分的 token，只保留新生成的
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"🤖 Qwen: {response}")

        # 将回复加入历史，支持多轮对话
        history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()