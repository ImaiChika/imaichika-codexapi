# main.py
from datetime import datetime
from pathlib import Path
import sys

from tqdm import tqdm

from src.analysis.layer1_regex import RegexAnalyzer
from src.analysis.layer2_nlp import InteractionNetwork, TextMiner
from src.analysis.layer3_reasoning import ReasoningLayer
from src.config import DATA_PROC_DIR, DATA_RAW_DIR
from src.loader import load_json_data
from src.profiling.group_profile import GroupProfiler
from src.profiling.user_profile import UserProfiler
from src.utils import save_json, setup_logger

sys.path.append(str(Path(__file__).parent))



def main():
    logger = setup_logger()
    logger.info("=== 启动全链路 API 研判系统 (Qwen-api版) ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_file_base = "test100" # 替换为你的文件名

    # 1. 初始化
    l1 = RegexAnalyzer()
    l2_nlp = TextMiner()
    l2_net = InteractionNetwork()
    l3 = ReasoningLayer()
    user_p = UserProfiler()
    group_p = GroupProfiler()

    # 2. 数据加载
    raw_data = load_json_data(DATA_RAW_DIR / f"{raw_file_base}.json")

    # 3. 循环分析
    processed_msgs = []
    for msg in tqdm(raw_data, desc="Step 1: AI Reasoning"):
        msg = l1.process_single_message(msg)
        msg.update(l2_nlp.process(msg))
        # 调用 Qwen API 进行分析
        msg["llm_decision"] = l3.analyze(msg)
        group_p.update(msg)
        processed_msgs.append(msg)

    # 4. 社交网络分析
    logger.info("Step 2: 计算社交拓扑权重...")
    l2_net.build_from_data(processed_msgs)
    network_stats = l2_net.analyze_centrality()
    # 同步 PageRank 到消息中，防止 UserProfiler 读不到数据
    for msg in processed_msgs:
        uname = msg.get("username")
        if uname in network_stats:
            msg["user_profile"] = network_stats[uname]
    # 5. 聚合报告 (关键：传入网络指标过滤 Group_Leader_A)
    # 在 group_profile.py 中实现 get_summary_context(network_stats)
    group_summary = group_p.get_summary_context(network_stats)
    user_p.aggregate(processed_msgs)
    final_users = user_p.finalize()

    # 6. 生成最终 API 研判报告
    logger.info("Step 3: 撰写深度侦察报告...")
    report_content = l3.generate_comprehensive_report(group_summary, final_users)

    # 7. 保存结果
    save_json(processed_msgs, Path(f"{raw_file_base}_full_{timestamp}"), DATA_PROC_DIR)
    save_json(network_stats, Path(f"{raw_file_base}_users_{timestamp}"), DATA_PROC_DIR)
    
    report_path = DATA_PROC_DIR / f"final_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        f.write("\n\n=== 自动化取证摘要 ===\n")
        f.write(f"最终锁定的受害者名单: {', '.join(group_summary['victim_list'])}\n")

    logger.info(f"分析完成！报告已存至: {report_path}")
    
if __name__ == "__main__":
    main()