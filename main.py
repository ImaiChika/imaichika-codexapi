import os
from datetime import datetime
from pathlib import Path
import sys

from tqdm import tqdm

from src.analysis.layer1_regex import RegexAnalyzer
from src.analysis.layer2_nlp import InteractionNetwork, TextMiner
from src.analysis.layer3_reasoning import ReasoningLayer
from src.config import (
    DATA_DB_DIR,
    DATA_PROC_DIR,
    DATA_RAW_DIR,
    HIGH_SIGNAL_KEYWORDS,
    LLM_MIN_L1_SCORE,
    LLM_MIN_L2_SCORE,
)
from src.linkage.identity_resolver import IdentityResolver
from src.loader import load_json_data
from src.profiling.group_profile import GroupProfiler
from src.profiling.user_profile import UserProfiler
from src.storage.multi_db import MultiDBManager
from src.utils import save_json, setup_logger

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.append(str(Path(__file__).parent))


def should_use_llm(message: dict) -> bool:
    """
    Cost-aware gate: only call LLM when message is risky or semantically ambiguous.
    """
    if message.get("is_system_msg"):
        return False

    text = str(message.get("text", "") or "")
    if not text:
        return False

    if bool(message.get("has_pii")):
        return True

    if float(message.get("l1_risk_score", 0) or 0) >= LLM_MIN_L1_SCORE:
        return True

    if float(message.get("l2_risk_score", 0) or 0) >= LLM_MIN_L2_SCORE:
        return True

    if any(k in text for k in HIGH_SIGNAL_KEYWORDS):
        return True

    # short/noisy messages can skip LLM
    if len(text.strip()) <= 6:
        return False

    return False


def iter_raw_files(raw_dir: Path):
    return sorted([p for p in raw_dir.glob("*.json") if p.is_file()])


def main():
    logger = setup_logger()
    logger.info("=== 启动多群聊联动研判系统 ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    l1 = RegexAnalyzer()
    l2_nlp = TextMiner()
    l2_net = InteractionNetwork()
    l3 = ReasoningLayer()
    user_p = UserProfiler()
    group_p = GroupProfiler()
    identity_resolver = IdentityResolver()
    db = MultiDBManager(DATA_DB_DIR)

    raw_files = iter_raw_files(DATA_RAW_DIR)
    if not raw_files:
        logger.warning(f"未在 {DATA_RAW_DIR} 发现任何 JSON 原始群聊文件")
        db.close()
        return

    logger.info(f"检测到 {len(raw_files)} 个原始群聊文件: {[p.name for p in raw_files]}")

    processed_msgs = []
    llm_calls = 0

    for raw_path in raw_files:
        raw_data = load_json_data(raw_path)
        logger.info(f"处理文件: {raw_path.name}, 消息数: {len(raw_data)}")

        for idx, msg in enumerate(tqdm(raw_data, desc=f"Step 1: {raw_path.stem}"), start=1):
            msg = dict(msg)
            msg["source_group"] = raw_path.stem
            msg["source_file"] = raw_path.name
            msg["msg_index"] = idx

            msg = l1.process_single_message(msg)
            msg.update(l2_nlp.process(msg))

            if should_use_llm(msg):
                msg["llm_decision"] = l3.analyze(msg)
                llm_calls += 1
            else:
                msg["llm_decision"] = l3.quick_analyze(msg)

            group_p.update(msg)
            processed_msgs.append(msg)
            db.store_message(msg)

    logger.info("Step 2: 计算全量社交拓扑权重...")
    l2_net.build_from_data(processed_msgs)
    network_stats = l2_net.analyze_centrality()

    for msg in processed_msgs:
        uname = msg.get("username")
        if uname in network_stats:
            msg["user_profile"] = network_stats[uname]

    user_p.aggregate(processed_msgs)
    final_users = user_p.finalize()
    group_summary = group_p.get_summary_context(network_stats)

    logger.info("Step 3: 进行跨群身份关联与事件溯源...")
    identity_result = identity_resolver.resolve(processed_msgs)
    identity_resolver.attach_cluster_labels(processed_msgs, identity_result["node_to_cluster"])
    trace_events = identity_resolver.build_trace_events(processed_msgs, identity_result["node_to_cluster"])
    linkage_summary = identity_resolver.summarize(identity_result["clusters"], trace_events)

    db.store_identity_clusters(identity_result["clusters"])
    db.store_trace_events(trace_events)

    logger.info("Step 4: 生成研判报告...")
    report_content = l3.generate_comprehensive_report(group_summary, final_users)

    processed_path = save_json(processed_msgs, Path(f"multi_groups_full_{timestamp}"), DATA_PROC_DIR)
    users_path = save_json(network_stats, Path(f"multi_groups_users_{timestamp}"), DATA_PROC_DIR)
    clusters_path = save_json(identity_result["clusters"], Path(f"identity_clusters_{timestamp}"), DATA_PROC_DIR)
    traces_path = save_json(trace_events, Path(f"cross_group_traces_{timestamp}"), DATA_PROC_DIR)
    linkage_path = save_json(linkage_summary, Path(f"linkage_summary_{timestamp}"), DATA_PROC_DIR)

    report_path = DATA_PROC_DIR / f"final_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        f.write("\n\n=== 自动化取证摘要 ===\n")
        f.write(f"最终锁定嫌疑人: {', '.join(group_summary.get('suspect_list', [])) or '无'}\n")
        f.write(f"最终锁定受害者/准受害者: {', '.join(group_summary.get('victim_list', [])) or '无'}\n")
        f.write(f"无关人士/水军: {', '.join(group_summary.get('irrelevant_list', [])) or '无'}\n")
        f.write(f"跨群身份簇数量: {linkage_summary.get('cluster_count', 0)}\n")
        f.write(f"跨群关联簇(>=2群): {linkage_summary.get('cross_group_cluster_count', 0)}\n")
        f.write(f"LLM调用数/总消息数: {llm_calls}/{len(processed_msgs)}\n")

    db.close()

    logger.info("分析完成")
    logger.info(f"处理消息输出: {processed_path}")
    logger.info(f"用户画像输出: {users_path}")
    logger.info(f"身份簇输出: {clusters_path}")
    logger.info(f"跨群事件输出: {traces_path}")
    logger.info(f"关联摘要输出: {linkage_path}")
    logger.info(f"最终报告输出: {report_path}")


if __name__ == "__main__":
    main()
