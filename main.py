import os
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List

from tqdm import tqdm

from src.analysis.layer1_regex import RegexAnalyzer
from src.analysis.layer2_nlp import InteractionNetwork, TextMiner
from src.analysis.layer3_reasoning import ReasoningLayer
from src.config import (
    DATA_DB_DIR,
    DATA_PROC_DIR,
    DATA_RAW_DIR,
    HIGH_SIGNAL_KEYWORDS,
    IMPLICIT_SIGNAL_KEYWORDS,
    LLM_MIN_L1_SCORE,
    LLM_MIN_L2_SCORE,
    REPORT_CLUE_CHAIN_TOP_K,
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
    只有当消息被视为高风险或语义模糊时，才调用LLM，以节省token。
    """
    if message.get("is_system_msg"):
        return False

    text = str(message.get("text", "") or "")
    if not text:
        return False

    if bool(message.get("has_pii")):
        return True

    pii_details = message.get("pii_details", {})
    if isinstance(pii_details, dict) and any(
        pii_details.get(k)
        for k in [
            "extracted_mobile_masked",
            "extracted_id_masked",
            "extracted_mobile_fragment",
            "extracted_id_fragment",
            "extracted_alias_clue",
            "extracted_address_hint",
        ]
    ):
        return True

    if float(message.get("l1_risk_score", 0) or 0) >= LLM_MIN_L1_SCORE:
        return True

    if float(message.get("l2_risk_score", 0) or 0) >= LLM_MIN_L2_SCORE:
        return True

    if any(k in text for k in HIGH_SIGNAL_KEYWORDS):
        return True

    if any(k in text for k in IMPLICIT_SIGNAL_KEYWORDS):
        return True

    # short/noisy messages can skip LLM
    if len(text.strip()) <= 6:
        return False

    return False


def iter_raw_files(raw_dir: Path):
    """按文件名稳定排序，保证多次运行时处理顺序一致。"""
    return sorted([p for p in raw_dir.glob("*.json") if p.is_file()])


def _format_clue_chain_lines(chains: List[Dict], top_k: int = REPORT_CLUE_CHAIN_TOP_K) -> List[str]:
    """把跨群软关联线索链渲染成报告尾部摘要。"""
    if not chains:
        return ["- 暂无可用线索链"]

    lines: List[str] = []
    for c in chains[:top_k]:
        victims = ", ".join(c.get("points_to", {}).get("victims", [])[:4]) or "无"
        suspects = ", ".join(c.get("points_to", {}).get("suspects", [])[:4]) or "无"

        pair_items = []
        for p in c.get("soft_identity_candidates", [])[:3]:
            pair_items.append(f"{p.get('account_a')}<->{p.get('account_b')}")
        pair_txt = "; ".join(pair_items) if pair_items else "无"

        lines.append(
            f"- [{c.get('chain_id', '')}][{c.get('confidence', 'low')}/{c.get('chain_type', 'correlation')}] "
            f"{c.get('clue_type', 'pii')}={c.get('clue_value', '')} | "
            f"受害者指向: {victims} | 嫌疑人关联: {suspects} | 同人候选: {pair_txt}"
        )

    return lines


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

    # Step 1: 对每条原始消息执行三层分析，并实时累积画像与数据库。
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

    # Step 2: 基于全量消息回看整个社交拓扑，再补用户影响力画像。
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

    # Step 3: 做跨群身份归并、事件追踪和软线索链生成。
    logger.info("Step 3: 进行跨群身份关联与事件溯源...")
    identity_result = identity_resolver.resolve(processed_msgs)
    identity_resolver.attach_cluster_labels(processed_msgs, identity_result["node_to_cluster"])
    trace_events = identity_resolver.build_trace_events(processed_msgs, identity_result["node_to_cluster"])
    linkage_summary = identity_resolver.summarize(identity_result["clusters"], trace_events)
    clue_chains = identity_resolver.build_clue_chains(processed_msgs, identity_result["node_to_cluster"], group_summary)

    db.store_identity_clusters(identity_result["clusters"])
    db.store_trace_events(trace_events)

    # Step 4: 生成最终研判报告与各类中间产物，便于答辩展示和复核。
    logger.info("Step 4: 生成研判报告...")
    report_content = l3.generate_comprehensive_report(group_summary, final_users)

    processed_path = save_json(processed_msgs, Path(f"multi_groups_full_{timestamp}"), DATA_PROC_DIR)
    users_path = save_json(network_stats, Path(f"multi_groups_users_{timestamp}"), DATA_PROC_DIR)
    clusters_path = save_json(identity_result["clusters"], Path(f"identity_clusters_{timestamp}"), DATA_PROC_DIR)
    traces_path = save_json(trace_events, Path(f"cross_group_traces_{timestamp}"), DATA_PROC_DIR)
    linkage_path = save_json(linkage_summary, Path(f"linkage_summary_{timestamp}"), DATA_PROC_DIR)
    clue_chain_path = save_json(clue_chains, Path(f"clue_chains_{timestamp}"), DATA_PROC_DIR)

    report_path = DATA_PROC_DIR / f"final_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        f.write("\n\n=== 自动化取证摘要 ===\n")
        f.write(f"最终锁定嫌疑人: {', '.join(group_summary.get('suspect_list', [])) or '无'}\n")
        f.write(f"最终锁定受害者/准受害者: {', '.join(group_summary.get('victim_list', [])) or '无'}\n")
        f.write(f"无关人士/水军: {', '.join(group_summary.get('irrelevant_list', [])) or '无'}\n")
        f.write(f"跨群身份簇数量: {linkage_summary.get('cluster_count', 0)}\n")
        f.write(f"跨群关联簇(>=2群): {linkage_summary.get('cross_group_cluster_count', 0)}\n")
        f.write(f"线索链数量(软关联): {len(clue_chains)}\n")
        f.write(f"LLM调用数/总消息数: {llm_calls}/{len(processed_msgs)}\n")

        f.write("\n=== 同人线索链（软关联，不等于强身份并人） ===\n")
        for line in _format_clue_chain_lines(clue_chains, REPORT_CLUE_CHAIN_TOP_K):
            f.write(f"{line}\n")

    db.close()

    logger.info("分析完成")
    logger.info(f"处理消息输出: {processed_path}")
    logger.info(f"用户画像输出: {users_path}")
    logger.info(f"身份簇输出: {clusters_path}")
    logger.info(f"跨群事件输出: {traces_path}")
    logger.info(f"关联摘要输出: {linkage_path}")
    logger.info(f"线索链输出: {clue_chain_path}")
    logger.info(f"最终报告输出: {report_path}")


if __name__ == "__main__":
    main()
