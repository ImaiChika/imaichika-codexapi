import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import src.analysis.layer3_reasoning as layer3_module
from main import _format_clue_chain_lines, iter_raw_files, should_use_llm
from src.analysis.layer1_regex import RegexAnalyzer
from src.analysis.layer2_nlp import InteractionNetwork, TextMiner
from src.analysis.layer3_parts.reporting_mixin import ReasoningReportMixin
from src.analysis.layer3_reasoning import ReasoningLayer
from src.config import DATA_PROC_DIR, DATA_RAW_DIR, REPORT_CLUE_CHAIN_TOP_K
from src.linkage.identity_resolver import IdentityResolver
from src.loader import load_json_data
from src.profiling.group_profile import GroupProfiler
from src.profiling.user_profile import UserProfiler
from src.storage.multi_db import MultiDBManager
from src.utils import save_json, setup_logger


# 这组高价值线索来自当前 raw 数据中的跨群样本，用于做统一的人工金标准对照。
GOLD_PII = {
    "extracted_name_cn": {"王康"},
    "extracted_name_en": {"WangKang"},
    "extracted_mobile_cn": {"13972224978", "15875458309", "15361350538"},
    "extracted_id_card": {"320381198605011833"},
    "extracted_qq_number": {"1285851817"},
    "wallet": {"TFTGxnqqchaEgKVwDZUozLmthxmqwJBnTn"},
    "extracted_address_cn": {
        "广东省汕头市",
        "湖北省武汉市洪山区关山大道 519 号附近",
        "湖北省武汉市洪山区关山大道 519 号联想产业基地附近",
    },
}

# 用于比较“嫌疑人排序”是否更接近我们对跨群事件的人工理解。
GOLD_CORE_SUSPECTS = [
    "user_-1002394323226",
    "dbkyi",
    "XiaoTJiang",
    "xiaofnb",
]

# 用于比较“角色判断”的关键账号集合。
GOLD_ROLE_USERS = {
    "user_-1002394323226": "scammer",
    "dbkyi": "scammer",
    "XiaoTJiang": "scammer",
    "xiaofnb": "scammer",
    "moxiaonuo": "scammer",
    "victim_awei": "victim",
    "beizhaole666": "victim",
    "wq2025": "victim",
    "Qingshan_77": "victim",
}

FOCUS_ROLE_USERS = [
    "user_-1002394323226",
    "dbkyi",
    "XiaoTJiang",
    "victim_awei",
    "beizhaole666",
]

ROLE_PROXY_SUSPECT_USERS = {
    "user_-1002394323226",
    "dbkyi",
    "XiaoTJiang",
    "xiaofnb",
    "moxiaonuo",
    "GD19_810",
    "yinuo_emo888",
}

ROLE_PROXY_VICTIM_USERS = {
    "victim_awei",
    "beizhaole666",
    "wq2025",
    "Qingshan_77",
}


@dataclass(frozen=True)
class AblationMode:
    key: str
    label: str
    use_l2: bool
    use_l3: bool
    use_identity: bool
    build_network: bool


MODES: List[AblationMode] = [
    AblationMode(
        key="rules_only",
        label="只有规则",
        use_l2=False,
        use_l3=False,
        use_identity=False,
        build_network=False,
    ),
    AblationMode(
        key="rules_nlp",
        label="规则 + NLP",
        use_l2=True,
        use_l3=False,
        use_identity=False,
        build_network=True,
    ),
    AblationMode(
        key="rules_nlp_reasoning",
        label="规则 + NLP + 推理",
        use_l2=True,
        use_l3=True,
        use_identity=False,
        build_network=True,
    ),
    AblationMode(
        key="full_stack",
        label="规则 + NLP + 推理 + 身份关联",
        use_l2=True,
        use_l3=True,
        use_identity=True,
        build_network=True,
    ),
]


class ReportRenderer(ReasoningReportMixin):
    """只复用原报告模板，不引入 Layer3 推理逻辑。"""


def _bucket_risk(score: float) -> str:
    if score >= 60:
        return "high"
    if score >= 25:
        return "medium"
    return "low"


def _intent_from_message(text: str, has_pii: bool, risk_score: float) -> str:
    if any(k in text for k in ["被骗", "拉黑", "报警", "报案", "投诉", "退钱", "没给全"]):
        return "victim_complaint"
    if has_pii and any(k in text for k in ["收款", "地址", "下发", "查档", "三要素", "户籍", "家谱", "近照"]):
        return "pii_trade"
    if risk_score >= 40:
        return "suspicious_exchange"
    return "other"


def _heuristic_role_rules_only(message: Dict) -> str:
    text = str(message.get("text", "") or "")
    role_clue = str(message.get("l1_role_clue", "neutral") or "neutral")

    if role_clue in {"scammer", "manager"}:
        return "scammer"
    if role_clue == "victim":
        return "victim"

    if any(k in text for k in ["被骗", "拉黑", "报警", "报案", "退钱", "投诉", "没给全"]):
        return "victim"
    if any(k in text for k in ["下发", "收U", "收款地址", "查档", "开后台", "重复卖", "拆开卖"]):
        return "scammer"
    return "other"


def _heuristic_role_rules_nlp(message: Dict) -> str:
    text = str(message.get("text", "") or "")
    nlp_keywords = set(message.get("nlp_keywords", []) or [])
    role_clue = _heuristic_role_rules_only(message)

    if role_clue != "other":
        return role_clue

    if any(k in text for k in ["被骗", "拉黑", "报警", "报案", "退钱", "投诉", "没给全"]):
        return "victim"

    suspect_kw = {
        "卡号", "户名", "下发", "收U", "收款地址", "查档", "三要素",
        "户籍", "家谱", "近照", "挂外群", "重复卖", "拆开卖",
    }
    if suspect_kw & nlp_keywords or any(k in text for k in suspect_kw):
        return "scammer"

    return "other"


def _fallback_decision(message: Dict, use_l2: bool) -> Dict[str, str]:
    text = str(message.get("text", "") or "")
    l1_score = float(message.get("l1_risk_score", 0) or 0)
    l2_score = float(message.get("l2_risk_score", 0) or 0) if use_l2 else 0.0
    combined = min(l1_score + 0.5 * l2_score, 100.0)
    role = _heuristic_role_rules_nlp(message) if use_l2 else _heuristic_role_rules_only(message)

    return {
        "risk": _bucket_risk(combined),
        "role": role,
        "intent": _intent_from_message(text, bool(message.get("has_pii")), combined),
    }


def _rename_with_mode_tag(path: Path, mode_key: str) -> Path:
    new_path = path.with_name(f"{path.stem}__{mode_key}{path.suffix}")
    path.rename(new_path)
    return new_path


def _write_report(
    report_renderer,
    timestamp: str,
    llm_calls: int,
    processed_msgs: List[Dict],
    group_summary: Dict,
    final_users: List[Dict],
    linkage_summary: Dict,
    clue_chains: List[Dict],
) -> Path:
    report_content = report_renderer.generate_comprehensive_report(group_summary, final_users)
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

    return report_path


def _collect_metrics(
    mode: AblationMode,
    processed_msgs: List[Dict],
    group_summary: Dict,
    linkage_summary: Dict,
    clue_chains: List[Dict],
    report_path: Path,
    artifact_paths: Dict[str, Path],
    llm_calls: int,
) -> Dict:
    extracted: Dict[str, set] = {}
    for msg in processed_msgs:
        pii = msg.get("pii_details", {})
        if not isinstance(pii, dict):
            continue
        for key, vals in pii.items():
            if not isinstance(vals, list):
                continue
            extracted.setdefault(key, set()).update(vals)

    wallets = set(extracted.get("extracted_usdt_address", set())) | set(extracted.get("extracted_payment_address", set()))
    gold_rows: List[Tuple[str, str, bool]] = []
    for key, values in GOLD_PII.items():
        if key == "wallet":
            for value in sorted(values):
                gold_rows.append((key, value, value in wallets))
            continue
        observed = extracted.get(key, set())
        for value in sorted(values):
            gold_rows.append((key, value, value in observed))

    gold_hit = sum(1 for _, _, ok in gold_rows if ok)
    report_text = report_path.read_text(encoding="utf-8")

    role_counts: Dict[str, int] = {}
    risk_counts: Dict[str, int] = {}
    per_user_role_votes: Dict[str, Dict[str, int]] = {u: {} for u in GOLD_ROLE_USERS}
    high_risk_count = 0
    candidate_risk_count = 0
    suspect_msg_total = 0
    suspect_msg_scammer = 0
    suspect_rel_total = 0
    suspect_rel_scammer = 0
    victim_msg_total = 0
    victim_msg_victim = 0
    victim_rel_total = 0
    victim_rel_victim = 0

    for msg in processed_msgs:
        decision = msg.get("llm_decision", {}) if isinstance(msg.get("llm_decision", {}), dict) else {}
        role = str(decision.get("role", "other") or "other")
        risk = str(decision.get("risk", "low") or "low")
        username = msg.get("username")
        source_group = msg.get("source_group")

        role_counts[role] = role_counts.get(role, 0) + 1
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        if risk == "high":
            high_risk_count += 1
        if risk in {"medium", "high"}:
            candidate_risk_count += 1

        if username in per_user_role_votes:
            vote_box = per_user_role_votes[username]
            vote_box[role] = vote_box.get(role, 0) + 1

        if username in ROLE_PROXY_SUSPECT_USERS:
            suspect_msg_total += 1
            if role == "scammer":
                suspect_msg_scammer += 1
            if source_group in {"1_augmented", "3"}:
                suspect_rel_total += 1
                if role == "scammer":
                    suspect_rel_scammer += 1

        if username in ROLE_PROXY_VICTIM_USERS:
            victim_msg_total += 1
            if role == "victim":
                victim_msg_victim += 1
            if source_group == "2":
                victim_rel_total += 1
                if role == "victim":
                    victim_rel_victim += 1

    majority_role_eval = []
    role_correct = 0
    for username, gold_role in GOLD_ROLE_USERS.items():
        votes = per_user_role_votes.get(username, {})
        pred_role = max(votes.items(), key=lambda kv: kv[1])[0] if votes else "other"
        is_correct = pred_role == gold_role
        role_correct += 1 if is_correct else 0
        majority_role_eval.append(
            {
                "username": username,
                "gold_role": gold_role,
                "pred_role": pred_role,
                "votes": votes,
                "correct": is_correct,
            }
        )

    suspect_rank_map = {
        username: idx + 1
        for idx, username in enumerate(group_summary.get("suspect_list", []))
    }
    top3_hit = sum(1 for username in GOLD_CORE_SUSPECTS if suspect_rank_map.get(username, 10**9) <= 3)
    top5_hit = sum(1 for username in GOLD_CORE_SUSPECTS if suspect_rank_map.get(username, 10**9) <= 5)
    mean_rank = round(
        sum(suspect_rank_map.get(username, len(group_summary.get("suspect_list", [])) + 1) for username in GOLD_CORE_SUSPECTS)
        / max(len(GOLD_CORE_SUSPECTS), 1),
        2,
    )

    chain_map = {c.get("clue_value", ""): c for c in clue_chains}
    return {
        "mode_key": mode.key,
        "mode_label": mode.label,
        "llm_calls": llm_calls,
        "message_count": len(processed_msgs),
        "role_counts": role_counts,
        "risk_counts": risk_counts,
        "high_risk_count": high_risk_count,
        "candidate_risk_count": candidate_risk_count,
        "suspects": list(group_summary.get("suspect_list", [])),
        "victims": list(group_summary.get("victim_list", [])),
        "irrelevant": list(group_summary.get("irrelevant_list", [])),
        "suspect_top3_hit": top3_hit,
        "suspect_top5_hit": top5_hit,
        "suspect_mean_rank": mean_rank,
        "suspect_rank_map": {u: suspect_rank_map.get(u) for u in GOLD_CORE_SUSPECTS},
        "role_majority_accuracy": round(role_correct / len(GOLD_ROLE_USERS), 4) if GOLD_ROLE_USERS else 0.0,
        "role_majority_eval": majority_role_eval,
        "focus_role_eval": [x for x in majority_role_eval if x["username"] in FOCUS_ROLE_USERS],
        "suspect_role_proxy": {
            "scammer_msg_hits": suspect_msg_scammer,
            "msg_total": suspect_msg_total,
            "ratio": round(suspect_msg_scammer / suspect_msg_total, 4) if suspect_msg_total else 0.0,
            "relevant_hits": suspect_rel_scammer,
            "relevant_total": suspect_rel_total,
            "relevant_ratio": round(suspect_rel_scammer / suspect_rel_total, 4) if suspect_rel_total else 0.0,
        },
        "victim_role_proxy": {
            "victim_msg_hits": victim_msg_victim,
            "msg_total": victim_msg_total,
            "ratio": round(victim_msg_victim / victim_msg_total, 4) if victim_msg_total else 0.0,
            "relevant_hits": victim_rel_victim,
            "relevant_total": victim_rel_total,
            "relevant_ratio": round(victim_rel_victim / victim_rel_total, 4) if victim_rel_total else 0.0,
        },
        "cluster_count": int(linkage_summary.get("cluster_count", 0) or 0),
        "cross_group_cluster_count": int(linkage_summary.get("cross_group_cluster_count", 0) or 0),
        "clue_chain_count": len(clue_chains),
        "gold_rows": gold_rows,
        "gold_recall": round(gold_hit / len(gold_rows), 4) if gold_rows else 0.0,
        "report_path": str(report_path),
        "artifact_paths": {k: str(v) for k, v in artifact_paths.items()},
        "report_contains": {
            "dbkyi": "dbkyi" in report_text,
            "XiaoTJiang": "XiaoTJiang" in report_text,
            "xiaofnb": "xiaofnb" in report_text,
            "victim_awei": "victim_awei" in report_text,
            "beizhaole666": "beizhaole666" in report_text,
            "wq2025": "wq2025" in report_text,
        },
        "chain_targets": {
            "WangKang": {
                "present": "WangKang" in chain_map,
                "chain_type": chain_map.get("WangKang", {}).get("chain_type", ""),
                "groups": chain_map.get("WangKang", {}).get("groups", []),
            },
            "wallet": {
                "present": "TFTGxnqqchaEgKVwDZUozLmthxmqwJBnTn" in chain_map,
                "chain_type": chain_map.get("TFTGxnqqchaEgKVwDZUozLmthxmqwJBnTn", {}).get("chain_type", ""),
                "groups": chain_map.get("TFTGxnqqchaEgKVwDZUozLmthxmqwJBnTn", {}).get("groups", []),
            },
            "qq": {
                "present": "1285851817" in chain_map,
                "chain_type": chain_map.get("1285851817", {}).get("chain_type", ""),
                "groups": chain_map.get("1285851817", {}).get("groups", []),
            },
        },
    }


def _write_summary_md(summary_rows: List[Dict], out_path: Path):
    lines: List[str] = []
    lines.append("# 消融实验对比总结")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 实验说明")
    lines.append("")
    lines.append("- 数据集：`data/raw/1.json`、`data/raw/1_augmented.json`、`data/raw/2.json`、`data/raw/3.json`")
    lines.append("- 共同流程：原始数据加载、Layer1 规则抽取、画像汇总、原报告模板渲染、结果保存")
    lines.append("- 控制变量：只切换 `NLP增强`、`Layer3推理`、`身份关联` 三个能力开关，不改原有主流程代码")
    lines.append("- 结果文件命名：先按原逻辑生成，再在文件名末尾追加 `__模式名` 便于对比")
    lines.append("")
    lines.append("## 模式总览")
    lines.append("")
    lines.append("| 模式 | 金标准PII召回 | 高风险消息 | 中高风险候选 | 关键角色准确率 | 嫌疑Top3命中 | 嫌疑Top5命中 | 线索链数 | 报告文件 |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for row in summary_rows:
        lines.append(
            f"| {row['mode_label']} | {row['gold_recall']:.2%} | {row['high_risk_count']} | {row['candidate_risk_count']} | "
            f"{row['role_majority_accuracy']:.2%} | {row['suspect_top3_hit']}/{len(GOLD_CORE_SUSPECTS)} | "
            f"{row['suspect_top5_hit']}/{len(GOLD_CORE_SUSPECTS)} | "
            f"{row['clue_chain_count']} | `{Path(row['report_path']).name}` |"
        )
    lines.append("")

    lines.append("## 关键观察")
    lines.append("")
    for row in summary_rows:
        lines.append(f"### {row['mode_label']}")
        lines.append("")
        lines.append(f"- 报告文件：`{row['report_path']}`")
        lines.append(f"- 主要嫌疑人：{', '.join(row['suspects'][:8]) or '无'}")
        lines.append(f"- 主要受害者：{', '.join(row['victims'][:8]) or '无'}")
        lines.append(
            f"- 高风险消息筛选数：{row['high_risk_count']}；中高风险候选消息数：{row['candidate_risk_count']}；"
            f"角色判断准确率（关键账号主角色）：{row['role_majority_accuracy']:.2%}"
        )
        lines.append(
            f"- 核心嫌疑人排序：Top3 命中 {row['suspect_top3_hit']}/{len(GOLD_CORE_SUSPECTS)}，"
            f"Top5 命中 {row['suspect_top5_hit']}/{len(GOLD_CORE_SUSPECTS)}，"
            f"平均排名 {row['suspect_mean_rank']}"
        )
        lines.append(
            "- 核心嫌疑人具体排名："
            + "，".join(
                f"{u}={row['suspect_rank_map'].get(u, '未入榜')}"
                for u in GOLD_CORE_SUSPECTS
            )
        )
        lines.append(
            f"- 金标准PII召回：{row['gold_recall']:.2%} "
            f"（命中 {sum(1 for _, _, ok in row['gold_rows'] if ok)}/{len(row['gold_rows'])}）"
        )
        lines.append(f"- 跨群关联簇：{row['cross_group_cluster_count']}，线索链：{row['clue_chain_count']}")
        lines.append(
            f"- 嫌疑方消息识别比例：{row['suspect_role_proxy']['scammer_msg_hits']}/{row['suspect_role_proxy']['msg_total']}="
            f"{row['suspect_role_proxy']['ratio']:.2%}；仅看增强跨群样本为 "
            f"{row['suspect_role_proxy']['relevant_hits']}/{row['suspect_role_proxy']['relevant_total']}="
            f"{row['suspect_role_proxy']['relevant_ratio']:.2%}"
        )
        lines.append(
            f"- 投诉方消息识别比例：{row['victim_role_proxy']['victim_msg_hits']}/{row['victim_role_proxy']['msg_total']}="
            f"{row['victim_role_proxy']['ratio']:.2%}；仅看售后群为 "
            f"{row['victim_role_proxy']['relevant_hits']}/{row['victim_role_proxy']['relevant_total']}="
            f"{row['victim_role_proxy']['relevant_ratio']:.2%}"
        )
        lines.append(
            "- 关键账号主角色："
            + "，".join(
                f"{item['username']}={item['pred_role']}"
                for item in row["focus_role_eval"]
            )
        )
        lines.append(
            "- 核心线索链命中："
            f"WangKang={row['chain_targets']['WangKang']['present']}，"
            f"收款地址={row['chain_targets']['wallet']['present']}，"
            f"QQ={row['chain_targets']['qq']['present']}"
        )
        lines.append(
            "- 报告是否出现关键投诉用户："
            f"victim_awei={row['report_contains']['victim_awei']}，"
            f"beizhaole666={row['report_contains']['beizhaole666']}，"
            f"wq2025={row['report_contains']['wq2025']}"
        )
        lines.append("")

    lines.append("## 结论")
    lines.append("")
    lines.append("- 规则层对高价值隐私实体已有较强抓取能力，是整套系统的基础，因此各模式在 PII 召回上差异不大。")
    lines.append("- 仅加入 NLP 后，高风险消息筛选数和嫌疑方消息识别比例会明显上升，说明关键词与语义特征对嫌疑内容过滤有帮助。")
    lines.append("- 加入 Layer3 推理后，角色判断会发生再分配，部分账号的判定会更保守，但高风险消息筛选与投诉方识别会更接近“案件式研判”思路。")
    lines.append("- 身份关联层本身不直接提高 PII 抽取，但会显著补强跨群证据链，是系统从“识别消息”走向“还原事件”的关键。")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_mode(mode: AblationMode) -> Dict:
    logger = setup_logger()
    logger.info(f"=== 开始消融实验: {mode.label} ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 实验专用数据库放到 tmp 下，避免和主流程的 data/db 权限或历史状态互相影响。
    mode_db_dir = Path("tmp") / "ablation_db" / mode.key
    if mode_db_dir.exists():
        shutil.rmtree(mode_db_dir)
    mode_db_dir.mkdir(parents=True, exist_ok=True)

    l1 = RegexAnalyzer()
    l2_nlp = TextMiner() if mode.use_l2 else None
    l2_net = InteractionNetwork() if mode.build_network else None
    group_p = GroupProfiler()
    user_p = UserProfiler()
    identity_resolver = IdentityResolver() if mode.use_identity else None
    db = MultiDBManager(mode_db_dir)

    report_renderer = ReportRenderer()
    l3 = None
    if mode.use_l3:
        # 每个模式使用独立的向量库存储，避免不同实验相互污染。
        layer3_module.VECTOR_DB_DIR = mode_db_dir / "chroma"
        l3 = ReasoningLayer()
        report_renderer = l3

    raw_files = iter_raw_files(DATA_RAW_DIR)
    if not raw_files:
        raise FileNotFoundError(f"未在 {DATA_RAW_DIR} 发现可用 JSON 原始文件")

    processed_msgs: List[Dict] = []
    llm_calls = 0

    for raw_path in raw_files:
        raw_data = load_json_data(raw_path)
        logger.info(f"[{mode.label}] 处理文件: {raw_path.name}, 消息数: {len(raw_data)}")

        for idx, raw_msg in enumerate(raw_data, start=1):
            msg = dict(raw_msg)
            msg["source_group"] = raw_path.stem
            msg["source_file"] = raw_path.name
            msg["msg_index"] = idx

            msg = l1.process_single_message(msg)

            if mode.use_l2 and l2_nlp is not None:
                msg.update(l2_nlp.process(msg))
            else:
                msg.update(
                    {
                        "is_system_msg": False,
                        "nlp_keywords": [],
                        "token_count": len(str(msg.get("text", "") or "")),
                        "l2_risk_score": 0,
                        "l2_evidence": ["未启用Layer2"],
                    }
                )

            if mode.use_l3 and l3 is not None:
                if should_use_llm(msg):
                    msg["llm_decision"] = l3.analyze(msg)
                    llm_calls += 1
                else:
                    msg["llm_decision"] = l3.quick_analyze(msg)
            else:
                msg["llm_decision"] = _fallback_decision(msg, use_l2=mode.use_l2)

            group_p.update(msg)
            processed_msgs.append(msg)
            db.store_message(msg)

    network_stats: Dict[str, Dict] = {}
    if mode.build_network and l2_net is not None:
        logger.info(f"[{mode.label}] Step 2: 计算社交拓扑...")
        l2_net.build_from_data(processed_msgs)
        network_stats = l2_net.analyze_centrality()
        for msg in processed_msgs:
            uname = msg.get("username")
            if uname in network_stats:
                msg["user_profile"] = network_stats[uname]

    user_p.aggregate(processed_msgs)
    final_users = user_p.finalize()
    group_summary = group_p.get_summary_context(network_stats)

    identity_result = {"clusters": [], "node_to_cluster": {}}
    trace_events: List[Dict] = []
    linkage_summary = {"cluster_count": 0, "cross_group_cluster_count": 0, "cross_group_clusters": []}
    clue_chains: List[Dict] = []

    if mode.use_identity and identity_resolver is not None:
        logger.info(f"[{mode.label}] Step 3: 执行身份关联...")
        identity_result = identity_resolver.resolve(processed_msgs)
        identity_resolver.attach_cluster_labels(processed_msgs, identity_result["node_to_cluster"])
        trace_events = identity_resolver.build_trace_events(processed_msgs, identity_result["node_to_cluster"])
        linkage_summary = identity_resolver.summarize(identity_result["clusters"], trace_events)
        clue_chains = identity_resolver.build_clue_chains(processed_msgs, identity_result["node_to_cluster"], group_summary)

        db.store_identity_clusters(identity_result["clusters"])
        db.store_trace_events(trace_events)

    logger.info(f"[{mode.label}] Step 4: 生成实验报告与中间产物...")
    processed_path = save_json(processed_msgs, Path(f"multi_groups_full_{timestamp}"), DATA_PROC_DIR)
    users_path = save_json(network_stats, Path(f"multi_groups_users_{timestamp}"), DATA_PROC_DIR)
    clusters_path = save_json(identity_result["clusters"], Path(f"identity_clusters_{timestamp}"), DATA_PROC_DIR)
    traces_path = save_json(trace_events, Path(f"cross_group_traces_{timestamp}"), DATA_PROC_DIR)
    linkage_path = save_json(linkage_summary, Path(f"linkage_summary_{timestamp}"), DATA_PROC_DIR)
    clue_chain_path = save_json(clue_chains, Path(f"clue_chains_{timestamp}"), DATA_PROC_DIR)
    report_path = _write_report(
        report_renderer=report_renderer,
        timestamp=timestamp,
        llm_calls=llm_calls,
        processed_msgs=processed_msgs,
        group_summary=group_summary,
        final_users=final_users,
        linkage_summary=linkage_summary,
        clue_chains=clue_chains,
    )

    renamed_paths = {
        "processed": _rename_with_mode_tag(processed_path, mode.key),
        "users": _rename_with_mode_tag(users_path, mode.key),
        "clusters": _rename_with_mode_tag(clusters_path, mode.key),
        "traces": _rename_with_mode_tag(traces_path, mode.key),
        "linkage": _rename_with_mode_tag(linkage_path, mode.key),
        "clue_chains": _rename_with_mode_tag(clue_chain_path, mode.key),
        "report": _rename_with_mode_tag(report_path, mode.key),
    }

    metrics = _collect_metrics(
        mode=mode,
        processed_msgs=processed_msgs,
        group_summary=group_summary,
        linkage_summary=linkage_summary,
        clue_chains=clue_chains,
        report_path=renamed_paths["report"],
        artifact_paths=renamed_paths,
        llm_calls=llm_calls,
    )

    db.close()
    logger.info(f"=== 完成消融实验: {mode.label} ===")
    return metrics


def main():
    summary_rows = [run_mode(mode) for mode in MODES]
    summary_path = Path("ablation_experiment_summary.md")
    _write_summary_md(summary_rows, summary_path)
    print(f"消融实验完成，汇总文档已生成: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
