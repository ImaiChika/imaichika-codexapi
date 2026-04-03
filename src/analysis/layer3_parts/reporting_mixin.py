from __future__ import annotations

from typing import Dict, List

from src.config import REPORT_CORE_USER_TOP_K, REPORT_PROFILE_LINE_TOP_K


class ReasoningReportMixin:
    """封装 Layer3 的最终报告渲染逻辑。"""

    def generate_comprehensive_report(self, group_stats, top_kols):
        """保持原始报告模板、字段顺序与措辞风格不变。"""
        suspect_assets = group_stats.get("suspect_assets", []) or []
        victim_leaks = group_stats.get("victim_leaks", []) or []
        victim_names = group_stats.get("victim_list", []) or []
        suspect_names = group_stats.get("suspect_list", []) or []
        irrelevant_names = set(group_stats.get("irrelevant_list", []) or [])

        core_users = []
        # 先优先放嫌疑人，再补受害者和高影响力成员，和原实现一致。
        for name in suspect_names:
            if name not in core_users:
                core_users.append(name)
            if len(core_users) >= REPORT_CORE_USER_TOP_K:
                break

        if len(core_users) < REPORT_CORE_USER_TOP_K:
            for name in victim_names:
                if name not in core_users:
                    core_users.append(name)
                if len(core_users) >= REPORT_CORE_USER_TOP_K:
                    break

        if len(core_users) < REPORT_CORE_USER_TOP_K:
            for u in top_kols:
                name = u.get("username", "unknown")
                if name not in core_users:
                    core_users.append(name)
                if len(core_users) >= REPORT_CORE_USER_TOP_K:
                    break

        if not victim_names:
            victim_names = ["暂无明确受害用户"]

        def format_evidence(items: List[Dict]) -> str:
            if not items:
                return "- 暂无"
            lines = []
            for e in items:
                lines.append(
                    f"- **{e.get('type', '其他')}**：{e.get('content', '')}"
                    f"（发布者：{e.get('owner', 'unknown')}，语境：{e.get('context', '')}）"
                )
            return "\n".join(lines)

        risk_level = "高" if (len(suspect_assets) >= 2 and len(victim_names) >= 1) else "中"

        role_map = {u.get("username", "unknown"): str(u.get("predicted_role", "other")) for u in top_kols}
        core_profiles = []
        for name in core_users[:REPORT_PROFILE_LINE_TOP_K]:
            role = role_map.get(name, "other")
            if name in irrelevant_names:
                desc = "无关人士/水军，主要为签到或闲聊噪声。"
            elif name in victim_names:
                desc = "疑似受害者，其信息可能被用于非法资金活动。"
            elif name in suspect_names or role == "scammer":
                desc = "疑似嫌疑人，具备资金或指令操控特征。"
            else:
                desc = "群内活跃成员，需结合更多证据研判。"
            core_profiles.append(f"- **{name}**：{desc}")

        intro_users = core_users[: min(6, REPORT_CORE_USER_TOP_K)]

        report = f"""**非法网络活动研判报告**

---

### 1. 案件定性
本案件属于**非法网络活动**，涉及**隐私信息泄露**与**资金相关操作**。核心人物包括**{', '.join(intro_users) if intro_users else '暂无'}**，其中疑似受害者为**{', '.join(victim_names)}**。

---

### 2. 核心成员画像
{chr(10).join(core_profiles) if core_profiles else '- 暂无'}

---

### 3. 隐私泄露与作案账号明细表

{format_evidence(victim_leaks)}

{format_evidence(suspect_assets)}

---

### 4. 打击建议
- **立即冻结涉案账户**，追查资金流向，锁定资金链。
- **对核心嫌疑人实施全链条打击**，包括指令发布、资金分发与执行节点。
- **对受害者开展保护与取证协助**，避免二次伤害。
- **持续监控群内关键词与账户复用行为**，防止同类活动扩散。

---
**风险等级：{risk_level}**"""

        return report
