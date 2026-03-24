from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PROC_DIR = ROOT_DIR / "data" / "processed"


class MainSmokeTest(unittest.TestCase):
    def test_main_generates_expected_report_contract(self):
        before_reports = set(DATA_PROC_DIR.glob("final_report_*.txt"))

        completed = subprocess.run(
            [sys.executable, str(ROOT_DIR / "main.py")],
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
        )

        if completed.returncode != 0:
            self.fail(
                "main.py 运行失败。\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )

        after_reports = sorted(DATA_PROC_DIR.glob("final_report_*.txt"), key=lambda p: p.stat().st_mtime)
        self.assertTrue(after_reports, "未生成任何 final_report_*.txt")

        new_reports = [p for p in after_reports if p not in before_reports]
        report_path = new_reports[-1] if new_reports else after_reports[-1]
        report_text = report_path.read_text(encoding="utf-8")

        for fragment in [
            "**非法网络活动研判报告**",
            "### 1. 案件定性",
            "### 2. 核心成员画像",
            "### 3. 隐私泄露与作案账号明细表",
            "### 4. 打击建议",
            "=== 自动化取证摘要 ===",
            "=== 同人线索链（软关联，不等于强身份并人） ===",
        ]:
            self.assertIn(fragment, report_text)

        self.assertRegex(report_text, r"风险等级：[高中]")
        self.assertRegex(report_text, r"LLM调用数/总消息数:\s*\d+/\d+")
        self.assertRegex(report_text, r"线索链数量\(软关联\):\s*\d+")


if __name__ == "__main__":
    unittest.main()
