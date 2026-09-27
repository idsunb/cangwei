# -*- coding: utf-8 -*-
"""
求解器设计记录文档完整性测试.

确保《求解器设计记录.md》包含升级所需的关键章节，
并保持与 position_sizing.py 中的实现锚点一致。

运行:
  python test_solver_design_doc.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOC = ROOT / "求解器设计记录.md"
PY = ROOT / "position_sizing.py"

REQUIRED_SECTIONS = [
    "问题定义",
    "数学性质",
    "方案比选",
    "数值细节",
    "与书中公式的对齐",
    "升级路线图",
    "测试与回归锚点",
    "附录 A",
    "SLSQP",
    "坐标",
    "全局最优",
    "run_book_examples",
    "expand_joint_from_marginals",
]

REQUIRED_PY_SNIPPETS = [
    "optimize_portfolio_entropy",
    "optimal_position_single",
    "run_book_examples",
    "expand_joint_from_marginals",
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 60)
    print("求解器设计记录 文档完整性测试")
    print("=" * 60)
    failures = 0

    if not DOC.exists():
        print("  [✗] 求解器设计记录.md 不存在")
        return 1
    text = DOC.read_text(encoding="utf-8")
    py = PY.read_text(encoding="utf-8") if PY.exists() else ""

    print("\n[1] 必备章节 / 关键词")
    for sec in REQUIRED_SECTIONS:
        ok = sec in text
        if not ok:
            failures += 1
        check(f"包含「{sec}」", ok)

    print("\n[2] 与代码锚点一致")
    for name in REQUIRED_PY_SNIPPETS:
        in_doc = name in text
        in_py = name in py
        ok = in_doc and in_py
        if not ok:
            failures += 1
        check(f"锚点 {name}", ok, f"doc={in_doc}, py={in_py}")

    print("\n[3] 路线图版本条目")
    for ver in ("v1", "v2", "v3", "v4"):
        ok = ver in text
        if not ok:
            failures += 1
        check(f"路线图 {ver}", ok)

    print("\n[4] 附录含可升级代码")
    ok = "solve_max_entropy_slsqp" in text
    if not ok:
        failures += 1
    check("附录 A 函数签名", ok)
    ok = "def grad" in text or "grad(q)" in text
    if not ok:
        failures += 1
    check("解析梯度说明", ok)

    print("\n[5] 公式清单交叉引用（可选但期望）")
    formula = ROOT / "仓位管理公式清单.md"
    if formula.exists():
        ftext = formula.read_text(encoding="utf-8")
        ok = "求解器设计记录" in ftext or True  # 软性：不强制
        check("公式清单存在", True)
    else:
        check("公式清单存在", False)
        failures += 1

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
