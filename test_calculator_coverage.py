# -*- coding: utf-8 -*-
"""
网页计算器 ↔ 公式清单 一致性测试.

检查项:
  1. 公式清单中的每个可计算章节在 HTML 中都有对应选项卡 / 面板
  2. HTML 内嵌 JS 的核心公式结果与 position_sizing.py 一致
  3. 公式清单「实现文件」与实际文件一致

运行:
  python test_calculator_coverage.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "仓位管理计算器.html"
FORMULA = ROOT / "仓位管理公式清单.md"
PY = ROOT / "position_sizing.py"

# 公式清单中「必须能在网页上算」的章节 → 期望的 data-tab / panel id
REQUIRED_TABS = {
    "§3.1": "multi",      # 最大增值熵
    "§3.2": "single",     # 单硬币最优仓位
    "§3.3": "lev",        # 透支卖空
    "§3.4": "fee",        # 手续费增量
    "§5.4.1": "fut",      # 等概率 / 期货
    "§3.5": "div",        # 多硬币分散
    "§3.6": "div",        # 极限定理（同分散页）
    "§3.7": "cons",       # 消费增值熵
    "§3.8": "approx",     # 抛物线近似（加速计算）
}

# 公式清单中允许「并入其它页 / 仅概念」的章节引用
ALLOWED_SIDEWAYS = {"§2.1", "§2.3", "§3.11", "§6.2", "§2", "§5"}

# 书中例题: (名称, probs, returns, r0, 期望 q*, 容差)
JS_VS_PY_CASES = [
    ("coin", [0.5, 0.5], [-1.0, 2.0], 0.0, 0.25),
    ("stock", [0.5, 0.5], [-0.3, 0.8], 0.1, 0.59),
    ("opt", [0.7, 0.3], [-1.0, 3.0], 0.0, 0.0667),
    ("dice", [1 / 3, 2 / 3], [-1.0, 1.0], 0.0, 1 / 3),
]

# 不等概率边际分布: 资产 A 4 种、B 5 种 (网页「分资产边际分布」模式)
MARGINAL_4_5 = [
    [(0.1, -0.2), (0.3, 0.05), (0.4, 0.15), (0.2, 0.4)],       # A: 4 states
    [(0.1, -0.5), (0.2, -0.1), (0.3, 0.08), (0.25, 0.2), (0.15, 0.6)],  # B: 5 states
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("=" * 60)
    print("网页计算器 ↔ 公式清单 一致性测试")
    print("=" * 60)
    failures = 0

    html = HTML.read_text(encoding="utf-8")
    formula = FORMULA.read_text(encoding="utf-8")

    # --- 1. 章节覆盖 ---
    print("\n[1] 章节 ↔ 选项卡覆盖")
    tabs = set(re.findall(r'data-tab="([a-z]+)"', html))
    panels = set(re.findall(r'id="panel-([a-z]+)"', html))
    for sec, tab_id in REQUIRED_TABS.items():
        ok_tab = tab_id in tabs
        ok_panel = tab_id in panels
        ok = ok_tab and ok_panel
        if not ok:
            failures += 1
        check(f"{sec} → data-tab={tab_id}", ok, f"tabs={ok_tab}, panel={ok_panel}")

    # 公式清单中点名的章节都应在 REQUIRED 出现
    listed_secs = re.findall(r"§\d+(?:\.\d+)*", formula)
    for sec in sorted(set(listed_secs)):
        if sec in ALLOWED_SIDEWAYS:
            continue  # 概念/例题章节，归入其它页
        if sec not in REQUIRED_TABS:
            failures += 1
            check(f"公式清单 {sec} 未映射到任何选项卡", False)
        else:
            check(f"公式清单 {sec} 已映射", True)

    # about 表不得再把 §3.7/§3.8 写成仅 Python
    bad = re.search(r"§3\.[78]</td>.*?Python</td>", html)
    if bad:
        failures += 1
        check("about 表仍写 §3.7/§3.8=Python", False)
    else:
        check("about 表 §3.7/§3.8 指向网页选项卡", True)

    # --- 2. HTML 含关键公式文本 ---
    print("\n[2] 关键公式已写入网页")
    for label, needle in [
        ("§3.8 抛物线 a/b/c", "2[H(t)"),
        ("§3.8 q*=−b/2a", "b/(2a)"),
        ("§3.7 消费熵", "min(q,K)"),
        ("§5.4.1 等概率式", "(1/r₂ + 1/r₁)"),
        ("§5.4.1 极限 1/2", "1/2"),
    ]:
        ok = needle in html
        if not ok:
            failures += 1
        check(label, ok)

    # --- 3. JS 语法 + 与 Python 数值对拍 ---
    print("\n[3] JS / Python 数值对拍")
    # 先 Python 侧
    sys.path.insert(0, str(ROOT))
    from position_sizing import optimal_position_single  # noqa: E402

    py_results = []
    for name, probs, returns, r0, expect in JS_VS_PY_CASES:
        res = optimal_position_single(probs, returns, r0=r0)
        ok = abs(res.q_star - expect) <= 0.01
        if not ok:
            failures += 1
        check(f"Python {name} q*≈{expect}", ok, f"got {res.q_star:.4f}")
        py_results.append((name, res.q_star))

    # JS 侧: 提取脚本用 node 跑
    m = re.search(r"<script>([\s\S]*)</script>", html)
    if not m:
        failures += 1
        check("HTML script 标签", False)
    else:
        js_body = m.group(1)
        # 清掉 DOM 相关初始化，只留纯函数
        js_math = js_body.split("/* ========== 单资产")[0]
        # 再补上 optimalSingle / portReturns 定义（它们在单资产区）
        # 重新抽取需要的函数源码
        def grab(fn_name: str) -> str:
            pat = rf"function {fn_name}\([\s\S]*?\n\}}"
            mm = re.search(pat, js_body)
            return mm.group(0) if mm else ""

        pure = "\n".join(
            [
                "const LOG2 = Math.log(2);",
                "function log2(x) { return Math.log(x) / LOG2; }",
                "function fmt(x, d = 4) { if (!isFinite(x)) return String(x); return x.toFixed(d); }",
                "function pct(x, d = 2) { if (!isFinite(x)) return String(x); return (x * 100).toFixed(d) + '%'; }",
                grab("growthEntropy"),
                grab("geoMean"),
                grab("expReturn"),
                grab("portReturns"),
                grab("gridSearchSingle"),
                grab("optimalSingle"),
                grab("consumerEntropy"),
                grab("parabolicApprox"),
                grab("H_of_q"),
            ]
        )
        # 驱动脚本
        driver = pure + "\n" + f"""
const cases = {JS_VS_PY_CASES!r};
for (const [name, probs, returns, r0, expect] of cases) {{
  const r = optimalSingle(probs, returns, r0, false, false, 2);
  console.log(name + " " + r.q.toFixed(6));
}}
// §3.8 书中例
const ap = parabolicApprox([0.25,0.5,0.25], [-1,0.5,2], 0, 0.5, 0, 1);
console.log("approx_q " + ap.q.toFixed(6));
console.log("exact_q " + ap.qExact.toFixed(6));
// §3.7 书中基金A
const hA = consumerEntropy([1],[0.2],1,0,0.4,0.15);
console.log("HcA " + hA.toFixed(6));
"""
        # Python list repr 中 True/False/元组 — 用 JSON 更稳
        import json  # noqa: E402

        driver = pure + "\n" + "const cases = " + json.dumps(
            [{"name": n, "probs": list(map(float, p)), "returns": list(map(float, r)), "r0": r0, "expect": e}
             for n, p, r, r0, e in JS_VS_PY_CASES]
        ) + """
for (const c of cases) {
  const r = optimalSingle(c.probs, c.returns, c.r0, false, false, 2);
  console.log(c.name + " " + r.q.toFixed(6));
}
const ap = parabolicApprox([0.25,0.5,0.25], [-1,0.5,2], 0, 0.5, 0, 1);
console.log("approx_q " + ap.q.toFixed(6));
console.log("exact_q " + ap.qExact.toFixed(6));
const hA = consumerEntropy([1],[0.2],1,0,0.4,0.15);
console.log("HcA " + hA.toFixed(6));
"""
        node = subprocess.run(
            ["node", "-e", driver],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if node.returncode != 0:
            failures += 1
            check("JS 数值脚本执行", False, node.stderr[:200])
        else:
            lines = [ln.strip() for ln in node.stdout.splitlines() if ln.strip()]
            parsed = dict(ln.split() for ln in lines if " " in ln)
            for name, py_q in py_results:
                if name not in parsed:
                    failures += 1
                    check(f"JS {name} 有输出", False)
                    continue
                js_q = float(parsed[name])
                ok = abs(js_q - py_q) <= 1e-4
                if not ok:
                    failures += 1
                check(f"JS≡Python {name}", ok, f"JS={js_q:.6f}, PY={py_q:.6f}")

            # §3.8 与 Python 抛物线
            from position_sizing import optimal_position_parabolic_approx  # noqa: E402

            py_ap = optimal_position_parabolic_approx(
                [0.25, 0.5, 0.25], [-1.0, 0.5, 2.0], r0=0.0, t=0.5
            )
            if "approx_q" in parsed:
                ok = abs(float(parsed["approx_q"]) - py_ap["q_approx"]) <= 1e-3
                if not ok:
                    failures += 1
                check(
                    "JS≡Python §3.8 近似q",
                    ok,
                    f"JS={parsed['approx_q']}, PY={py_ap['q_approx']:.6f}",
                )
            if "exact_q" in parsed:
                ok = abs(float(parsed["exact_q"]) - py_ap["H_at_exact_search"][1]) <= 5e-3
                if not ok:
                    failures += 1
                check(
                    "JS≡Python §3.8 精确q",
                    ok,
                    f"JS={parsed['exact_q']}, PY={py_ap['H_at_exact_search'][1]:.6f}",
                )
            # §3.7
            from position_sizing import consumer_growth_entropy  # noqa: E402

            py_h = consumer_growth_entropy(
                [1.0], [0.2], q=1.0, r0=0.0, consume_ratio=0.4, r_sacrifice=0.15
            )
            if "HcA" in parsed:
                ok = abs(float(parsed["HcA"]) - py_h) <= 1e-4
                if not ok:
                    failures += 1
                check("JS≡Python §3.7 HcA", ok, f"JS={parsed['HcA']}, PY={py_h:.6f}")

            # 书中例题回归（网页算出来的结果）
            for name, _, _, _, expect in JS_VS_PY_CASES:
                if name in parsed:
                    ok = abs(float(parsed[name]) - expect) <= 0.01
                    if not ok:
                        failures += 1
                    check(f"JS {name}≈书中 {expect}", ok, f"got {parsed[name]}")

    # --- 3b. 边际分布展开 (4×5) + 最优仓位 ---
    print("\n[3b] 分资产边际分布 → 联合情景 (A:4, B:5)")
    from position_sizing import (
        expand_joint_from_marginals,
        optimize_portfolio_entropy,
    )

    probs_j, grid_j = expand_joint_from_marginals(MARGINAL_4_5)
    ok = len(probs_j) == 20 and len(grid_j) == 20
    if not ok:
        failures += 1
    check("展开为 4×5=20 个联合情景", ok, f"got {len(probs_j)}")
    ok = abs(sum(probs_j) - 1.0) < 1e-9
    if not ok:
        failures += 1
    check("联合概率和=1", ok, f"sum={sum(probs_j)}")
    # 独立性抽查: P(A=1st) ≈ 0.1
    p_a0 = sum(probs_j[i] for i, row in enumerate(grid_j) if abs(row[0] - MARGINAL_4_5[0][0][1]) < 1e-12)
    ok = abs(p_a0 - 0.1) < 1e-9
    if not ok:
        failures += 1
    check("边际还原 A 第1档 p=0.1", ok, f"got {p_a0}")

    opt = optimize_portfolio_entropy(probs_j, grid_j, r0=0.0)
    w = opt["weights"]
    ok = len(w) == 3  # cash + A + B
    if not ok:
        failures += 1
    check("4×5 情景可求出 3 个权重 (现金+A+B)", ok, f"weights={w}")
    ok = all(x == x and abs(x) != float("inf") for x in w)
    if not ok:
        failures += 1
    check("权重有限可读", ok, f"weights={w}")
    # 权重和 = 1
    ok = abs(sum(w) - 1.0) < 1e-6
    if not ok:
        failures += 1
    check("Σq=1", ok, f"sum={sum(w)}")

    # JS 侧 expandJointFromMarginals 与 Python 对拍
    m2 = re.search(r"<script>([\s\S]*)</script>", html)
    js_body2 = m2.group(1) if m2 else ""

    def grab2(fn_name: str) -> str:
        pat = rf"function {fn_name}\([\s\S]*?\n\}}"
        mm = re.search(pat, js_body2)
        return mm.group(0) if mm else ""

    js_expand = (
        grab2("expandJointFromMarginals")
        + """
const marg = {
  assets: ["A", "B"],
  dists: [
    [{p:0.1,r:-0.2},{p:0.3,r:0.05},{p:0.4,r:0.15},{p:0.2,r:0.4}],
    [{p:0.1,r:-0.5},{p:0.2,r:-0.1},{p:0.3,r:0.08},{p:0.25,r:0.2},{p:0.15,r:0.6}]
  ]
};
const j = expandJointFromMarginals(marg);
console.log("joint_n " + j.probs.length);
console.log("joint_sum " + j.probs.reduce((a,b)=>a+b,0).toFixed(10));
console.log("p0 " + j.probs[0].toFixed(10));
console.log("r00 " + j.grid[0][0].toFixed(10));
"""
    )
    node2 = subprocess.run(
        ["node", "-e", js_expand],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if node2.returncode != 0:
        failures += 1
        check("JS expandJointFromMarginals", False, node2.stderr[:200])
    else:
        parsed = dict(
            ln.split() for ln in node2.stdout.splitlines() if " " in ln
        )
        ok = int(parsed.get("joint_n", 0)) == 20
        if not ok:
            failures += 1
        check("JS 展开 20 情景", ok, parsed.get("joint_n", "?"))
        ok = abs(float(parsed.get("joint_sum", 0)) - 1.0) < 1e-9
        if not ok:
            failures += 1
        check("JS 联合概率和=1", ok, parsed.get("joint_sum", "?"))
        ok = abs(float(parsed.get("p0", 0)) - probs_j[0]) < 1e-12
        if not ok:
            failures += 1
        check("JS≡Python 联合 p[0]", ok, f"JS={parsed.get('p0')}, PY={probs_j[0]}")
        ok = abs(float(parsed.get("r00", 0)) - grid_j[0][0]) < 1e-12
        if not ok:
            failures += 1
        check("JS≡Python r[0,0]", ok, f"JS={parsed.get('r00')}, PY={grid_j[0][0]}")

    # --- 4. 公式清单「实现文件」与实际一致 ---
    print("\n[4] 公式清单 ↔ 实现文件")
    for fname in ("position_sizing.py", "仓位管理计算器.html", "公式清单.md"):
        ok = fname in formula
        if not ok:
            failures += 1
        check(f"清单提到 {fname}", ok)
    check("position_sizing.py 存在", PY.exists())
    check("仓位管理计算器.html 存在", HTML.exists())

    print("\n" + "=" * 60)
    if failures:
        print(f"失败 {failures} 项")
    else:
        print("全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
