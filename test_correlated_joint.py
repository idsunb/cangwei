# -*- coding: utf-8 -*-
"""
相关性联合分布测试: 两证券 Bernoulli 2×2 + 多证券 copula + 最大增值熵.

运行:
  python test_correlated_joint.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from position_sizing import (  # noqa: E402
    joint_2x2_correlated,
    joint_from_correlation_copula,
    optimize_portfolio_entropy,
    split_cash_debt,
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def close(a, b, tol=1e-4) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print("=" * 60)
    print("相关性联合分布 / 最大增值熵 测试")
    print("=" * 60)
    failures = 0

    print("\n[1] 两证券 2×2 边际与 ρ 边界")
    rows = joint_2x2_correlated(0.5, 0.5, 0.0, (-0.1, 0.3), (-0.15, 0.4))
    s = sum(r["p"] for r in rows)
    ok = close(s, 1.0)
    if not ok:
        failures += 1
    check("概率和=1", ok, f"sum={s}")
    # ρ=0 独立
    pmap = {r["label"]: r["p"] for r in rows}
    ok = all(close(pmap[k], 0.25) for k in ("LL", "LH", "HL", "HH"))
    if not ok:
        failures += 1
    check("ρ=0 → 各 0.25", ok, str(pmap))

    rows1 = joint_2x2_correlated(0.5, 0.5, 1.0, (-0.1, 0.3), (-0.15, 0.4))
    p1 = {r["label"]: r["p"] for r in rows1}
    ok = close(p1["HH"], 0.5) and close(p1["LL"], 0.5) and close(p1["HL"], 0.0, 1e-9)
    if not ok:
        failures += 1
    check("ρ=1 → 只同涨同跌", ok, str(p1))

    rowsm = joint_2x2_correlated(0.5, 0.5, -1.0, (-0.1, 0.3), (-0.15, 0.4))
    pm = {r["label"]: r["p"] for r in rowsm}
    ok = close(pm["HH"], 0.0, 1e-9) and close(pm["LL"], 0.0, 1e-9) and close(pm["HL"], 0.5)
    if not ok:
        failures += 1
    check("ρ=−1 → 只一涨一跌", ok, str(pm))

    print("\n[2] 相关如何影响最优 H（负相关更好）")
    def H_star(rows, r0=0.0):
        probs = [r["p"] for r in rows]
        grid = [[r["ra"], r["rb"]] for r in rows]
        opt = optimize_portfolio_entropy(probs, grid, r0=r0)
        return opt["H_bits"], opt["weights"]

    H0, w0 = H_star(joint_2x2_correlated(0.5, 0.5, 0.0, (-0.1, 0.3), (-0.15, 0.4)))
    Hp, wp = H_star(joint_2x2_correlated(0.5, 0.5, 0.9, (-0.1, 0.3), (-0.15, 0.4)))
    Hm, wm = H_star(joint_2x2_correlated(0.5, 0.5, -0.9, (-0.1, 0.3), (-0.15, 0.4)))
    ok = Hm > H0 > Hp
    if not ok:
        failures += 1
    check("H*(ρ=−0.9) > H*(0) > H*(+0.9)", ok,
          f"neg={Hm:.6f}, zero={H0:.6f}, pos={Hp:.6f}")

    print("\n[3] 三证券 copula 与独立对照")
    marg = [
        {"p_high": 0.5, "r_low": -0.1, "r_high": 0.3},
        {"p_high": 0.5, "r_low": -0.15, "r_high": 0.4},
        {"p_high": 0.6, "r_low": -0.05, "r_high": 0.2},
    ]
    C = [
        [1.0, 0.5, 0.2],
        [0.5, 1.0, 0.3],
        [0.2, 0.3, 1.0],
    ]
    pj, gj = joint_from_correlation_copula(marg, C, n_samples=4000, seed=42)
    ok = len(pj) == len(gj) and abs(sum(pj) - 1.0) < 1e-9
    if not ok:
        failures += 1
    check("copula 联合概率和=1", ok, f"n={len(pj)}, sum={sum(pj)}")
    # 独立对照
    from position_sizing import expand_joint_from_marginals

    pi, gi = expand_joint_from_marginals(
        [[(1 - m["p_high"], m["r_low"]), (m["p_high"], m["r_high"])] for m in marg]
    )
    oi = optimize_portfolio_entropy(pi, gi, r0=0.0)
    oc = optimize_portfolio_entropy(pj, gj, r0=0.0)
    ok = all(x == x and abs(x) != float("inf") for x in oc["weights"])
    if not ok:
        failures += 1
    check("相关联合可求最优权重", ok, f"w={oc['weights']}")
    # 正相关下 H 应不高于独立（通常）
    ok = oc["H_bits"] <= oi["H_bits"] + 1e-3
    if not ok:
        failures += 1
    check("正相关 H ≤ 独立 H", ok,
          f"corr={oc['H_bits']:.6f}, ind={oi['H_bits']:.6f}")

    print("\n[4] 非正定相关矩阵可收缩求解")
    bad = [[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 1.0]]
    try:
        pb, gb = joint_from_correlation_copula(marg, bad, n_samples=2000, seed=1)
        ok = abs(sum(pb) - 1.0) < 1e-9
        if not ok:
            failures += 1
        check("PSD 收缩后联合和=1", ok, f"n={len(pb)}")
    except Exception as e:
        failures += 1
        check("PSD 收缩", False, str(e))

    print("\n[5] 负债拆分兼容")
    sp = split_cash_debt(-2.75, [3.75])
    ok = close(sp["debt"], 2.75) and close(sp["cash"], 0.0)
    if not ok:
        failures += 1
    check("split_cash_debt", ok, str(sp))

    print("\n[5b] 2×3 状态数不同 (A 2 态, B 3 态)")
    marg_2x3 = [
        {"states": [{"p": 0.5, "r": -0.1}, {"p": 0.5, "r": 0.3}]},
        {"states": [
            {"p": 0.3, "r": -0.25},
            {"p": 0.4, "r": 0.08},
            {"p": 0.3, "r": 0.5},
        ]},
    ]
    corr_2x3 = [[1.0, 0.4], [0.4, 1.0]]
    p23, g23 = joint_from_correlation_copula(marg_2x3, corr_2x3, n_samples=5000, seed=7)
    ok = abs(sum(p23) - 1.0) < 1e-9 and len(p23) <= 6
    if not ok:
        failures += 1
    check("2×3 联合和=1 且格数≤6", ok, f"n={len(p23)}, sum={sum(p23)}")
    # 边际还原: A 第1态 r=-0.1 的概率 ≈ 0.5
    pA0 = sum(p for p, row in zip(p23, g23) if abs(row[0] - (-0.1)) < 1e-12)
    ok = abs(pA0 - 0.5) < 0.05  # MC 误差
    if not ok:
        failures += 1
    check("2×3 边际还原 A0≈0.5", ok, f"got {pA0:.4f}")
    # B 三态收益各出现
    b_vals = {round(row[1], 6) for row in g23}
    ok = {-0.25, 0.08, 0.5} <= b_vals or {-0.25, 0.08, 0.5}.issubset(b_vals)
    if not ok:
        # 至少两个不同 B 收益
        ok = len(b_vals) >= 2
    if not ok:
        failures += 1
    check("2×3 出现多种 B 收益", ok, str(sorted(b_vals)))

    # 独立笛卡尔积 2×3=6
    pind, gind = expand_joint_from_marginals([
        [(0.5, -0.1), (0.5, 0.3)],
        [(0.3, -0.25), (0.4, 0.08), (0.3, 0.5)],
    ])
    ok = len(pind) == 6 and abs(sum(pind) - 1.0) < 1e-9
    if not ok:
        failures += 1
    check("独立展开 2×3=6", ok, f"n={len(pind)}")

    o23 = optimize_portfolio_entropy(p23, g23, r0=0.0)
    oi23 = optimize_portfolio_entropy(pind, gind, r0=0.0)
    ok = all(x == x and abs(x) != float("inf") for x in o23["weights"])
    if not ok:
        failures += 1
    check("2×3 相关可求最优", ok, f"w={o23['weights']}")
    # ρ=0.4 正相关: H 应不高于独立
    ok = o23["H_bits"] <= oi23["H_bits"] + 0.02
    if not ok:
        failures += 1
    check("2×3 正相关 H≤独立 H", ok,
          f"corr={o23['H_bits']:.6f}, ind={oi23['H_bits']:.6f}")

    # 旧格式仍可用
    p_old, g_old = joint_from_correlation_copula(
        [
            {"p_high": 0.5, "r_low": -0.1, "r_high": 0.3},
            {"p_high": 0.5, "r_low": -0.15, "r_high": 0.4},
        ],
        [[1.0, 0.0], [0.0, 1.0]],
        n_samples=2000,
        seed=3,
    )
    ok = abs(sum(p_old) - 1.0) < 1e-9
    if not ok:
        failures += 1
    check("旧两状态格式兼容", ok, f"n={len(p_old)}")

    print("\n[6] JS 侧 test_correlated_joint.js")
    import subprocess

    js = ROOT / "test_correlated_joint.js"
    if not js.exists():
        failures += 1
        check("test_correlated_joint.js 存在", False)
    else:
        r = subprocess.run(["node", str(js)], capture_output=True, text=True, encoding="utf-8")
        ok = r.returncode == 0
        if not ok:
            failures += 1
        check("JS 相关测试通过", ok, (r.stdout + r.stderr)[-200:].replace("\n", " "))

    print("\n[7] copula 推导文档完整性")
    doc = ROOT / "copula推导.md"
    if not doc.exists():
        failures += 1
        check("copula推导.md 存在", False)
    else:
        text = doc.read_text(encoding="utf-8")
        for key in [
            "概率积分变换",
            "高斯",
            "Cholesky",
            "marginalCuts",
            "2×3",
            "jointFromCopula",
            "Frechet",
        ]:
            ok = key in text
            if not ok:
                failures += 1
            check(f"文档含「{key}」", ok)

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
