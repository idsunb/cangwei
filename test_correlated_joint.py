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

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
