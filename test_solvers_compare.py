# -*- coding: utf-8 -*-
"""
多资产求解器对比测试: 爬山 / SLSQP / §3.8 近似初值+SLSQP.

运行:
  python test_solvers_compare.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from position_sizing import (  # noqa: E402
    approx_warmstart_weights,
    compare_solvers,
    expand_joint_from_marginals,
    optimize_portfolio_entropy,
    optimal_position_single,
    solve_max_entropy_slsqp,
    solve_max_entropy_warmstart,
)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def close(a, b, tol=1e-4) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    print("=" * 60)
    print("多资产求解器对比测试 (爬山 / SLSQP / 近似初值)")
    print("=" * 60)
    failures = 0

    # 书中两硬币独立 4 情景
    probs_coin = [0.25] * 4
    grid_coin = [[-1, -1], [-1, 2], [2, -1], [2, 2]]

    print("\n[1] SLSQP 与书中单硬币闭式一致")
    # 单资产单硬币 → q*=0.25
    one = [[0.5], [0.5]]
    # 两情景两状态: 亏 / 赢
    one_grid = [[-1.0], [2.0]]
    s1 = solve_max_entropy_slsqp([0.5, 0.5], one_grid, r0=0.0)
    closed = optimal_position_single([0.5, 0.5], [-1.0, 2.0], r0=0.0)
    ok = close(s1["asset_weights"][0], closed.q_star, 1e-5)
    if not ok:
        failures += 1
    check(
        "SLSQP q≈闭式 0.25",
        ok,
        f"slsqp={s1['asset_weights'][0]:.6f}, closed={closed.q_star:.6f}",
    )
    ok = s1["success"] or close(s1["asset_weights"][0], 0.25, 1e-4)
    if not ok:
        failures += 1
    check("SLSQP success", ok, s1.get("message", ""))

    print("\n[2] SLSQP vs 爬山 (两硬币 4 情景)")
    hill = optimize_portfolio_entropy(probs_coin, grid_coin, r0=0.0)
    sls = solve_max_entropy_slsqp(probs_coin, grid_coin, r0=0.0)
    ok = close(sls["H_bits"], hill["H_bits"], 5e-3)
    if not ok:
        failures += 1
    check(
        "H(SLSQP)≈H(爬山)",
        ok,
        f"slsqp={sls['H_bits']:.6f}, hill={hill['H_bits']:.6f}",
    )
    # 权重各约 0.23
    w = sls["asset_weights"]
    ok = all(close(x, 0.23, 0.05) for x in w)
    if not ok:
        failures += 1
    check("SLSQP 各资产≈0.23", ok, f"weights={w}")
    # SLSQP 的 H 不应显著差于爬山
    ok = sls["H_bits"] >= hill["H_bits"] - 1e-3
    if not ok:
        failures += 1
    check("H(SLSQP) ≥ H(爬山)−1e-3", ok, f"{sls['H_bits']:.6f} vs {hill['H_bits']:.6f}")

    print("\n[3] §3.8 近似初值 + SLSQP")
    q_init = approx_warmstart_weights(probs_coin, grid_coin, r0=0.0, t=0.5)
    ok = len(q_init) == 2 and all(0 <= x <= 1 for x in q_init)
    if not ok:
        failures += 1
    check("warmstart 向量在约束内", ok, f"q_init={q_init}")
    warm = solve_max_entropy_warmstart(probs_coin, grid_coin, r0=0.0)
    ok = close(warm["H_bits"], sls["H_bits"], 1e-3)
    if not ok:
        failures += 1
    check(
        "H(warm)≈H(SLSQP)",
        ok,
        f"warm={warm['H_bits']:.6f}, slsqp={sls['H_bits']:.6f}",
    )
    ok = warm["method"] == "approx_warmstart_slsqp" and "warmstart" in warm
    if not ok:
        failures += 1
    check("warm 方法元数据", ok)

    print("\n[4] 4×5 边际展开后三方法一致")
    marg = [
        [(0.1, -0.2), (0.3, 0.05), (0.4, 0.15), (0.2, 0.4)],
        [(0.1, -0.5), (0.2, -0.1), (0.3, 0.08), (0.25, 0.2), (0.15, 0.6)],
    ]
    pj, gj = expand_joint_from_marginals(marg)
    cmp = compare_solvers(pj, gj, r0=0.0)
    Hs = cmp["H_bits"]
    ok = all(h is not None and h == h and h > float("-inf") for h in Hs.values())
    if not ok:
        failures += 1
    check("三方法 H 有限", ok, str({k: round(v, 6) for k, v in Hs.items()}))
    # 最优与次优 H 差
    h_sls = Hs["slsqp"]
    h_hill = Hs["hill_climb"]
    h_warm = Hs["approx_warmstart_slsqp"]
    ok = h_sls >= max(h_hill, h_warm) - 1e-4 or abs(h_sls - max(h_hill, h_warm)) < 1e-3
    if not ok:
        failures += 1
    check("SLSQP 不劣于其它 (凸问题全局)", ok, f"slsqp={h_sls:.6f}")
    ok = abs(h_warm - h_sls) < 1e-3
    if not ok:
        failures += 1
    check("warm≈slsqp", ok, f"warm={h_warm:.6f}, slsqp={h_sls:.6f}")
    ok = cmp["best"] in ("slsqp", "approx_warmstart_slsqp", "hill_climb")
    if not ok:
        failures += 1
    check("compare_solvers.best 可读", ok, cmp["best"])

    print("\n[5] 三资产 / 不等概率 压力")
    probs3 = [0.1, 0.2, 0.3, 0.25, 0.15]
    grid3 = [
        [-0.2, -0.1, 0.0],
        [0.05, -0.3, 0.1],
        [0.15, 0.1, 0.05],
        [0.3, 0.2, -0.05],
        [0.1, 0.5, 0.4],
    ]
    cmp3 = compare_solvers(probs3, grid3, r0=0.05)
    h3 = cmp3["H_bits"]
    ok = abs(h3["approx_warmstart_slsqp"] - h3["slsqp"]) < 1e-3
    if not ok:
        failures += 1
    check(
        "3资产 warm≈slsqp",
        ok,
        f"warm={h3['approx_warmstart_slsqp']:.6f}, slsqp={h3['slsqp']:.6f}",
    )
    ok = h3["slsqp"] >= h3["hill_climb"] - 5e-3
    if not ok:
        failures += 1
    check(
        "3资产 slsqp ≥ hill−5e-3",
        ok,
        f"slsqp={h3['slsqp']:.6f}, hill={h3['hill_climb']:.6f}",
    )
    w3 = cmp3["weights"]["slsqp"]
    ok = abs(sum(w3) - 1.0) < 1e-6
    if not ok:
        failures += 1
    check("Σq=1", ok, f"sum={sum(w3)}")

    print("\n[6] allow_short / allow_leverage 边界")
    # 等概率 {-0.1, 0.3} 单资产: 闭式 q'≈3.33 → 杠限时取 M
    s_lev = solve_max_entropy_slsqp(
        [0.5, 0.5], [[-0.1], [0.3]], r0=0.0, allow_leverage=True, max_multiple=2.0
    )
    ok = s_lev["asset_weights"][0] > 1.0
    if not ok:
        failures += 1
    check("透支时 q>1", ok, f"q={s_lev['asset_weights'][0]:.4f}")
    ok = s_lev["asset_weights"][0] <= 2.0 + 1e-6
    if not ok:
        failures += 1
    check("透支不超 M", ok, f"q={s_lev['asset_weights'][0]:.4f}")

    print("\n[7] 方法元数据齐全")
    for key, m in [
        ("hill", optimize_portfolio_entropy([0.5, 0.5], [[-1.0], [2.0]])),
        ("slsqp", solve_max_entropy_slsqp([0.5, 0.5], [[-1.0], [2.0]])),
        ("warm", solve_max_entropy_warmstart([0.5, 0.5], [[-1.0], [2.0]])),
    ]:
        ok = "method" in m and "weights" in m and "H_bits" in m
        if not ok:
            failures += 1
        check(f"{key} 结构完整", ok, m.get("method"))

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
