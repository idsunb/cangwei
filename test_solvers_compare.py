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

    print("\n[6b] 用户例: {0.5|+0.08, 0.5|-0.05}, r0=0 → 闭式 q*=3.75")
    # 书中 (3.2.3): q' = -(P1Δ1+P2Δ2)/(Δ1Δ2)*R0
    # D1=-0.05, D2=0.08 → q' = 3.75
    u_p = [0.5, 0.5]
    u_r = [[0.08], [-0.05]]
    closed_u = optimal_position_single([0.5, 0.5], [0.08, -0.05], r0=0.0,
                                       allow_leverage=True, max_multiple=10)
    ok = close(closed_u.q_star, 3.75, 1e-6)
    if not ok:
        failures += 1
    check("闭式 q*=3.75", ok, f"got {closed_u.q_star}")

    s_M1 = solve_max_entropy_slsqp(u_p, u_r, r0=0.0, allow_leverage=True, max_multiple=1.0)
    ok = close(s_M1["asset_weights"][0], 1.0, 1e-5)
    if not ok:
        failures += 1
    check("M=1 时 q=1（透支上限卡住）", ok, f"q={s_M1['asset_weights'][0]:.6f}")

    s_M2 = solve_max_entropy_slsqp(u_p, u_r, r0=0.0, allow_leverage=True, max_multiple=2.0)
    ok = close(s_M2["asset_weights"][0], 2.0, 1e-5)
    if not ok:
        failures += 1
    check("M=2 时 q=2", ok, f"q={s_M2['asset_weights'][0]:.6f}")

    s_M5 = solve_max_entropy_slsqp(u_p, u_r, r0=0.0, allow_leverage=True, max_multiple=5.0)
    ok = close(s_M5["asset_weights"][0], 3.75, 1e-4)
    if not ok:
        failures += 1
    check("M=5 时 q=3.75（达闭式最优）", ok, f"q={s_M5['asset_weights'][0]:.6f}")
    ok = s_M5["debt"] > 1.0 and close(s_M5["cash"], 0.0, 1e-9)
    if not ok:
        failures += 1
    check("透支后显示为负债(现金0)", ok, f"cash={s_M5['cash']:.6f}, debt={s_M5['debt']:.6f}")
    # H(3.75) > H(1)
    ok = s_M5["H_bits"] > s_M1["H_bits"] + 1e-3
    if not ok:
        failures += 1
    check("H(q=3.75)>H(q=1)", ok,
          f"H5={s_M5['H_bits']:.6f}, H1={s_M1['H_bits']:.6f}")

    # 爬山与 warmstart 同样尊重 M
    hill_m5 = optimize_portfolio_entropy(
        u_p, u_r, r0=0.0, allow_leverage=True, max_multiple=5.0
    )
    ok = close(hill_m5["asset_weights"][0], 3.75, 0.05)
    if not ok:
        failures += 1
    check("爬山 M=5 → q≈3.75", ok, f"q={hill_m5['asset_weights'][0]:.4f}")
    warm_m5 = solve_max_entropy_warmstart(
        u_p, u_r, r0=0.0, allow_leverage=True, max_multiple=5.0
    )
    ok = close(warm_m5["asset_weights"][0], 3.75, 1e-3)
    if not ok:
        failures += 1
    check("warm M=5 → q≈3.75", ok, f"q={warm_m5['asset_weights'][0]:.4f}")

    print("\n[6c] 现金/标的/负债 三项拆分 (透支 2.75)")
    from position_sizing import split_cash_debt

    sp = split_cash_debt(-2.75, [3.75])
    ok = close(sp["cash"], 0.0) and close(sp["debt"], 2.75) and close(sp["asset_sum"], 3.75)
    if not ok:
        failures += 1
    check("拆分 现金0/负债2.75/标的3.75", ok, str(sp))
    sp0 = split_cash_debt(0.4, [0.6])
    ok = close(sp0["cash"], 0.4) and close(sp0["debt"], 0.0)
    if not ok:
        failures += 1
    check("无透支时 debt=0", ok, str(sp0))
    # SLSQP M=5 结果里 cash/debt 字段
    s = s_M5
    ok = close(s.get("debt", -1), 2.75, 1e-4) and close(s.get("cash", -1), 0.0, 1e-9)
    if not ok:
        failures += 1
    check("SLSQP 结果含 cash/debt", ok,
          f"cash={s.get('cash')}, debt={s.get('debt')}, raw={s.get('cash_raw')}")
    ok = abs(s["cash"] + s["debt"] + abs(s["cash_raw"])) >= 0  # sanity
    ok = close(s["cash_raw"], -2.75, 1e-4) and close(s["cash"] + 0 * s["debt"], 0.0, 1e-9)
    if not ok:
        failures += 1
    check("cash_raw 仍为 -2.75 供计算", ok, f"raw={s['cash_raw']}")

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

    print("\n[9] 对比结果含耗时 elapsed_ms")
    probs = [0.25] * 4
    grid = [[-1, -1], [-1, 2], [2, -1], [2, 2]]
    cmp_t = compare_solvers(probs, grid, r0=0.0)
    em = cmp_t.get("elapsed_ms", {})
    ok = all(k in em and em[k] is not None and em[k] >= 0 for k in
             ("hill_climb", "slsqp", "approx_warmstart_slsqp"))
    if not ok:
        failures += 1
    check("compare_solvers.elapsed_ms 齐全", ok, str({k: round(v, 3) for k, v in em.items()}))
    ok = all("elapsed_ms" in v for v in cmp_t["methods"].values())
    if not ok:
        failures += 1
    check("各方法结果含 elapsed_ms", ok)

    print("\n[8] 三资产+透支: 爬山/初值 与 SLSQP 对齐")
    hard_p = [0.1, 0.2, 0.3, 0.25, 0.15]
    hard_g = [
        [-0.2, -0.1, 0.0],
        [0.05, -0.3, 0.1],
        [0.15, 0.1, 0.05],
        [0.3, 0.2, -0.05],
        [0.1, 0.5, 0.4],
    ]
    hs = solve_max_entropy_slsqp(hard_p, hard_g, r0=0.0, allow_leverage=True, max_multiple=5.0)
    hh = optimize_portfolio_entropy(hard_p, hard_g, r0=0.0, allow_leverage=True, max_multiple=5.0)
    hw = solve_max_entropy_warmstart(hard_p, hard_g, r0=0.0, allow_leverage=True, max_multiple=5.0)
    ok = abs(hh["H_bits"] - hs["H_bits"]) < 5e-3
    if not ok:
        failures += 1
    check("爬山 H≈SLSQP", ok, f"hill={hh['H_bits']:.6f}, slsqp={hs['H_bits']:.6f}")
    ok = abs(hw["H_bits"] - hs["H_bits"]) < 1e-3
    if not ok:
        failures += 1
    check("warm H≈SLSQP", ok, f"warm={hw['H_bits']:.6f}, slsqp={hs['H_bits']:.6f}")
    # Σq 约束
    ok = abs(sum(hh["asset_weights"]) - 5.0) < 1e-3 or sum(hh["asset_weights"]) <= 5.0 + 1e-6
    if not ok:
        failures += 1
    check("爬山 Σq≤M", ok, f"sum={sum(hh['asset_weights']):.6f}")
    ok = all(x <= 5.0 + 1e-6 for x in hh["asset_weights"])
    if not ok:
        failures += 1
    check("爬山 各 q≤M", ok, str(hh["asset_weights"]))

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
