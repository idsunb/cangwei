# -*- coding: utf-8 -*-
"""
《投资组合的熵理论和信息价值》(鲁晨光, 1997) 仓位管理公式实现.

对应章节:
  §2.1  基本概念 (收益率 / 产出比 / 几何平均)
  §3.1  最大增值熵原理
  §3.2  单硬币打赌下注优化
  §3.3  允许透支和卖空
  §3.4  考虑转移成本的增量优化
  §3.5  多硬币打赌下注优化
  §3.6  分散投资极限定理
  §3.7  消费增值熵
  §3.8  近似抛物线优化
  §5.4  期货盈亏空间定头寸
  §6.2  期权头寸控制
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 基本概念 §2.1
# ---------------------------------------------------------------------------

def excess_return(r: float, r0: float) -> float:
    """超常收益 Δ = r - r0."""
    return r - r0


def expected_return(probs: Sequence[float], returns: Sequence[float]) -> float:
    """期望(算术平均)收益 r_a = Σ P_i r_i."""
    return float(sum(p * r for p, r in zip(probs, returns)))


def std_dev(probs: Sequence[float], returns: Sequence[float]) -> float:
    """标准方差 σ = sqrt(Σ P_i (r_i - r_a)^2)."""
    ra = expected_return(probs, returns)
    return float(math.sqrt(sum(p * (r - ra) ** 2 for p, r in zip(probs, returns))))


def geometric_mean_return(probs: Sequence[float], returns: Sequence[float]) -> float:
    """几何平均收益 r_g = ∏ (1+r_i)^{P_i} - 1."""
    log_rg = 0.0
    for p, r in zip(probs, returns):
        R = 1.0 + r
        if R <= 0:
            # 产出比非正：几何平均无定义(亏光)，返回 -1
            return -1.0
        log_rg += p * math.log(R)
    return math.exp(log_rg) - 1.0


# ---------------------------------------------------------------------------
# 增值熵 §3.1
# ---------------------------------------------------------------------------

def growth_entropy(
    probs: Sequence[float],
    portfolio_returns: Sequence[float],
    base: float = 2.0,
) -> float:
    """
    增值熵 H = Σ P_i log(R_i),  R_i = 1 + 组合在情景 i 下的收益.

    base=2 时单位为比特。若某情景 R_i <= 0，返回 -inf(亏光).
    """
    H = 0.0
    for p, r in zip(probs, portfolio_returns):
        if p == 0:
            continue
        R = 1.0 + r
        if R <= 0:
            return float("-inf")
        H += p * (math.log(R) / math.log(base))
    return H


def portfolio_scenario_returns(
    weights: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
) -> List[float]:
    """
    组合各情景收益率.
    weights: [q0, q1, ..., qN] 现金+各资产, Σq = 1
    asset_returns: shape (情景数, 资产数 N)
    """
    q0 = weights[0]
    qs = np.asarray(weights[1:], dtype=float)
    R0 = 1.0 + r0
    out = []
    for row in asset_returns:
        port_R = q0 * R0 + float(np.dot(qs, np.asarray(row, dtype=float) + 1.0))
        out.append(port_R - 1.0)
    return out


def entropy_of_weights(
    weights: Sequence[float],
    probs: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
    base: float = 2.0,
) -> float:
    rs = portfolio_scenario_returns(weights, asset_returns, r0=r0)
    return growth_entropy(probs, rs, base=base)


# ---------------------------------------------------------------------------
# §3.2 单硬币最优仓位
# ---------------------------------------------------------------------------

@dataclass
class SinglePositionResult:
    """单硬币模型最优仓位结果."""
    q_raw: float          # 未截断的 q'
    q_star: float         # 截断到 [0,1] 后的最优仓位
    cash: float           # q0 = 1 - q_star
    H: float              # 最优处增值熵(比特)
    r_g: float            # 最优几何平均收益
    r_a: float            # 期望收益
    formula: str          # 使用的公式说明
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "q_raw": self.q_raw,
            "q_star": self.q_star,
            "cash": self.cash,
            "H_bits": self.H,
            "geometric_return": self.r_g,
            "expected_return": self.r_a,
            "formula": self.formula,
            "notes": self.notes,
        }


def optimal_position_single(
    probs: Sequence[float],
    returns: Sequence[float],
    r0: float = 0.0,
    allow_leverage: bool = False,
    allow_short: bool = False,
    max_multiple: float = 1.0,
    r_loan: Optional[float] = None,
) -> SinglePositionResult:
    """
    单硬币打赌最优投资比例公式 (3.2.3).

    q' = -(P1 Δ1 + P2 Δ2) / (Δ1 Δ2) * R0

    参数
    ----
    probs, returns : 两种(或多种)情景的概率与收益率
    r0             : 存款/国债/无风险利率
    allow_leverage  : 是否允许透支/贷款
    allow_short     : 是否允许卖空 (q < 0)
    max_multiple    : 允许的最大资金倍数 M (1+可透支倍数)
    r_loan          : 贷款利率, 默认等于 r0

    返回 SinglePositionResult
    """
    if len(probs) != len(returns):
        raise ValueError("probs 与 returns 长度必须相同")
    if abs(sum(probs) - 1.0) > 1e-9:
        raise ValueError("概率之和必须为 1")

    r_loan = r0 if r_loan is None else r_loan
    R0 = 1.0 + r0
    R_loan = 1.0 + r_loan

    # 期望与熵辅助
    def H_at(q: float) -> float:
        # 分段资金成本: q>1 用贷款利率
        if q > 1:
            cash_R = (1.0 - q) * R_loan  # 借款为负现金
            rs = [cash_R - 1.0 + q * r for r in returns]
            # 等价: 组合收益 = - (q-1) r_loan + q r  ? 书中:
            # R = -q0' R0' + q R_k,  q0'=q-1
            rs = [-(q - 1.0) * R_loan + q * (1.0 + r) - 1.0 for r in returns]
        elif q < 0:
            # 卖空: 剩余现金 q0 = 1-|q| = 1+q, 卖空抵押同值
            # R = (1-|q|) R0 + q R_k = (1+q) R0 + q (1+r)  — 见书中 0≥q≥-1
            rs = [(1.0 + q) * R0 + q * (1.0 + r) - 1.0 for r in returns]
            if q < -1:
                # 超过 -1: 借款卖空
                rs = [-(abs(q) - 1.0) * R_loan + q * (1.0 + r) - 1.0 for r in returns]
        else:
            rs = [(1.0 - q) * R0 + q * (1.0 + r) - 1.0 for r in returns]
        return growth_entropy(probs, rs, base=2.0)

    # 需要恰好两种"方向相反"的超额收益才有闭式解
    deltas = [r - r0 for r in returns]
    notes_list: List[str] = []

    # 多情景或同号 Δ: 用数值搜索
    need_numeric = len(returns) != 2 or not (
        (deltas[0] < 0 < deltas[1]) or (deltas[1] < 0 < deltas[0])
    )

    if not need_numeric:
        # 规范: D1 = 较小超额收益(亏损侧), D2 = 较大超额收益(盈利侧)
        if deltas[0] <= deltas[1]:
            P1, P2 = probs[0], probs[1]
            D1, D2 = deltas[0], deltas[1]
        else:
            P1, P2 = probs[1], probs[0]
            D1, D2 = deltas[1], deltas[0]
        # 期望超常收益(用于判断空仓)
        E_excess = sum(p * d for p, d in zip(probs, deltas))
        if D1 >= 0:
            # 只赢不亏 → 满仓 (书中 §3.2)
            q_unbounded = 1.0
            notes_list.append("只赢不亏 → 满仓")
        elif D2 <= 0:
            q_unbounded = 0.0
            notes_list.append("只亏不赢 → 空仓(不许卖空)")
        elif E_excess <= 0:
            # 期望超常收益 ≤ 0 → 空仓更优 (空仓条件)
            q_unbounded = 0.0
            notes_list.append(f"期望超常收益={E_excess:.6f}≤0 → 空仓")
        else:
            # q' = -(P1 D1 + P2 D2)/(D1 D2) * R0
            q_unbounded = -((P1 * D1 + P2 * D2) / (D1 * D2)) * R0
            notes_list.append(
                f"闭式解 q' = -(P1Δ1+P2Δ2)/(Δ1Δ2)·R0 = {q_unbounded:.6f}"
            )
    else:
        # 数值一维搜索
        notes_list.append("多状态或同号超额收益 → 数值搜索")
        q_unbounded = _numeric_single(probs, returns, r0, r_loan, allow_leverage, allow_short, max_multiple, H_at)

    # 截断
    lo, hi = 0.0, 1.0
    if allow_short:
        lo = -max_multiple
    if allow_leverage:
        hi = max_multiple

    q_star = float(np.clip(q_unbounded, lo, hi))
    if q_unbounded > hi:
        notes_list.append(f"触及杠杆上限 M={max_multiple}")
    if q_unbounded < lo:
        notes_list.append(f"触及卖空下限 -M={-max_multiple}")

    # 统一在最优 q 处算熵
    if q_star > 1:
        rs = [-(q_star - 1.0) * R_loan + q_star * (1.0 + r) - 1.0 for r in returns]
    elif q_star < 0:
        if q_star >= -1:
            rs = [(1.0 + q_star) * R0 + q_star * (1.0 + r) - 1.0 for r in returns]
        else:
            rs = [-(abs(q_star) - 1.0) * R_loan + q_star * (1.0 + r) - 1.0 for r in returns]
    else:
        rs = [(1.0 - q_star) * R0 + q_star * (1.0 + r) - 1.0 for r in returns]

    H = growth_entropy(probs, rs, base=2.0)
    r_g = (2.0 ** H - 1.0) if H > float("-inf") else -1.0
    # 几何平均用自然底更直观: exp(Σ p ln R)-1
    log_rg = 0.0
    ok = True
    for p, r in zip(probs, rs):
        if 1.0 + r <= 0:
            ok = False
            break
        log_rg += p * math.log(1.0 + r)
    r_g = math.exp(log_rg) - 1.0 if ok else -1.0

    r_a = expected_return(probs, rs)

    formula = "q* = clip( -(P1Δ1+P2Δ2)/(Δ1Δ2) · R0 , [L,U] )"
    if need_numeric:
        formula = "q* = argmax_q H(q)  (数值搜索)"

    return SinglePositionResult(
        q_raw=float(q_unbounded),
        q_star=q_star,
        cash=1.0 - q_star,
        H=H,
        r_g=r_g,
        r_a=r_a,
        formula=formula,
        notes="; ".join(notes_list),
    )


def _numeric_single(probs, returns, r0, r_loan, allow_leverage, allow_short, max_multiple, H_at) -> float:
    lo = -max_multiple if allow_short else 0.0
    hi = max_multiple if allow_leverage else 1.0
    # 排除导致亏光的 q: 1+R_port < 0
    grid = np.linspace(lo, hi, 2001)
    best_q, best_H = 0.0, float("-inf")
    for q in grid:
        h = H_at(float(q))
        if h > best_H:
            best_H, best_q = h, float(q)
    return best_q


# ---------------------------------------------------------------------------
# §2.3 / §5.4.1 简化公式
# ---------------------------------------------------------------------------

def optimal_position_simple(probs: Sequence[float], returns: Sequence[float]) -> float:
    """
    r0=0 时的简单优化公式 (§2.3):
        q* = (P1 r1 + P2 r2) / |r1 r2|
    要求恰好两种收益。
    """
    if len(returns) != 2:
        raise ValueError("简化公式仅适用于两种收益")
    P1, P2 = probs
    r1, r2 = returns
    return (P1 * r1 + P2 * r2) / abs(r1 * r2)


def optimal_position_equal_prob(r_loss: float, r_gain: float, fee: float = 0.0) -> float:
    """
    等概率双侧简化式 (§5.4.1):
        q* = -1/(2 d') * (1/r2 + 1/r1),  d' = 1 ± fee
    这里 fee 为手续费率 d; 加仓方向 d'=1-d, 一般展示用 d'=1+fee 上界时更保守.
    书中未计资金成本; 此处 fee=0 时退化为 -0.5*(1/r2+1/r1).
    """
    d_prime = 1.0 + fee
    return -1.0 / (2.0 * d_prime) * (1.0 / r_gain + 1.0 / r_loss)


def optimal_position_asym_limit(p_loss: float, r_loss: float = -2.0, r_gain: float = 3.0) -> float:
    """
    盈亏幅度为 (r_loss, r_gain) 时, 最优仓位随亏损概率 P1 的公式.
    书中特例 r1=-2, r2=3 (R0=1, 允许卖空):
        q* = (3 - 5 P1) / 6
    这里给出一般形式: q' = -(P1 Δ1 + P2 Δ2)/(Δ1 Δ2), Δ=r (r0=0),
    并截断到 [-1, 1]; P1=0 → 1, P1=1 → -1 时使用书中极限约定.
    """
    r0 = 0.0
    D1, D2 = r_loss - r0, r_gain - r0
    P1 = p_loss
    P2 = 1.0 - P1
    if P1 <= 0.0:
        return 1.0
    if P1 >= 1.0:
        return -1.0
    q = -((P1 * D1 + P2 * D2) / (D1 * D2)) * (1.0 + r0)
    return float(np.clip(q, -1.0, 1.0))


# ---------------------------------------------------------------------------
# §3.3 透支 + 卖空
# ---------------------------------------------------------------------------

@dataclass
class LeverageShortResult:
    q_star: float
    region: str
    q_prime: float
    q_double_prime: float
    H: float
    r_g: float
    notes: str

    def as_dict(self) -> dict:
        return {
            "q_star": self.q_star,
            "region": self.region,
            "q_prime": self.q_prime,
            "q_double_prime": self.q_double_prime,
            "H_bits": self.H,
            "geometric_return": self.r_g,
            "notes": self.notes,
        }


def optimal_position_leverage_short(
    probs: Sequence[float],
    returns: Sequence[float],
    r0: float = 0.1,
    r_loan: float = 0.15,
    max_multiple: float = 2.0,
) -> LeverageShortResult:
    """
    允许透支和卖空时的单硬币最优仓位 (§3.3).

    M = max_multiple = 1 + 可透支倍数.
    分段: 透支满仓 / 透支区 / 正常多头 / 空仓 / 卖空 / 深度卖空.
    """
    P1, P2 = probs
    r1, r2 = returns
    # 规范: 损失方 Δ<0
    if (r1 - r0) > (r2 - r0):
        r1, r2 = r2, r1
        P1, P2 = P2, P1

    R0 = 1.0 + r0
    RP = 1.0 + r_loan
    D1, D2 = r1 - r0, r2 - r0          # 正常区超常收益
    D1p, D2p = r1 - r_loan, r2 - r_loan  # 透支区
    Dm1, Dm2 = r1 + r0, r2 + r0          # 卖空区(抵押得利息 r0)
    Dm1p, Dm2p = r1 + r_loan, r2 + r_loan

    def raw(Da, Db, Rx, Pa, Pb) -> float:
        if Da == 0 or Db == 0:
            return 0.0
        return -((Pa * Da + Pb * Db) / (Da * Db)) * Rx

    q_pp = raw(D1p, D2p, RP, P1, P2)   # 透支区 q''
    q_p = raw(D1, D2, R0, P1, P2)      # 正常区 q'
    q_mp = raw(Dm1, Dm2, R0, P1, P2)   # 卖空区 q_'
    q_mpp = raw(Dm1p, Dm2p, RP, P1, P2)

    # 期望超常收益判断空仓: E 在 [-r0, r0] 附近 → 空仓
    E = P1 * r1 + P2 * r2

    M = max_multiple
    if q_pp >= M:
        q, region = M, "透支满仓"
    elif q_pp >= 1.0:
        q, region = q_pp, "透支区"
    elif q_p >= 0.0 and q_p <= 1.0 and E > r0:
        q, region = q_p, "正常多头"
    elif -r0 <= E <= r0 or (q_p < 0 and q_mp > 0):
        # 空仓带
        if q_p < 0 and not (q_mp <= 0 <= q_p and abs(q_mp) < 1):
            # 用书中: 0 if -r0 ≤ E ≤ r0
            if -r0 <= E <= r0:
                q, region = 0.0, "空仓"
            elif q_p < 0:
                # 进入卖空判定
                if -1.0 <= q_mp <= 0.0:
                    q, region = q_mp, "卖空区"
                elif q_mpp < -1.0 and q_mpp > -M:
                    q, region = q_mpp, "深度卖空"
                elif q_mpp <= -M:
                    q, region = -M, "卖空上限"
                else:
                    q, region = 0.0, "空仓"
            else:
                q, region = float(np.clip(q_p, 0, 1)), "正常多头"
        else:
            q, region = 0.0, "空仓"
    elif q_p > 1.0:
        q, region = min(q_p, M), "透支区" if q_p < M else "透支满仓"
    elif q_p < 0:
        if -1.0 <= q_mp <= 0.0:
            q, region = q_mp, "卖空区"
        elif q_mpp < -1.0 and q_mpp > -M:
            q, region = q_mpp, "深度卖空"
        elif q_mpp <= -M:
            q, region = -M, "卖空上限"
        else:
            q, region = 0.0, "空仓"
    else:
        q, region = float(np.clip(q_p, 0, 1)), "正常多头"

    # 再精确化空仓: 书中空仓条件 Δ2 ≤ (R0-1-P1 Δ1)/P2 且 q'<0
    if region == "正常多头" and q_p < 0:
        region = "空仓"
        q = 0.0

    q = float(np.clip(q, -M, M))

    # 熵
    if q > 1:
        rs = [-(q - 1) * RP + q * (1 + r) - 1 for r in (r1, r2)]
    elif q >= 0:
        rs = [(1 - q) * R0 + q * (1 + r) - 1 for r in (r1, r2)]
    elif q >= -1:
        rs = [(1 + q) * R0 + q * (1 + r) - 1 for r in (r1, r2)]
    else:
        rs = [-(abs(q) - 1) * RP + q * (1 + r) - 1 for r in (r1, r2)]

    H = growth_entropy([P1, P2], rs, base=2.0)
    if H > float("-inf"):
        log_rg = sum(p * math.log(1 + r) for p, r in zip([P1, P2], rs) if 1 + r > 0)
        r_g = math.exp(log_rg) - 1 if all(1 + r > 0 for r in rs) else -1.0
    else:
        r_g = -1.0

    return LeverageShortResult(
        q_star=q,
        region=region,
        q_prime=q_p,
        q_double_prime=q_pp,
        H=H,
        r_g=r_g,
        notes=f"E={E:.4f}, M={M}, r0={r0}, r_loan={r_loan}",
    )


# ---------------------------------------------------------------------------
# §3.4 手续费增量优化
# ---------------------------------------------------------------------------

@dataclass
class IncrementResult:
    delta_q: float
    q_new_raw: float
    q_new: float
    hold: bool
    formula_terms: dict
    notes: str

    def as_dict(self) -> dict:
        return {
            "delta_q": self.delta_q,
            "q_new_raw": self.q_new_raw,
            "q_new": self.q_new,
            "hold": self.hold,
            **self.formula_terms,
            "notes": self.notes,
        }


def optimal_position_increment(
    q_current: float,
    probs: Sequence[float],
    returns: Sequence[float],
    r0: float = 0.0,
    fee: float = 0.0075,
    q_min: float = -1.0,
    q_max: float = 1.0,
) -> IncrementResult:
    """
    考虑交易手续费的仓位增量优化公式 (3.4.3).

    d = fee, 加仓时 d' = 1-d, 减仓时 d' = 1+d (书中: 增加头寸用-, 减少用+).
    这里对闭式解分别试算加/减方向, 并优先选择 |Δq| 较小且提高 H 的方向;
    若 Δq 落在"不动带"(|Δq| 很小且手续费下不划算)则 hold=True.

    当 q_current=0 时退化为含手续费的比例优化.
    """
    P1, P2 = probs
    r1, r2 = returns
    R0 = 1.0 + r0
    q_h = q_current
    q0_h = 1.0 - q_h

    def H_of(q: float) -> float:
        rs = [(1 - q) * R0 + q * (1 + r) - 1 for r in returns]
        # 若 q 卖空更复杂; 此处实现书中假设 1≥q≥-1 的简化现金结构
        if q < 0:
            rs = [(1 - abs(q)) * R0 + q * (1 + r) - 1 for r in returns]
        return growth_entropy(probs, rs, base=2.0)

    # 分别用加仓 d'=1-d 与减仓 d'=1+d 计算闭式 Δq*
    results = []
    for sign, d_prime in (("increase", 1.0 - fee), ("decrease", 1.0 + fee)):
        d1s = d_prime * (1 + r1) - R0
        d2s = d_prime * (1 + r2) - R0
        R01 = q0_h * R0 + q_h * (1 + r1)
        R02 = q0_h * R0 + q_h * (1 + r2)
        if d1s == 0 or d2s == 0:
            continue
        dq = -((P1 * d1s * R02 + P2 * d2s * R01) / (d1s * d2s))
        results.append((sign, d_prime, dq, d1s, d2s, R01, R02))

    if not results:
        return IncrementResult(0.0, q_h, q_h, True, {}, "无法计算")

    # 选择使 H 提高最多且满足区间约束的 Δq
    best = None
    best_H = H_of(q_h)
    for sign, d_prime, dq, d1s, d2s, R01, R02 in results:
        q_new_raw = q_h + dq
        # 方向一致性: increase 应 dq>=0, decrease 应 dq<=0 (书中的分段)
        if sign == "increase" and dq < 0:
            continue
        if sign == "decrease" and dq > 0:
            continue
        q_new = float(np.clip(q_new_raw, q_min, q_max))
        h = H_of(q_new)
        if h > best_H + 1e-12:
            best_H = h
            best = (sign, d_prime, dq, q_new_raw, q_new, d1s, d2s, R01, R02)

    if best is None:
        # 允许不动
        return IncrementResult(
            delta_q=0.0,
            q_new_raw=q_h,
            q_new=q_h,
            hold=True,
            formula_terms={"fee": fee},
            notes="手续费下持仓不变 (白带区)",
        )

    sign, d_prime, dq, q_raw, q_new, d1s, d2s, R01, R02 = best
    hold = abs(q_new - q_h) < 1e-9
    return IncrementResult(
        delta_q=dq,
        q_new_raw=q_raw,
        q_new=q_new,
        hold=hold,
        formula_terms={
            "direction": sign,
            "d_prime": d_prime,
            "fee": fee,
            "delta_hash_1": d1s,
            "delta_hash_2": d2s,
            "R_hash_01": R01,
            "R_hash_02": R02,
            "H_before": H_of(q_h),
            "H_after": best_H,
        },
        notes="Δq* = -(P1 Δ#1 R#02 + P2 Δ#2 R#01)/(Δ#1 Δ#2)",
    )


# ---------------------------------------------------------------------------
# §3.5 / §3.6 多硬币
# ---------------------------------------------------------------------------

def multi_coin_entropy(
    n_assets: int,
    q_total: float,
    r_loss: float,
    r_gain: float,
    p_loss: float,
    base: float = 2.0,
) -> float:
    """
    N 个独立同分布硬币、等权 q_k = q/N 时的增值熵 (3.5.4).
    r0=0.
    """
    p = p_loss
    q_win = 1.0 - p
    N = n_assets
    H = 0.0
    for S in range(N + 1):
        # S = 赢的次数 (书中符号可能对调; 这里 S 赢, N-S 输)
        # 按书中: P 对应 r1(亏), Q 对应 r2(赢)
        # P(S_loss) with S_loss = N-S ...
        # 使用: 输 k 次, 赢 N-k 次
        k_loss = S
        k_win = N - S
        # 二项概率
        logC = math.lgamma(N + 1) - math.lgamma(k_loss + 1) - math.lgamma(k_win + 1)
        prob = math.exp(logC + k_loss * math.log(p) + k_win * math.log(q_win)) if q_win > 0 or k_win == 0 else 0.0
        if p <= 0 and k_loss > 0:
            prob = 0.0
        if q_win <= 0 and k_win > 0:
            prob = 0.0
        # 组合收益: 总仓位 q, 均分 N 份
        # 收益 = (k_loss * r_loss + k_win * r_gain) * (q/N) * N? 不对
        # 每资产比例 q/N, 总收益 Σ (q/N) r_i = (q/N)(k_loss r_loss + k_win r_gain)
        r_port = (q_total / N) * (k_loss * r_loss + k_win * r_gain)
        R = 1.0 + r_port
        if R <= 0:
            return float("-inf")
        if prob > 0:
            H += prob * (math.log(R) / math.log(base))
    return H


def optimize_multi_coin_equal(
    n_assets: int,
    r_loss: float = -1.0,
    r_gain: float = 2.0,
    p_loss: float = 0.5,
    allow_leverage: bool = False,
    allow_short: bool = False,
) -> dict:
    """
    等权多硬币最优总仓位 q* (数值), 以及各项 q_k = q*/N.
    对应书中表 2.5: 硬币 {-1,2}, p=0.5 → N=1:25%, N=2:23%×2 ...
    """
    hi = 2.0 if allow_leverage else 1.0
    lo = -1.0 if allow_short else 0.0
    # 可行域: 每资产 r 导致部分亏光则熵 -inf, 网格搜索
    grid = np.linspace(lo, hi, 4001)
    best_q, best_H = 0.0, float("-inf")
    for q in grid:
        H = multi_coin_entropy(n_assets, float(q), r_loss, r_gain, p_loss)
        if H > best_H:
            best_H, best_q = H, float(q)
    q_star = float(best_q)
    # 实际几何平均
    log_rg = 0.0
    p = p_loss
    q_win = 1 - p
    N = n_assets
    ok = True
    for k_loss in range(N + 1):
        k_win = N - k_loss
        logC = math.lgamma(N + 1) - math.lgamma(k_loss + 1) - math.lgamma(k_win + 1)
        prob = math.exp(logC + (k_loss * math.log(p) if p > 0 or k_loss == 0 else float("-inf"))
                        + (k_win * math.log(q_win) if q_win > 0 or k_win == 0 else float("-inf")))
        r_port = (q_star / N) * (k_loss * r_loss + k_win * r_gain)
        if 1 + r_port <= 0:
            ok = False
            break
        log_rg += prob * math.log(1 + r_port)
    r_g = math.exp(log_rg) - 1 if ok else -1.0
    return {
        "n_assets": n_assets,
        "q_total": q_star,
        "q_each": q_star / N,
        "H_bits": best_H,
        "geometric_return": r_g,
    }


def diversification_limit_check(
    r_loss: float = -1.0,
    r_gain: float = 2.0,
    p_loss: float = 0.5,
    n_list: Sequence[int] = (1, 2, 3, 4, 8, 16, 64),
) -> List[dict]:
    """分散投资极限定理数值演示: N↑ → r_g → E."""
    E = p_loss * r_loss + (1 - p_loss) * r_gain
    rows = []
    for N in n_list:
        info = optimize_multi_coin_equal(N, r_loss, r_gain, p_loss)
        info["E"] = E
        rows.append(info)
    return rows


# ---------------------------------------------------------------------------
# §3.7 消费增值熵
# ---------------------------------------------------------------------------

def consumer_growth_entropy(
    probs: Sequence[float],
    returns: Sequence[float],
    q: float = 1.0,
    r0: float = 0.0,
    consume_ratio: float = 0.4,
    r_sacrifice: float = 0.15,
) -> float:
    """
    消费增值熵 Hc (§3.7).

    K = consume_ratio 可能用于消费的资金比例
    K_- = 1-K
    q_-0 = max(0, K_- - q)  非消费现金比例 (书中符号略糊, 按文意)
    q_-0'' = max(0, q - K_-) 占用消费资金的比例
    Hc = Σ P log(1 + r0 q_-0 - r_sacrifice q_-0'' + r_i q)
    """
    Km = 1.0 - consume_ratio
    # 文中例: r0=0, K=0.4, r_s=0.15, q=1 → Hc=log(1-0.15*0.4+0.2)
    # 即占用消费资金的比例为 K(而非 K_-), 抵扣非消费现金利息按 max(0, 1-q-K) 一类简化:
    # Hc = log(1 + r0*max(0,1-q) - r_s*min(q,K) + r*q) 在 r0=0 时 → 1 - r_s*K + r*q
    q_use_consume = min(max(q, 0.0), consume_ratio)
    q_free_cash = max(0.0, 1.0 - q)
    rs = [
        r0 * q_free_cash - r_sacrifice * q_use_consume + r * q
        for r in returns
    ]
    return growth_entropy(probs, rs, base=2.0)


# ---------------------------------------------------------------------------
# §3.8 抛物线近似
# ---------------------------------------------------------------------------

def optimal_position_parabolic_approx(
    probs: Sequence[float],
    returns: Sequence[float],
    r0: float = 0.0,
    t: float = 0.7,
    q_min: float = 0.0,
    q_max: float = 1.0,
) -> dict:
    """
    近似增值熵 h(q)=a q^2 + b q + c, 三点 q=0, t/2, t (§3.8).
    q* ≈ clip(-b/(2a), q_min, q_max).
    """
    def H(q: float) -> float:
        R0 = 1.0 + r0
        rs = [(1 - q) * R0 + q * (1 + r) - 1 for r in returns]
        return growth_entropy(probs, rs, base=2.0)

    H0, Ht2, Ht = H(0.0), H(t / 2.0), H(t)
    a = 2.0 * (Ht - 2.0 * Ht2 + H0) / (t ** 2)
    b = (4.0 * Ht2 - Ht - 3.0 * H0) / t
    c = H0
    if a == 0:
        q = q_max if b > 0 else q_min
    else:
        q = -b / (2.0 * a)
    q_clip = float(np.clip(q, q_min, q_max))
    return {
        "a": a, "b": b, "c": c,
        "q_approx": q_clip,
        "q_unclipped": q,
        "H_at_q": H(q_clip),
        "H_at_exact_search": max(
            (H(float(x)), float(x)) for x in np.linspace(q_min, q_max, 1001)
        ),
        "t": t,
    }


# ---------------------------------------------------------------------------
# 多资产数值优化 (对应书中"电脑优化")
# ---------------------------------------------------------------------------

def optimize_portfolio_entropy(
    probs: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
    allow_short: bool = False,
    allow_leverage: bool = False,
    max_multiple: float = 1.0,
    n_assets: Optional[int] = None,
) -> dict:
    """
    最大增值熵原理的数值实现 (§3.1).

    probs: 长度 W 的情景概率
    asset_returns: W × N 矩阵, 每格为该资产在该情景的收益率 r
    返回最优权重 [q0, q1, ..., qN].
    """
    P = np.asarray(probs, dtype=float)
    A = np.asarray(asset_returns, dtype=float)
    W, N = A.shape
    if n_assets is not None:
        assert N == n_assets
    R0 = 1.0 + r0

    lo_asset = -max_multiple if allow_short else 0.0
    hi_asset = max_multiple if allow_leverage else 1.0

    def H_of(q_assets: np.ndarray) -> float:
        q_sum = float(np.sum(q_assets))
        q0 = 1.0 - q_sum
        if not allow_leverage and q0 < -1e-12:
            return float("-inf")
        if not allow_short and np.any(q_assets < -1e-12):
            return float("-inf")
        # 组合产出比
        # R_w = q0 R0 + Σ q_k (1+r_wk)
        port_R = q0 * R0 + (A + 1.0) @ q_assets
        if np.any(port_R <= 0):
            return float("-inf")
        return float(np.sum(P * np.log(port_R) / np.log(2.0)))

    # 多起点坐标下降 + 网格粗搜
    best_q = np.zeros(N)
    best_H = H_of(best_q)

    # 粗网格: 单资产扫描
    for k in range(N):
        for qk in np.linspace(lo_asset, hi_asset, 51):
            qa = np.zeros(N)
            qa[k] = qk
            h = H_of(qa)
            if h > best_H:
                best_H, best_q = h, qa.copy()

    # 随机多起点 + 投影梯度式爬山
    rng = np.random.default_rng(42)
    for _ in range(30):
        q = best_q + rng.normal(0, 0.15, size=N)
        q = np.clip(q, lo_asset, hi_asset)
        # 投影: 总仓位不超过约束
        if not allow_leverage:
            s = q.sum()
            if s > max_multiple:
                q = q * (max_multiple / s)
        h = H_of(q)
        if h > best_H:
            best_H, best_q = h, q

    # 坐标爬山
    step = 0.08
    for _ in range(80):
        improved = False
        for k in range(N):
            for direction in (step, -step):
                q = best_q.copy()
                q[k] = float(np.clip(q[k] + direction, lo_asset, hi_asset))
                if not allow_leverage and q.sum() > max_multiple + 1e-12:
                    continue
                h = H_of(q)
                if h > best_H + 1e-12:
                    best_H, best_q = h, q
                    improved = True
        if not improved:
            step *= 0.5
            if step < 1e-4:
                break

    q0 = 1.0 - float(best_q.sum())
    port_rs = list(q0 * R0 + (A + 1.0) @ best_q - 1.0)  # wrong shape
    # 正确:
    port_R = q0 * R0 + (A + 1.0) @ best_q
    port_rs = list(port_R - 1.0)
    H = growth_entropy(P, port_rs, base=2.0)
    if all(1 + r > 0 for r in port_rs):
        r_g = math.exp(sum(p * math.log(1 + r) for p, r in zip(P, port_rs))) - 1
    else:
        r_g = -1.0

    return {
        "weights": [q0] + list(map(float, best_q)),
        "cash": q0,
        "asset_weights": list(map(float, best_q)),
        "H_bits": H,
        "geometric_return": r_g,
        "expected_return": float(P @ np.asarray(port_rs)),
        "scenario_returns": port_rs,
        "method": "hill_climb",
    }


# ---------------------------------------------------------------------------
# 凸优化求解器: SLSQP (§3.1 标准型) 与 §3.8 近似初值 + 精修
# ---------------------------------------------------------------------------

def _portfolio_metrics(P, Delta, R0, q_assets):
    """由 q_assets 算 H(bits), r_g, r_a, port_rs, port_R."""
    port_R = R0 + Delta @ q_assets
    if np.any(port_R <= 0):
        return {
            "H_bits": float("-inf"),
            "geometric_return": -1.0,
            "expected_return": float("-inf"),
            "scenario_returns": [float("-inf")] * len(P),
            "port_R": port_R,
        }
    # H bits
    H = float(np.sum(P * np.log(port_R) / np.log(2.0)))
    r_g = float(np.exp(np.sum(P * np.log(port_R))) - 1.0)
    port_rs = port_R - 1.0
    r_a = float(P @ port_rs)
    return {
        "H_bits": H,
        "geometric_return": r_g,
        "expected_return": r_a,
        "scenario_returns": list(map(float, port_rs)),
        "port_R": port_R,
    }


def _neg_H_grad(q, P, Delta, R0, allow_short, allow_leverage, max_multiple):
    """
    目标 f(q) = -Σ P ln(R0 + Δ q)  及解析梯度.

    ∂f/∂q_k = -Σ_i P_i Δ_ik / port_R_i
    """
    port_R = R0 + Delta @ q
    if np.any(port_R <= 1e-14):
        # 大惩罚 + 指向可行的伪梯度
        return 1e10, np.zeros_like(q)
    f = -float(np.sum(P * np.log(port_R)))
    g = -(Delta.T @ (P / port_R))
    return f, g


def _project_box_sum(q, lo, hi, sum_max):
    """投影到 [lo,hi]^N 且 Σq ≤ sum_max."""
    q = np.clip(q, lo, hi)
    s = float(q.sum())
    if s > sum_max:
        # 等比收缩到可行
        if s > 0:
            q = q * (sum_max / s)
        else:
            q = np.zeros_like(q)
        q = np.clip(q, lo, hi)
    return q


def approx_warmstart_weights(
    probs: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
    t: float = 0.5,
    allow_short: bool = False,
    allow_leverage: bool = False,
    max_multiple: float = 1.0,
) -> List[float]:
    """
    §3.8 思想的多资产初值: 对每个资产单独做抛物线近似 (其它资产=0),
    再投影到约束集, 作为求解器 warm-start.

    单资产: h(q)≈a q²+b q+c, 三点 q=0,t/2,t; q_k≈clip(-b/2a, L, U).
    多资产: 各 q_k 独立近似后若 Σq 超限则等比缩放到 sum_max.
    """
    P = np.asarray(probs, dtype=float)
    A = np.asarray(asset_returns, dtype=float)
    W, N = A.shape
    P = P / P.sum()
    R0 = 1.0 + r0
    Delta = (1.0 + A) - R0
    lo = -max_multiple if allow_short else 0.0
    hi = max_multiple if allow_leverage else 1.0
    sum_max = max_multiple if allow_leverage else 1.0

    q_init = np.zeros(N)
    for k in range(N):
        def H_of(x, k=k):
            q = np.zeros(N)
            q[k] = x
            m = _portfolio_metrics(P, Delta, R0, q)
            return m["H_bits"]

        H0 = H_of(0.0)
        Ht2 = H_of(t / 2.0)
        Ht = H_of(t)
        # 无效三点 → 0
        if not all(np.isfinite([H0, Ht2, Ht])):
            q_init[k] = 0.0
            continue
        a = 2.0 * (Ht - 2.0 * Ht2 + H0) / (t * t)
        b = (4.0 * Ht2 - Ht - 3.0 * H0) / t
        if a == 0:
            q = hi if b > 0 else lo
        else:
            q = -b / (2.0 * a)
        q_init[k] = float(np.clip(q, lo, hi))

    return list(map(float, _project_box_sum(q_init, lo, hi, sum_max)))


def solve_max_entropy_slsqp(
    probs: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
    allow_short: bool = False,
    allow_leverage: bool = False,
    max_multiple: float = 1.0,
    x0: Optional[Sequence[float]] = None,
) -> dict:
    """
    SLSQP 求解最大增值熵 (凸问题, 局部=全局).

    min_q  -Σ_i P_i ln( R0 + Σ_k q_k Δ_ik )
    s.t.   lo ≤ q_k ≤ hi,  Σ q_k ≤ sum_max

    x0 可传 §3.8 近似初值 (warm-start)。
    """
    from scipy.optimize import minimize

    P = np.asarray(probs, dtype=float)
    A = np.asarray(asset_returns, dtype=float)
    W, N = A.shape
    P = P / P.sum()
    R0 = 1.0 + r0
    Delta = (1.0 + A) - R0
    lo = -max_multiple if allow_short else 0.0
    hi = max_multiple if allow_leverage else 1.0
    sum_max = max_multiple if allow_leverage else 1.0

    def fun(q):
        return _neg_H_grad(q, P, Delta, R0, allow_short, allow_leverage, max_multiple)[0]

    def jac(q):
        return _neg_H_grad(q, P, Delta, R0, allow_short, allow_leverage, max_multiple)[1]

    if x0 is None:
        q0 = np.zeros(N)
    else:
        q0 = _project_box_sum(
            np.asarray(x0, dtype=float), lo, hi, sum_max
        )

    cons = [{"type": "ineq", "fun": lambda q: sum_max - float(q.sum())}]
    bounds = [(lo, hi)] * N
    res = minimize(
        fun,
        q0,
        jac=jac,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 300},
    )
    q = np.clip(res.x, lo, hi)
    if q.sum() > sum_max + 1e-9:
        q = _project_box_sum(q, lo, hi, sum_max)
    q0_cash = 1.0 - float(q.sum())
    met = _portfolio_metrics(P, Delta, R0, q)
    return {
        "weights": [q0_cash] + list(map(float, q)),
        "cash": q0_cash,
        "asset_weights": list(map(float, q)),
        "H_bits": met["H_bits"],
        "geometric_return": met["geometric_return"],
        "expected_return": met["expected_return"],
        "scenario_returns": met["scenario_returns"],
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "method": "slsqp",
        "x0": list(map(float, q0)),
    }


def solve_max_entropy_warmstart(
    probs: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
    allow_short: bool = False,
    allow_leverage: bool = False,
    max_multiple: float = 1.0,
    t: float = 0.5,
) -> dict:
    """
    §3.8 近似初值 + SLSQP 精修 (大规模推荐路径).

    步骤:
      1) 各资产抛物线近似得 q_init
      2) 以 q_init 为 x0 跑 SLSQP
    """
    q_init = approx_warmstart_weights(
        probs,
        asset_returns,
        r0=r0,
        t=t,
        allow_short=allow_short,
        allow_leverage=allow_leverage,
        max_multiple=max_multiple,
    )
    res = solve_max_entropy_slsqp(
        probs,
        asset_returns,
        r0=r0,
        allow_short=allow_short,
        allow_leverage=allow_leverage,
        max_multiple=max_multiple,
        x0=q_init,
    )
    res["method"] = "approx_warmstart_slsqp"
    res["warmstart"] = q_init
    return res


def compare_solvers(
    probs: Sequence[float],
    asset_returns: Sequence[Sequence[float]],
    r0: float = 0.0,
    allow_short: bool = False,
    allow_leverage: bool = False,
    max_multiple: float = 1.0,
) -> dict:
    """
    三种方法对比: 爬山 / SLSQP / 近似初值+SLSQP.

    返回各方法结果 + H 最大者名次.
    """
    hill = optimize_portfolio_entropy(
        probs,
        asset_returns,
        r0=r0,
        allow_short=allow_short,
        allow_leverage=allow_leverage,
        max_multiple=max_multiple,
    )
    hill["method"] = "hill_climb"
    sls = solve_max_entropy_slsqp(
        probs, asset_returns, r0, allow_short, allow_leverage, max_multiple
    )
    warm = solve_max_entropy_warmstart(
        probs, asset_returns, r0, allow_short, allow_leverage, max_multiple
    )
    rows = {
        "hill_climb": hill,
        "slsqp": sls,
        "approx_warmstart_slsqp": warm,
    }
    # 排名按 H_bits
    ranked = sorted(
        rows.items(),
        key=lambda kv: (kv[1].get("H_bits", float("-inf"))),
        reverse=True,
    )
    return {
        "methods": rows,
        "ranking": [k for k, _ in ranked],
        "best": ranked[0][0],
        "H_bits": {k: v.get("H_bits") for k, v in rows.items()},
        "weights": {k: v.get("weights") for k, v in rows.items()},
    }


# ---------------------------------------------------------------------------
# 分资产边际分布 → 联合情景 (独立假设)
# ---------------------------------------------------------------------------

def expand_joint_from_marginals(
    marginals: Sequence[Sequence[Tuple[float, float]]],
) -> Tuple[List[float], List[List[float]]]:
    """
    将各资产边际分布展开为联合情景矩阵 (独立假设).

    Parameters
    ----------
    marginals:
        每个元素是该资产的 [(概率, 收益率), ...], 各资产条数可以不同.
        例如资产 A 4 种、B 5 种.

    Returns
    -------
    probs, grid
        probs[i] 为联合情景概率 Π p_k(i_k)
        grid[i][k] 为第 k 种资产在该情景的收益率
    """
    from itertools import product

    if not marginals:
        raise ValueError("至少需要一个资产")
    dists: List[List[Tuple[float, float]]] = []
    for k, d in enumerate(marginals):
        pairs = [(float(p), float(r)) for p, r in d]
        s = sum(p for p, _ in pairs)
        if s <= 0:
            raise ValueError(f"资产{k}概率和必须 > 0")
        if abs(s - 1.0) > 1e-9:
            pairs = [(p / s, r) for p, r in pairs]
        dists.append(pairs)

    counts = [len(d) for d in dists]
    total = 1
    for c in counts:
        total *= c
    if total > 5000:
        raise ValueError(f"联合情景数 {total} 过大 (>5000)")

    probs: List[float] = []
    grid: List[List[float]] = []
    for idx in product(*[range(c) for c in counts]):
        p = 1.0
        row: List[float] = []
        for k, i in enumerate(idx):
            pk, rk = dists[k][i]
            p *= pk
            row.append(rk)
        probs.append(p)
        grid.append(row)
    return probs, grid


# ---------------------------------------------------------------------------
# §3.11 简易风险对比示例: 相同 E,σ 不同最优仓位
# ---------------------------------------------------------------------------

def compare_same_moments() -> List[dict]:
    """
    书中图2.5例:
      证券I: {0.25|0, 0.75|1}  当前价1 → 未来0或2 → r = -1 或 +1? 
      书中: 价格 0 和 2, 概率 1/4 和 3/4 → R=0 或 2 → r=-1 或 +1, E=0.5, σ=0.866
      证券II: 对称反转以 0.5 为中心: r 分布使 E=0.5, σ相同
    这里按书中表2.7口径:
      I: r ∈ {-1, 1} with p 0.25/0.75 → E=0.5, σ=0.866, R_g(1)=(0)^{0.25}... 亏光
      实际书中 R_g(q=1)=0 对 I。
    简化复现:
    """
    rows = []
    # 证券 I: 25% 亏光, 75% 赚100% (价 0→0, 2→ 假设成本1 则 R=0,2)
    probs = [0.25, 0.75]
    rets_I = [-1.0, 1.0]
    res_I = optimal_position_single(probs, rets_I, r0=0.0)
    rows.append({
        "name": "证券I",
        "E": expected_return(probs, rets_I),
        "sigma": std_dev(probs, rets_I),
        "q_star": res_I.q_star,
        "r_g_opt": res_I.r_g,
        "H_opt": res_I.H,
    })
    # 证券 II: 书中 σ 相同、E 相同但分布不对称反转 — 按表 2.7: q*≥100%, r_g≥1.32
    # 近似: {1/3|-0.5?, ...} 用能得 E=0.5 且无亏光的分布
    # 取 r ∈ {0, 0.666...}? 保持 E=0.5, σ=0.866 → var=0.75
    # 解: 两状态 p, r1, r2。 用书中描述"以0.5为中心对称反转"
    # I: r=-1 (p=0.25), r=1 (p=0.75)  → 中心 E=0.5
    # II 反转: 概率质量换到另一侧: p(-1)=0.75, p(1)=0.25 → E=-0.5 不对
    # 按表2.7 数值直接引用说明即可
    return rows


# ---------------------------------------------------------------------------
# 书中例题自测
# ---------------------------------------------------------------------------

def _approx(a: float, b: float, tol: float = 5e-3) -> bool:
    return abs(a - b) <= tol


def run_book_examples() -> dict:
    """验证书中关键例题."""
    checks = []

    # 例1 §2.3/§3.2: 掷硬币 {-1,2} 等概率 → 0.25
    r1 = optimal_position_single([0.5, 0.5], [-1.0, 2.0], r0=0.0)
    checks.append(("掷硬币 q*=0.25", _approx(r1.q_star, 0.25), r1.q_star))

    # 简化公式
    qs = optimal_position_simple([0.5, 0.5], [-1.0, 2.0])
    checks.append(("简化公式 q*=0.25", _approx(qs, 0.25), qs))

    # 例2 §4.5.2: 股票国债 {0.5|-0.3, 0.5|0.8}, r0=0.1 → 0.59
    r2 = optimal_position_single([0.5, 0.5], [-0.3, 0.8], r0=0.1)
    checks.append(("股票/国债 q*=0.59", _approx(r2.q_star, 0.59, 0.01), r2.q_star))

    # 例3 §5.4.1: {-0.1, 0.3} → 3.33
    r3 = optimal_position_single([0.5, 0.5], [-0.1, 0.3], r0=0.0, allow_leverage=True, max_multiple=10)
    checks.append(("期货小波动 q'≈3.33", _approx(r3.q_raw, 3.33, 0.02), r3.q_raw))

    # {-1, 3} → 0.33
    r4 = optimal_position_single([0.5, 0.5], [-1.0, 3.0], r0=0.0)
    checks.append(("期货大波动 q*≈0.33", _approx(r4.q_star, 0.33, 0.01), r4.q_star))

    # 等概率简化
    qe = optimal_position_equal_prob(-1.0, 3.0, fee=0.0)
    checks.append(("等概率简化 q*=1/3", _approx(qe, 1.0 / 3.0, 0.01), qe))

    # 例4 §6.2 期权 {0.7|-1, 0.3|3} → 0.067
    r5 = optimal_position_single([0.7, 0.3], [-1.0, 3.0], r0=0.0)
    checks.append(("期权 q*≈0.067", _approx(r5.q_star, 0.067, 0.005), r5.q_star))

    # 例5 期权发行 Δ={-3,1}, P={0.2,0.8} → 0.067
    r6 = optimal_position_single([0.2, 0.8], [-3.0, 1.0], r0=0.0)
    checks.append(("期权发行 q*≈0.067", _approx(r6.q_star, 0.067, 0.005), r6.q_star))

    # §2.3 {0.5|-0.5, 0.5|1.5} → 2/3
    r7 = optimal_position_single([0.5, 0.5], [-0.5, 1.5], r0=0.0)
    checks.append(("小盈亏 q*=2/3", _approx(r7.q_star, 2.0 / 3.0, 0.01), r7.q_star))

    # 骰子 {1/3|-1, 2/3|1} → 1/3
    r8 = optimal_position_single([1.0 / 3.0, 2.0 / 3.0], [-1.0, 1.0], r0=0.0)
    checks.append(("骰子 q*=1/3", _approx(r8.q_star, 1.0 / 3.0, 0.01), r8.q_star))

    # 多硬币 N=1,2,3,4 ≈ 0.25, 0.46 total / each 0.23...
    m1 = optimize_multi_coin_equal(1, -1.0, 2.0, 0.5)
    checks.append(("多硬币N=1 q_each≈0.25", _approx(m1["q_each"], 0.25, 0.02), m1["q_each"]))
    m2 = optimize_multi_coin_equal(2, -1.0, 2.0, 0.5)
    checks.append(("多硬币N=2 q_each≈0.23", _approx(m2["q_each"], 0.23, 0.02), m2["q_each"]))

    # 消费熵例 §3.7: q=1, r=0.2, r0=0, K=0.4, r_s=0.15
    # Hc = log(1 - 0.15*0.4 + 0.2) = log(1.14), 比特 = log2(1.14)
    hA = consumer_growth_entropy([1.0], [0.2], q=1.0, r0=0.0, consume_ratio=0.4, r_sacrifice=0.15)
    expected_hA = math.log(1.14) / math.log(2)
    checks.append(("消费熵A≈log2(1.14)", _approx(hA, expected_hA, 0.01), hA))
    # 基金B: {-0.28, 1} 等概率 → 书中 Hc=log(1.13) 近似
    hB = consumer_growth_entropy(
        [0.5, 0.5], [-0.28, 1.0], q=1.0, r0=0.0, consume_ratio=0.4, r_sacrifice=0.15
    )
    # 0.5*log(1-0.06-0.28)+0.5*log(1-0.06+1)=0.5*log(0.66*1.94)→log(1.14?) 书中 1.13
    checks.append(("消费熵B≈0.5*log2(0.66*1.94)", _approx(hB, 0.5 * math.log(0.66 * 1.94) / math.log(2), 0.02), hB))

    # 抛物线近似例题 (P=1/4,1/2,1/4; Δ=-1,0.5,2; r0=0) → q*≈0.46
    # 三状态收益与 Δ 相同 (r0=0)
    par = optimal_position_parabolic_approx(
        [0.25, 0.5, 0.25],
        [-1.0, 0.5, 2.0],
        r0=0.0,
        t=0.5,
    )
    # 书中表3.2: t=0.5 时近似 q*≈0.464, 精确 q*≈0.4609
    exact_q = par["H_at_exact_search"][1]
    checks.append(("三状态精确 q*≈0.46", _approx(exact_q, 0.4609, 0.02), exact_q))
    checks.append(("抛物线近似 q*≈0.46", _approx(par["q_approx"], 0.464, 0.05), par["q_approx"]))

    # 亏光概率上限
    # Δ1=-1, Δ2 很大, P1=0.5 → q* < 0.5
    r9 = optimal_position_single([0.5, 0.5], [-1.0, 100.0], r0=0.0)
    checks.append(("亏光P=0.5 → q*<0.5", r9.q_star < 0.5, r9.q_star))

    # 增值熵: 最优 q 的 H >= 满仓 H (期权例; 满仓会亏光故 H=-inf)
    H_opt = r5.H
    H_full = growth_entropy([0.7, 0.3], [-1.0, 3.0], base=2.0)
    checks.append(("期权 H(q*)>H(1)", H_opt > H_full, (H_opt, H_full)))
    checks.append(("期权 H(q*)有限", H_opt > float("-inf"), H_opt))

    # 多资产: 反相关改善
    # 两资产四情景独立硬币
    # 情景: LL, LW, WL, WW
    probs4 = [0.25, 0.25, 0.25, 0.25]
    # 资产A, B 均为 {-1, 2} 独立
    # 实际用单边两情景相关简化: 用 optimize on 2 assets 2 scenarios 不够
    # 构造 4 情景
    # A: L→-1, W→2; B 独立同
    asset4 = [
        [-1, -1],
        [-1, 2],
        [2, -1],
        [2, 2],
    ]
    opt4 = optimize_portfolio_entropy(probs4, asset4, r0=0.0)
    # 各约 23%
    w = opt4["asset_weights"]
    checks.append(("2独立资产各≈0.23", all(_approx(x, 0.23, 0.05) for x in w), w))
    checks.append(("2资产 H>单资产全仓", opt4["H_bits"] > H_full, opt4["H_bits"]))

    # 增量优化 q=0 时接近无手续费最优(扣除手续费后略小)
    inc = optimal_position_increment(
        0.0, [0.5, 0.5], [-1.0, 2.0], r0=0.0, fee=0.01, q_min=0, q_max=1
    )
    checks.append(("增量优化 q_new≤0.25+ε", inc.q_new <= 0.25 + 0.05, inc.q_new))

    passed = sum(1 for _, ok, _ in checks if ok)
    return {
        "passed": passed,
        "total": len(checks),
        "checks": [{"name": n, "ok": bool(ok), "value": v} for n, ok, v in checks],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("《投资组合的熵理论和信息价值》仓位公式自测")
    print("=" * 60)
    result = run_book_examples()
    for c in result["checks"]:
        mark = "✓" if c["ok"] else "✗"
        print(f"  [{mark}] {c['name']}: value={c['value']}")
    print(f"\n通过 {result['passed']}/{result['total']}")

    # 演示输出
    print("\n--- 演示: 期权 {0.7|-1, 0.3|3} ---")
    r = optimal_position_single([0.7, 0.3], [-1.0, 3.0], r0=0.0)
    print(r)

    print("\n--- 演示: 透支卖空 P=0.5, r={-0.1,0.1}? 用书中 r0=0.1,r_loan=0.15, r={-0.3,0.8} ---")
    r = optimal_position_single([0.5, 0.5], [-0.3, 0.8], r0=0.1, allow_leverage=True, max_multiple=2.0, r_loan=0.15)
    print(r)

    print("\n--- 演示: 分散投资极限 ---")
    for row in diversification_limit_check():
        print(
            f"  N={row['n_assets']:3d}  q_each={row['q_each']:.4f}  "
            f"r_g={row['geometric_return']:.4f}  E={row['E']:.4f}"
        )

    print("\n--- 演示: 三状态数值优化 ---")
    opt = optimize_portfolio_entropy(
        [0.25, 0.5, 0.25],
        [[-1.0], [0.5], [2.0]],
        r0=0.0,
    )
    print(f"  weights={opt['weights']}, H={opt['H_bits']:.4f} bit, r_g={opt['geometric_return']:.4f}")


if __name__ == "__main__":
    main()
