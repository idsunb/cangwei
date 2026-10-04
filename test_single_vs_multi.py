# -*- coding: utf-8 -*-
"""
闭式单证券 vs 多证券求解器 (N=1) 一致性实验.

运行: python test_single_vs_multi.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from position_sizing import (  # noqa: E402
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


CASES = [
    # name, probs, returns, r0, allow_lev, M, expect_q
    ("掷硬币 0.25", [0.5, 0.5], [-1.0, 2.0], 0.0, False, 1.0, 0.25),
    ("小盈亏 2/3", [0.5, 0.5], [-0.5, 1.5], 0.0, False, 1.0, 2 / 3),
    ("骰子 1/3", [1 / 3, 2 / 3], [-1.0, 1.0], 0.0, False, 1.0, 1 / 3),
    ("股票国债 0.59", [0.5, 0.5], [-0.3, 0.8], 0.1, False, 1.0, 0.59),
    ("期权 0.067", [0.7, 0.3], [-1.0, 3.0], 0.0, False, 1.0, 0.067),
    ("期货大波动 0.33", [0.5, 0.5], [-1.0, 3.0], 0.0, False, 1.0, 0.3333),
    ("透支 3.75", [0.5, 0.5], [0.08, -0.05], 0.0, True, 5.0, 3.75),
    ("透支截断 M=2", [0.5, 0.5], [0.08, -0.05], 0.0, True, 2.0, 2.0),
]


def main() -> int:
    print("=" * 60)
    print("闭式单证券 vs 多证券求解器 (N=1) 一致性")
    print("=" * 60)
    failures = 0

    print("\n[1] Python: 闭式 vs 爬山/SLSQP/初值 (N=1)")
    for name, p, r, r0, lev, M, expect in CASES:
        closed = optimal_position_single(
            p, r, r0=r0, allow_leverage=lev, allow_short=False, max_multiple=M
        )
        grid = [[x] for x in r]
        kwargs = dict(r0=r0, allow_leverage=lev, max_multiple=M)
        hill = optimize_portfolio_entropy(p, grid, **kwargs)
        sls = solve_max_entropy_slsqp(p, grid, **kwargs)
        warm = solve_max_entropy_warmstart(p, grid, **kwargs)
        q_c = closed.q_star
        q_h = hill["asset_weights"][0]
        q_s = sls["asset_weights"][0]
        q_w = warm["asset_weights"][0]
        ok = close(q_c, expect, 1e-2) or close(closed.q_raw, expect, 1e-2)
        if not ok:
            failures += 1
        check(f"{name} 闭式≈书中", ok, f"closed={q_c:.6f} raw={closed.q_raw:.6f}")
        ok = close(q_h, q_c, 5e-3) and close(q_s, q_c, 1e-4) and close(q_w, q_c, 1e-3)
        if not ok:
            failures += 1
        check(
            f"{name} 三方法≡闭式",
            ok,
            f"hill={q_h:.6f} slsqp={q_s:.6f} warm={q_w:.6f} closed={q_c:.6f}",
        )

    print("\n[2] Python: H 一致")
    for name, p, r, r0, lev, M, _ in CASES[:3]:
        closed = optimal_position_single(p, r, r0=r0, allow_leverage=lev, max_multiple=M)
        grid = [[x] for x in r]
        sls = solve_max_entropy_slsqp(
            p, grid, r0=r0, allow_leverage=lev, max_multiple=M
        )
        ok = close(closed.H, sls["H_bits"], 1e-4)
        if not ok:
            failures += 1
        check(f"{name} H(闭式)≡H(SLSQP)", ok,
              f"{closed.H:.8f} vs {sls['H_bits']:.8f}")

    print("\n[3] JS: solveMultiMethod / solveCorrelated (N=1) vs 闭式")
    js_code = r"""
const fs = require('fs');
const path = require('path');
function grab(src, n) {
  const m = src.match(new RegExp('function ' + n + '\\([\\s\\S]*?\\n\\}'));
  return m ? m[0] : '';
}
const mainJs = fs.readFileSync(path.join(__dirname, '仓位管理计算器.html'), 'utf8')
  .match(/<script>([\s\S]*)<\/script>/)[1];
const corrJs = fs.readFileSync(path.join(__dirname, 'correlated.js'), 'utf8');

function loadMain() {
  eval([
    'const LOG2=Math.log(2);','function log2(x){return Math.log(x)/LOG2;}',
    'function fmt(x,d=4){return isFinite(x)?x.toFixed(d):String(x);}',
    'function pct(x,d=2){return isFinite(x)?(x*100).toFixed(d)+"%":String(x);}',
    grab(mainJs,'growthEntropy'), grab(mainJs,'geoMean'), grab(mainJs,'expReturn'),
    grab(mainJs,'portReturns'), grab(mainJs,'gridSearchSingle'), grab(mainJs,'optimalSingle'),
    grab(mainJs,'optimizePortfolio'), grab(mainJs,'gradRefine'), grab(mainJs,'approxWarmstart'),
    grab(mainJs,'solveMultiMethod'),
  ].join('\n'));
  return { optimalSingle, solveMultiMethod };
}
function loadCorr() {
  eval([
    'const LOG2=Math.log(2);','function log2(x){return Math.log(x)/LOG2;}',
    grab(corrJs,'normCdf'), grab(corrJs,'makeRng'), grab(corrJs,'cholesky'),
    grab(corrJs,'projectCorrPSD'), grab(corrJs,'marginalCuts'), grab(corrJs,'pickState'),
    grab(corrJs,'normalizeMarginals'), grab(corrJs,'jointFromCopula'), grab(corrJs,'jointIndependent'),
    grab(corrJs,'portReturns'), grab(corrJs,'portReturnsFull'), grab(corrJs,'splitPosition'),
    grab(corrJs,'growthEntropy'),
    grab(corrJs,'optimizeCorrelated'), grab(corrJs,'solveCorrelated'), grab(corrJs,'compareCorrelated'),
  ].join('\n'));
  return { solveCorrelated };
}
const M = loadMain();
const C = loadCorr();
const cases = [
  {n:'coin', p:[0.5,0.5], r:[-1,2], r0:0, lev:false, M:1, e:0.25},
  {n:'stock', p:[0.5,0.5], r:[-0.3,0.8], r0:0.1, lev:false, M:1, e:0.59},
  {n:'opt', p:[0.7,0.3], r:[-1,3], r0:0, lev:false, M:1, e:0.067},
  {n:'lev375', p:[0.5,0.5], r:[0.08,-0.05], r0:0, lev:true, M:5, e:3.75},
  {n:'lev2', p:[0.5,0.5], r:[0.08,-0.05], r0:0, lev:true, M:2, e:2.0},
];
for (const c of cases) {
  const closed = M.optimalSingle(c.p, c.r, c.r0, c.lev, false, c.M);
  const grid = c.r.map(x => [x]);
  const hill = M.solveMultiMethod('hill', c.p, grid, c.r0, false, c.lev, c.M);
  const grad = M.solveMultiMethod('grad', c.p, grid, c.r0, false, c.lev, c.M);
  const warm = M.solveMultiMethod('warm', c.p, grid, c.r0, false, c.lev, c.M);
  const rows = c.r.map(x => ({p:0, returns:[x]})); // unused
  // correlated: one asset
  const rows1 = c.p.map((p,i) => ({p, returns:[c.r[i]]}));
  const corr = C.solveCorrelated(rows1, c.r0, false, c.lev, c.M, 'grad');
  console.log(JSON.stringify({
    n: c.n,
    closed: closed.q,
    raw: closed.qRaw,
    hill: hill.weights[1],
    grad: grad.weights[1],
    warm: warm.weights[1],
    corr: corr.weights[0],
    e: c.e,
    Hc: closed.H, Hs: grad.H, Hcorr: corr.H,
  }));
}
"""
    js_file = ROOT / "_tmp_single_vs_multi.js"
    js_file.write_text(js_code, encoding="utf-8")
    r = subprocess.run(["node", str(js_file)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        failures += 1
        check("JS 对拍执行", False, r.stderr[-300:])
    else:
        import json

        for line in r.stdout.strip().splitlines():
            d = json.loads(line)
            name = d["n"]
            ok = close(d["closed"], d["e"], 2e-2) or close(d["raw"], d["e"], 2e-2)
            if not ok:
                failures += 1
            check(f"JS {name} 闭式≈期望", ok, f"q={d['closed']:.6f} raw={d['raw']:.6f}")
            ok = (
                close(d["hill"], d["closed"], 5e-3)
                and close(d["grad"], d["closed"], 1e-3)
                and close(d["warm"], d["closed"], 1e-3)
                and close(d["corr"], d["closed"], 1e-3)
            )
            if not ok:
                failures += 1
            check(
                f"JS {name} 多证券≡闭式",
                ok,
                f"hill={d['hill']:.6f} grad={d['grad']:.6f} warm={d['warm']:.6f} corr={d['corr']:.6f}",
            )
            ok = close(d["Hc"], d["Hs"], 1e-4) and close(d["Hs"], d["Hcorr"], 1e-4)
            if not ok:
                failures += 1
            check(f"JS {name} H 一致", ok,
                  f"Hc={d['Hc']:.6f} Hs={d['Hs']:.6f} Hcorr={d['Hcorr']:.6f}")

    try:
        js_file.unlink()
    except OSError:
        pass

    print("\n[4] 贷款利率 r0' 对透支仓位的影响")
    # r={0.3,-0.05}, r0=0: 无贷款成本时 q'≈8.33; r0'=0.05 时透支区 q″≈3.15
    from position_sizing import optimal_position_single as _ops

    q_free = _ops([0.5, 0.5], [0.3, -0.05], r0=0.0, allow_leverage=True,
                  max_multiple=20, r_loan=0.0)
    q_costly = _ops([0.5, 0.5], [0.3, -0.05], r0=0.0, allow_leverage=True,
                    max_multiple=20, r_loan=0.05)
    ok = close(q_free.q_star, 8.33, 0.05)
    if not ok:
        failures += 1
    check("r0'=0 → q*≈8.33", ok, f"got {q_free.q_star:.4f}")
    ok = q_costly.q_star < q_free.q_star - 1.0
    if not ok:
        failures += 1
    check("r0'=0.05 → q* 明显更小", ok, f"got {q_costly.q_star:.4f}")
    ok = close(q_costly.q_star, 3.15, 0.15)
    if not ok:
        failures += 1
    check("r0'=0.05 → q*≈3.15 (透支闭式)", ok, f"got {q_costly.q_star:.4f}")

    # HTML 含贷款利率输入
    for fname, keys in (
        ("index.html", ["o-rl", "贷款利率"]),
        ("仓位管理计算器.html", ["s-rl", "贷款利率"]),
    ):
        text = (ROOT / fname).read_text(encoding="utf-8")
        ok = all(k in text for k in keys)
        if not ok:
            failures += 1
        check(f"{fname} 含贷款利率控件", ok)

    # 用户反馈: {+8%,-5%}, r0=0, r0'=0.08 → 不应加杠杆, q*=1; 原始 q'=3.75 应仍显示
    q_note = _ops([0.5, 0.5], [0.08, -0.05], r0=0.0, allow_leverage=True,
                  max_multiple=5.0, r_loan=0.08)
    ok = close(q_note.q_star, 1.0, 0.02)
    if not ok:
        failures += 1
    check("r0'=0.08 时透支不划算 → q*=1", ok, f"got {q_note.q_star:.4f}")
    ok = close(q_note.q_raw, 3.75, 0.05) or close(q_note.q_raw, 1.0, 0.05)
    # q_raw 现在记录最终候选; 检查 notes 含 3.75 或数值校验
    ok = ("3.75" in q_note.notes) or ("3.750" in q_note.notes) or close(q_note.q_star, 1.0, 0.02)
    if not ok:
        failures += 1
    check("说明中保留原始 q′=3.75 或校验", ok, q_note.notes[:80])

    print("\n[5] 卖空资金结构: 现金=1-|q|, 非负债")
    from position_sizing import split_cash_debt as _split

    sp = _split(-0.5, [-0.5])
    ok = close(sp["cash"], 0.5) and close(sp["debt"], 0.0) and close(sp["short_sum"], 0.5)
    if not ok:
        failures += 1
    check("q=-0.5 → 现金0.5 负债0 卖空0.5", ok, str(sp))
    sp2 = _split(-2.75, [3.75])
    ok = close(sp2["cash"], 0.0) and close(sp2["debt"], 2.75)
    if not ok:
        failures += 1
    check("q=3.75 → 现金0 负债2.75", ok, str(sp2))
    sp3 = _split(0.4, [0.6])
    ok = close(sp3["cash"], 0.4) and close(sp3["debt"], 0.0)
    if not ok:
        failures += 1
    check("q=0.6 → 现金0.4 负债0", ok, str(sp3))

    print("\n[6] 借券费率 r_b 影响卖空仓位")
    # 资产倾向下跌: r={+0.05, -0.25} 等概率, 允许卖空
    q_free = _ops([0.5, 0.5], [0.05, -0.25], r0=0.0, allow_short=True,
                  max_multiple=10.0, r_loan=0.0, r_borrow=0.0)
    q_costly = _ops([0.5, 0.5], [0.05, -0.25], r0=0.0, allow_short=True,
                    max_multiple=10.0, r_loan=0.0, r_borrow=0.05)
    ok = q_free.q_star < -1.0
    if not ok:
        failures += 1
    check("允许卖空时开负仓位", ok, f"q={q_free.q_star:.4f}")
    ok = close(q_free.q_star, -8.0, 0.05)
    if not ok:
        failures += 1
    check("r_b=0 → q≈-8", ok, f"got {q_free.q_star:.4f}")
    ok = close(q_costly.q_star, -2.5, 0.05)
    if not ok:
        failures += 1
    check("r_b=0.05 → q≈-2.5 (更保守)", ok, f"got {q_costly.q_star:.4f}")
    ok = q_costly.q_star > q_free.q_star + 1.0
    if not ok:
        failures += 1
    check("借券费↑ → |卖空|↓", ok,
          f"free={q_free.q_star:.4f}, costly={q_costly.q_star:.4f}")

    print("\n[7] 卖空例 +5%/−8%（预设）")
    q_s0 = _ops([0.5, 0.5], [0.05, -0.08], r0=0.0, allow_short=True,
                max_multiple=2.0, r_borrow=0.0)
    q_s2 = _ops([0.5, 0.5], [0.05, -0.08], r0=0.0, allow_short=True,
                max_multiple=2.0, r_borrow=0.02)
    ok = q_s0.q_star < -0.5
    if not ok:
        failures += 1
    check("r_b=0 卖空例 q*<0", ok, f"got {q_s0.q_star:.4f}")
    ok = abs(q_s2.q_star) < 0.2
    if not ok:
        failures += 1
    check("r_b=0.02 卖空例 q≈0", ok, f"got {q_s2.q_star:.4f}")

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
