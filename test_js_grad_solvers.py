# -*- coding: utf-8 -*-
"""
网页多资产求解器 JS 运行时测试.

回归: gradRefine 曾因 for(let it=...) 循环变量作用域在 return 处
抛出 "it is not defined"。本测试实际执行 JS 的
梯度精修 / §3.8 初值+梯度 路径, 确保不抛错且 H 有限.

运行:
  python test_js_grad_solvers.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HTML = ROOT / "仓位管理计算器.html"


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "✓" if ok else "✗"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def grab(js: str, fn_name: str) -> str:
    m = re.search(rf"function {fn_name}\([\s\S]*?\n\}}", js)
    return m.group(0) if m else ""


def main() -> int:
    print("=" * 60)
    print("网页 JS 梯度/近似初值求解器 运行时测试")
    print("=" * 60)
    failures = 0

    html = HTML.read_text(encoding="utf-8")
    m = re.search(r"<script>([\s\S]*)</script>", html)
    if not m:
        print("  [✗] 未找到 <script>")
        return 1
    js = m.group(1)

    print("\n[1] 静态: 不再在 return 处引用块级循环变量")
    # for (let it ... ) 之后 return ... it  的模式应消失
    bad = re.search(r"for \(let it =[\s\S]*?return \{[^}]*\bit\b", js)
    ok = bad is None
    if not ok:
        failures += 1
    check("无 for(let it)→return it 作用域错误", ok)

    print("\n[2] 运行时: solveMultiMethod 三条路径")
    driver = "\n".join(
        [
            "const LOG2 = Math.log(2);",
            "function log2(x) { return Math.log(x) / LOG2; }",
            grab(js, "growthEntropy"),
            grab(js, "geoMean"),
            grab(js, "expReturn"),
            grab(js, "optimizePortfolio"),
            grab(js, "gradRefine"),
            grab(js, "approxWarmstart"),
            grab(js, "solveMultiMethod"),
            r"""
const probs = [0.25, 0.25, 0.25, 0.25];
const grid = [[-1, -1], [-1, 2], [2, -1], [2, 2]];
function run(m) {
  const r = solveMultiMethod(m, probs, grid, 0, false, false);
  return { m, H: r.H, w: r.weights, method: r.method, nit: r.nit };
}
const out = {};
for (const key of ["hill", "grad", "warm"]) {
  try {
    out[key] = run(key);
    out[key].error = null;
  } catch (e) {
    out[key] = { error: String(e) };
  }
}
console.log(JSON.stringify(out));
""",
        ]
    )
    node = subprocess.run(
        ["node", "-e", driver],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if node.returncode != 0:
        failures += 1
        check("JS 驱动执行", False, node.stderr[-300:])
    else:
        import json

        raw = node.stdout.strip().splitlines()[-1] if node.stdout.strip() else "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
            failures += 1
            check("解析 JS 输出", False, raw[:200])

        for key, label in [
            ("hill", "坐标爬山"),
            ("grad", "解析梯度"),
            ("warm", "§3.8初值+梯度"),
        ]:
            d = data.get(key, {})
            err = d.get("error")
            ok = err is None
            if not ok:
                failures += 1
            check(f"{label} 不抛错", ok, err or "")
            H = d.get("H")
            ok = H is not None and H == H and abs(H) != float("inf")
            if not ok:
                failures += 1
            check(f"{label} H 有限", ok, f"H={H}")
            w = d.get("w") or []
            ok = len(w) >= 1 and abs(sum(w) - 1.0) < 1e-6
            if not ok:
                failures += 1
            check(f"{label} Σq=1", ok, f"w={w}")

        # 三方法 H 应接近 (凸问题)
        if all(data.get(k, {}).get("H") is not None for k in ("hill", "grad", "warm")):
            Hs = [data[k]["H"] for k in ("hill", "grad", "warm")]
            ok = max(Hs) - min(Hs) < 5e-3
            if not ok:
                failures += 1
            check("三方法 H 接近", ok, f"Hs={Hs}")
            # grad/warm 不应显著差于 hill
            ok = data["grad"]["H"] >= data["hill"]["H"] - 5e-3
            if not ok:
                failures += 1
            check("grad ≥ hill−5e-3", ok, f"grad={data['grad']['H']}, hill={data['hill']['H']}")

    print("\n[3] 特例: 书中两硬币 H≈0.1624 bit")
    driver2 = "\n".join(
        [
            "const LOG2 = Math.log(2);",
            "function log2(x) { return Math.log(x) / LOG2; }",
            grab(js, "growthEntropy"),
            grab(js, "optimizePortfolio"),
            grab(js, "gradRefine"),
            grab(js, "approxWarmstart"),
            grab(js, "solveMultiMethod"),
            r"""
const r = solveMultiMethod("warm", [0.25,0.25,0.25,0.25],
  [[-1,-1],[-1,2],[2,-1],[2,2]], 0, false, false);
console.log("H " + r.H.toFixed(6));
console.log("method " + r.method);
""",
        ]
    )
    n2 = subprocess.run(
        ["node", "-e", driver2], capture_output=True, text=True, encoding="utf-8"
    )
    if n2.returncode != 0:
        failures += 1
        check("warm 书例", False, n2.stderr[-200:])
    else:
        out = {ln.split()[0]: ln.split()[1] for ln in n2.stdout.splitlines() if " " in ln}
        H = float(out.get("H", "nan"))
        ok = abs(H - 0.162364) < 5e-3
        if not ok:
            failures += 1
        check("warm H≈0.1624", ok, f"H={H}")

    print("\n" + "=" * 60)
    print(f"失败 {failures} 项" if failures else "全部通过")
    print("=" * 60)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
