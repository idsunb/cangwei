/**
 * JS 相关联合分布自检 (由 test_correlated_joint.py 或单独 node 运行)
 */
const fs = require("fs");
const path = require("path");
const src = fs.readFileSync(path.join(__dirname, "correlated.js"), "utf8");

function grab(name) {
  const m = src.match(new RegExp("function " + name + "\\([\\s\\S]*?\\n\\}"));
  return m ? m[0] : "";
}

const code = [
  "const LOG2=Math.log(2); const log2=x=>Math.log(x)/LOG2;",
  grab("joint2x2"),
  grab("normCdf"),
  grab("makeRng"),
  grab("cholesky"),
  grab("projectCorrPSD"),
  grab("marginalCuts"),
  grab("pickState"),
  grab("normalizeMarginals"),
  grab("jointFromCopula"),
  grab("jointIndependent"),
  grab("portReturns"),
  grab("portReturnsFull"),
  grab("splitPosition"),
  grab("growthEntropy"),
  grab("optimizeCorrelated"),
  grab("solveCorrelated"),
  grab("compareCorrelated"),
  grab("optimalSingleClosed"),
  grab("fillWeightTable"),
  grab("fillResultCards"),
  grab("fillJointTable"),
  grab("joint2x2"),
  "function fmt(x,d=4){return isFinite(x)?x.toFixed(d):String(x);}",
  "function pct(x,d=2){return isFinite(x)?(x*100).toFixed(d)+'%':String(x);}",
].join("\n");

eval(code);

function close(a, b, t = 1e-4) {
  return Math.abs(a - b) <= t;
}

let fail = 0;
function check(name, ok, detail) {
  console.log((ok ? "  [✓] " : "  [✗] ") + name + (detail ? " — " + detail : ""));
  if (!ok) fail++;
}

console.log("JS 相关联合分布");
const ind = joint2x2(0.5, 0.5, 0, -0.1, 0.3, -0.15, 0.4).map((r) => r.p);
check("ρ=0 各0.25", ind.every((p) => close(p, 0.25)), ind.join(","));
const r1 = joint2x2(0.5, 0.5, 1, -0.1, 0.3, -0.15, 0.4);
const p1 = Object.fromEntries(r1.map((r) => [r.label, r.p]));
check("ρ=1 同涨同跌", close(p1["A涨·B涨"], 0.5) && close(p1["A跌·B跌"], 0.5), JSON.stringify(p1));
const rm = joint2x2(0.5, 0.5, -1, -0.1, 0.3, -0.15, 0.4);
const pm = Object.fromEntries(rm.map((r) => [r.label, r.p]));
check("ρ=−1 一涨一跌", close(pm["A涨·B跌"], 0.5) && close(pm["A跌·B涨"], 0.5), JSON.stringify(pm));

function H(rho) {
  const rows = joint2x2(0.5, 0.5, rho, -0.1, 0.3, -0.15, 0.4);
  return optimizeCorrelated(rows, 0, false, false, 1).H;
}
const Hm = H(-0.9), H0 = H(0), Hp = H(0.9);
check("H 负相关>独立>正相关", Hm > H0 && H0 > Hp, `${Hm.toFixed(6)},${H0.toFixed(6)},${Hp.toFixed(6)}`);

const marg = [
  { pHigh: 0.5, rLow: -0.1, rHigh: 0.3 },
  { pHigh: 0.5, rLow: -0.15, rHigh: 0.4 },
  { pHigh: 0.6, rLow: -0.05, rHigh: 0.2 },
];
const corr = [
  [1, 0.5, 0.2],
  [0.5, 1, 0.3],
  [0.2, 0.3, 1],
];
const joint = jointFromCopula(marg, corr, 3000, 42);
const s = joint.reduce((a, b) => a + b.p, 0);
check("copula 概率和≈1", Math.abs(s - 1) < 1e-9, s.toFixed(10));
check("copula 情景数≤8", joint.length <= 8 && joint.length >= 4, String(joint.length));
const o = optimizeCorrelated(joint, 0, false, false, 1);
check("copula 可优化", isFinite(o.H) && isFinite(o.weights[0]), `H=${o.H.toFixed(6)}`);

const Rbad = [
  [1, 0.9, 0.9],
  [0.9, 1, 0.9],
  [0.9, 0.9, 1],
];
const Rp = projectCorrPSD(Rbad, 3);
const L = cholesky(Rp);
check("非正定可投影", !!L, JSON.stringify(Rp[0].map((x) => +x.toFixed(3))));

// 2×3: A 2 态, B 3 态
const m23 = [
  { name: "A", states: [{ p: 0.5, r: -0.1 }, { p: 0.5, r: 0.3 }] },
  {
    name: "B",
    states: [
      { p: 0.3, r: -0.25 },
      { p: 0.4, r: 0.08 },
      { p: 0.3, r: 0.5 },
    ],
  },
];
const j23 = jointFromCopula(m23, [[1, 0.4], [0.4, 1]], 5000, 7);
const s23 = j23.reduce((a, b) => a + b.p, 0);
check("2×3 概率和≈1", Math.abs(s23 - 1) < 1e-9, s23.toFixed(8));
check("2×3 情景数≤6", j23.length <= 6 && j23.length >= 3, String(j23.length));
const pA0 = j23.filter((r) => Math.abs(r.returns[0] + 0.1) < 1e-12).reduce((a, b) => a + b.p, 0);
check("2×3 边际 A0≈0.5", Math.abs(pA0 - 0.5) < 0.08, pA0.toFixed(4));
const o23 = optimizeCorrelated(j23, 0, false, false, 1);
check("2×3 可优化", isFinite(o23.H), `H=${o23.H.toFixed(6)}`);
const i23 = jointIndependent(m23);
check("2×3 独立 6 格", i23.length === 6, String(i23.length));

// 三种求解方法
if (typeof solveCorrelated === "function" && typeof compareCorrelated === "function") {
  const rh = solveCorrelated(j23, 0, false, true, 5, "hill");
  const rg = solveCorrelated(j23, 0, false, true, 5, "grad");
  const rw = solveCorrelated(j23, 0, false, true, 5, "warm");
  check("hill 可跑", isFinite(rh.H), `H=${rh.H.toFixed(6)}`);
  check("grad 可跑", isFinite(rg.H), `H=${rg.H.toFixed(6)}`);
  check("warm 可跑", isFinite(rw.H), `H=${rw.H.toFixed(6)}`);
  const Hs = [rh.H, rg.H, rw.H];
  check("三方法 H 接近", Math.max(...Hs) - Math.min(...Hs) < 5e-3, Hs.map((x) => x.toFixed(6)).join(","));
  const cmp = compareCorrelated(j23, 0, false, true, 5);
  check("compare 三结果", cmp.results.length === 3, cmp.best);
  // ρ=0 时三方法应一致（精确独立）
  const j0 = jointFromCopula(m23, [[1, 0], [0, 1]], 8000, 42);
  const g0 = solveCorrelated(j0, 0, false, true, 5, "grad");
  const i0 = solveCorrelated(jointIndependent(m23), 0, false, true, 5, "grad");
  const dq = Math.max(...g0.weights.map((x, k) => Math.abs(x - i0.weights[k])));
  check("ρ=0 三方法口径下 q* 一致", dq < 1e-9, `Δq=${dq}`);
} else {
  check("solveCorrelated 已导出", false);
}

// 单证券闭式 (index.html)
if (typeof optimalSingleClosed === "function") {
  const s1 = optimalSingleClosed([0.5, 0.5], [-1, 2], 0, false, false, 2);
  check("index 单证券 coin q≈0.25", Math.abs(s1.q - 0.25) < 1e-6, s1.q.toFixed(6));
  const s2 = optimalSingleClosed([0.5, 0.5], [-0.3, 0.8], 0.1, false, false, 2);
  check("index 单证券 stock q≈0.59", Math.abs(s2.q - 0.589286) < 1e-4, s2.q.toFixed(6));
  const s3 = optimalSingleClosed([0.5, 0.5], [0.08, -0.05], 0, true, false, 5);
  check("index 单证券 lev q≈3.75", Math.abs(s3.q - 3.75) < 1e-4, s3.q.toFixed(6));
  const s4 = optimalSingleClosed([0.7, 0.3], [-1, 3], 0, false, false, 2);
  check("index 单证券 opt q≈0.067", Math.abs(s4.q - 0.066667) < 1e-4, s4.q.toFixed(6));
} else {
  check("optimalSingleClosed 已导出", false);
}

if (typeof fillWeightTable === "function") {
  // 模拟 DOM
  const el = { innerHTML: "" };
  fillWeightTable(el, [{ name: "证券 A", q: 0.6154 }, { name: "证券 B", q: 1.3846 }], {
    cashRaw: -1.0,
  });
  const html = el.innerHTML;
  check("fillWeightTable 含项目列", html.indexOf("项目") >= 0);
  check("fillWeightTable 含现金/负债", html.indexOf("现金") >= 0 && html.indexOf("负债") >= 0);
  check("fillWeightTable 含证券A/B", html.indexOf("证券 A") >= 0 && html.indexOf("证券 B") >= 0);
  check("fillWeightTable 含占比", html.indexOf("占比") >= 0);
} else {
  check("fillWeightTable 已导出", false);
}

// 两/多证券 贷款与借券
{
  const rows = [
    { p: 0.5, returns: [0.05, -0.08] },
    { p: 0.5, returns: [0.05, -0.08] },
  ];
  // 其实两情景同 — 用 4 情景两资产
  const rows2 = [
    { p: 0.25, returns: [0.05, 0.05] },
    { p: 0.25, returns: [0.05, -0.08] },
    { p: 0.25, returns: [-0.08, 0.05] },
    { p: 0.25, returns: [-0.08, -0.08] },
  ];
  const a = solveCorrelated(rows2, 0, true, false, 2, "grad", 0, 0);
  const b = solveCorrelated(rows2, 0, true, false, 2, "grad", 0, 0.05);
  check("两证券 r_b 影响卖空", Math.abs(a.weights[0] - b.weights[0]) > 0.01 || Math.abs(a.weights[1] - b.weights[1]) > 0.01,
    JSON.stringify(a.weights) + " vs " + JSON.stringify(b.weights));
  const rowsPos = [
    { p: 0.25, returns: [0.3, 0.3] },
    { p: 0.25, returns: [0.3, -0.1] },
    { p: 0.25, returns: [-0.1, 0.3] },
    { p: 0.25, returns: [-0.1, -0.1] },
  ];
  const c = solveCorrelated(rowsPos, 0, false, true, 5, "grad", 0, 0);
  const d = solveCorrelated(rowsPos, 0, false, true, 5, "grad", 0.2, 0);
  const sumC = c.weights[0] + c.weights[1];
  const sumD = d.weights[0] + d.weights[1];
  check("两证券 r_loan 影响透支", sumC > sumD + 0.01 || Math.abs(c.H - d.H) > 1e-4,
    "sum " + sumC.toFixed(3) + " vs " + sumD.toFixed(3) + " H " + c.H.toFixed(4) + "/" + d.H.toFixed(4));
}

if (typeof fillResultCards === "function") {
  const el = { innerHTML: "" };
  fillResultCards(el, {
    qItems: [{ name: "证券 A", q: 0.6154 }, { name: "证券 B", q: 1.3846 }],
    cash: 0, debt: 1.0, rg: 0.17, ra: 0.2, H: 0.23, capital: 100000,
    extra: [{ k: "标的合计", v: "200.00%" }],
  });
  check("fillResultCards 显示 q*", el.innerHTML.indexOf("q*") >= 0);
  check("fillResultCards 显示证券A/B q", el.innerHTML.indexOf("证券 A") >= 0 && el.innerHTML.indexOf("0.6154") >= 0);
  check("fillResultCards 显示金额", el.innerHTML.indexOf("投入金额") >= 0);
} else {
  check("fillResultCards 已导出", false);
}

if (typeof fillJointTable === "function") {
  const elJ = { innerHTML: "" };
  const rowsJ = joint2x2(0.5, 0.5, 0, -0.1, 0.3, -0.15, 0.4);
  fillJointTable(elJ, rowsJ, 2);
  check("联合表含原始收益列", elJ.innerHTML.indexOf("原始收益") >= 0);
  check("联合表含等权组合收益", elJ.innerHTML.indexOf("等权组合收益") >= 0);
  check("联合表含组合收益(q*)", elJ.innerHTML.indexOf("组合收益(q*)") >= 0);
} else {
  check("fillJointTable 已导出", false);
}

if (typeof fillWeightCompareTable === "function") {
  const elC = { innerHTML: "" };
  fillWeightCompareTable(elC, [
    { name: "A", q: 0.4, qInd: 0.5 },
    { name: "B", q: 0.6, qInd: 0.5 },
  ]);
  check("对照表含 q*（相关）", elC.innerHTML.indexOf("q*（相关）") >= 0);
  check("对照表含 q*（独立）", elC.innerHTML.indexOf("q*（独立）") >= 0);
  check("对照表含占比", elC.innerHTML.indexOf("占比") >= 0);
} else {
  check("fillWeightCompareTable 已导出", false);
}

console.log(fail ? `失败 ${fail} 项` : "全部通过");
process.exit(fail ? 1 : 0);
