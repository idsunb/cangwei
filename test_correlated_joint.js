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
  grab("growthEntropy"),
  grab("optimizeCorrelated"),
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
check("ρ=1 同涨同跌", close(p1["HH"], 0.5) && close(p1["LL"], 0.5), JSON.stringify(p1));
const rm = joint2x2(0.5, 0.5, -1, -0.1, 0.3, -0.15, 0.4);
const pm = Object.fromEntries(rm.map((r) => [r.label, r.p]));
check("ρ=−1 一涨一跌", close(pm["HL"], 0.5) && close(pm["LH"], 0.5), JSON.stringify(pm));

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

console.log(fail ? `失败 ${fail} 项` : "全部通过");
process.exit(fail ? 1 : 0);
