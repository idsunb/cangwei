/* 相关性 · 最大增值熵计算器 */
"use strict";

const LOG2 = Math.log(2);
const log2 = (x) => Math.log(x) / LOG2;

function fmt(x, d = 4) {
  if (!isFinite(x)) return String(x);
  return x.toFixed(d);
}
function pct(x, d = 2) {
  if (!isFinite(x)) return String(x);
  return (x * 100).toFixed(d) + "%";
}
function num(id) {
  return parseFloat(document.getElementById(id).value);
}

/* ===================== 联合分布 ===================== */

/**
 * 两证券、两状态的精确 2×2 联合表（Bernoulli）。
 * pA = P(A=涨), pB = P(B=涨), rho = corr(状态/收益)
 * 返回 4 个情景 {pa, sa, sb, ra, rb}，sa/sb∈{0跌,1涨}
 */
function joint2x2(pA, pB, rho, rAL, rAH, rBL, rBH) {
  const sA = Math.sqrt(pA * (1 - pA) * pB * (1 - pB));
  let p11 = pA * pB + rho * sA;
  const lo = Math.max(0, pA + pB - 1);
  const hi = Math.min(pA, pB);
  const p11c = Math.min(hi, Math.max(lo, p11));
  const p10 = pA - p11c;
  const p01 = pB - p11c;
  const p00 = 1 - pA - pB + p11c;
  const rows = [
    { p: p00, sa: 0, sb: 0, ra: rAL, rb: rBL, returns: [rAL, rBL], label: "A跌·B跌" },
    { p: p01, sa: 0, sb: 1, ra: rAL, rb: rBH, returns: [rAL, rBH], label: "A跌·B涨" },
    { p: p10, sa: 1, sb: 0, ra: rAH, rb: rBL, returns: [rAH, rBL], label: "A涨·B跌" },
    { p: p11c, sa: 1, sb: 1, ra: rAH, rb: rBH, returns: [rAH, rBH], label: "A涨·B涨" },
  ];
  // 数值噪声
  rows.forEach((r) => {
    r.p = Math.max(0, r.p);
  });
  const s = rows.reduce((a, b) => a + b.p, 0) || 1;
  rows.forEach((r) => (r.p = r.p / s));
  return rows;
}

/* 标准正态 CDF（Abramowitz-Stegun） */
function normCdf(x) {
  // Hart / 7.1.26 近似
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989422804014327 * Math.exp((-x * x) / 2);
  const p =
    d * t * (0.31938153 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
  return x >= 0 ? 1 - p : p;
}

/* Box-Muller + LCG（可复现） */
function makeRng(seed) {
  let s = seed >>> 0 || 42;
  return function () {
    s = (1664525 * s + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function cholesky(S) {
  const n = S.length;
  const L = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = S[i][j];
      for (let k = 0; k < j; k++) sum -= L[i][k] * L[j][k];
      if (i === j) {
        if (sum <= 1e-12) return null;
        L[i][i] = Math.sqrt(sum);
      } else {
        L[i][j] = sum / L[j][j];
      }
    }
  }
  return L;
}

/** 最近相关矩阵（Higham 简化：对称化 + 特征截断用幂迭代近似 → 投影对角） */
function projectCorrPSD(R, n) {
  const A = R.map((row) => row.slice());
  // 对称
  for (let i = 0; i < n; i++) {
    A[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      const v = (A[i][j] + A[j][i]) / 2;
      A[i][j] = A[j][i] = Math.max(-0.999, Math.min(0.999, v));
    }
  }
  // 若 Cholesky 失败，向 (1-ε)I + εA 收缩直到成功
  const L = cholesky(A);
  if (L) return A;
  for (let t = 0; t < 30; t++) {
    const shrink = 0.05 * (t + 1);
    const B = A.map((row, i) =>
      row.map((v, j) => (i === j ? 1 : v * (1 - Math.min(0.95, shrink))))
    );
    if (cholesky(B)) return B;
  }
  return A.map((row, i) => row.map((_, j) => (i === j ? 1 : 0)));
}

/**
 * 资金结构拆分（含卖空, 书中 §3.3）:
 *   多头 q>0:  现金 = 1-Σq  (负则记为负债, 透支)
 *   卖空 q<0:  占用 |q| 作抵押 → 现金减少 |q|
 *              q ∈ [-1,0]: 现金 = 1-|q|, 负债 = 0
 *              |q|>1 深度卖空: 超出部分为负债
 *   恒等式: 现金 + 标的多头合计 = 1 + 负债 - 卖空绝对值合计
 */
function splitPosition(qs) {
  let longSum = 0, shortSum = 0;
  qs.forEach((q) => {
    if (q > 0) longSum += q;
    else if (q < 0) shortSum += -q;
  });
  // 先锁定卖空抵押, 再配多头; 不足则透支为负债
  const locked = shortSum;
  const cashBefore = 1 - locked - longSum;
  const cash = Math.max(0, cashBefore);
  const debt = Math.max(0, -cashBefore);
  return {
    cash,
    debt,
    shortSum,
    longSum,
    cashRaw: cashBefore,
    assetSum: longSum,
    shortSumAbs: shortSum,
  };
}

/**
 * 相关 vs 独立 对照仓位表: 项目 | q*（相关） | q*（独立） | 占比
 * items: [{name, q, qInd}]；现金/负债由 splitPosition 自动补全
 */
function fillWeightCompareTable(tableEl, items) {
  const qs = items.map((it) => it.q);
  const qsInd = items.map((it) => it.qInd);
  const sp = splitPosition(qs);
  const spInd = splitPosition(qsInd);
  const assetSum = qs.reduce((a, b) => a + b, 0);
  let th = "<thead><tr><th>项目</th><th>q*（相关）</th><th>q*（独立）</th><th>占比</th></tr></thead><tbody>";
  th += `<tr><td>现金</td><td>${fmt(sp.cash, 4)}</td><td>${fmt(spInd.cash, 4)}</td><td>${pct(sp.cash)}</td></tr>`;
  th += `<tr><td>负债</td><td>${fmt(sp.debt, 4)}</td><td>${fmt(spInd.debt, 4)}</td><td>${pct(sp.debt)}</td></tr>`;
  items.forEach((it) => {
    const isShort = it.q < -1e-12;
    th += `<tr><td>${it.name}${isShort ? "（卖空）" : ""}</td><td>${fmt(it.q, 4)}</td><td>${fmt(it.qInd, 4)}</td><td>${pct(it.q)}</td></tr>`;
  });
  th += `<tr><td><b>标的合计</b></td><td><b>${fmt(assetSum, 4)}</b></td><td><b>${fmt(qsInd.reduce((a, b) => a + b, 0), 4)}</b></td><td><b>${pct(assetSum)}</b></td></tr>`;
  th += "</tbody>";
  tableEl.innerHTML = th;
}

function fillWeightTable(tableEl, items, opts) {
  opts = opts || {};
  const qs = items.map((it) => it.q);
  const sp = splitPosition(qs);
  const cash = sp.cash;
  const debt = sp.debt;
  const assetSum = items.reduce((a, b) => a + b.q, 0);
  let th = "<thead><tr><th>项目</th><th>q*</th><th>占比</th></tr></thead><tbody>";
  th += `<tr><td>现金</td><td>${fmt(cash, 4)}</td><td>${pct(cash)}</td></tr>`;
  th += `<tr><td>负债</td><td>${fmt(debt, 4)}</td><td>${pct(debt)}</td></tr>`;
  items.forEach((it) => {
    const isShort = it.q < -1e-12;
    th += `<tr><td>${it.name}${isShort ? "（卖空）" : ""}</td><td>${fmt(it.q, 4)}</td><td>${pct(it.q)}</td></tr>`;
  });
  th += `<tr><td><b>标的合计</b></td><td><b>${fmt(assetSum, 4)}</b></td><td><b>${pct(assetSum)}</b></td></tr>`;
  th += "</tbody>";
  tableEl.innerHTML = th;
}
function marginalCuts(states) {
  const ps = states.map((s) => Math.max(0, s.p));
  const sum = ps.reduce((a, b) => a + b, 0) || 1;
  const cuts = [0];
  let acc = 0;
  for (const p of ps) {
    acc += p / sum;
    cuts.push(Math.min(1, acc));
  }
  cuts[cuts.length - 1] = 1;
  return cuts;
}

function pickState(u, cuts) {
  // cuts.length = n+1
  for (let j = 0; j < cuts.length - 1; j++) {
    if (u <= cuts[j + 1] + 1e-15) return j;
  }
  return cuts.length - 2;
}

/** 危机相关加成: 非对角 ρ += dRho, 并夹紧到 [-0.99,0.99] */
function stressCorr(corr, dRho) {
  if (!dRho) return corr.map((r) => r.slice());
  return corr.map((row, i) =>
    row.map((v, j) => {
      if (i === j) return 1;
      return Math.max(-0.99, Math.min(0.99, v + dRho));
    })
  );
}

/** Student-t CDF（正则化不完全 Beta） */
function tCdf(t, nu) {
  if (!isFinite(t)) return t > 0 ? 1 : 0;
  const x = nu / (nu + t * t);
  const p = 0.5 * ibeta(x, nu / 2, 0.5);
  return t > 0 ? 1 - p : p;
}

/** 正则化不完全 Beta I_x(a,b)（连分式，Numerical Recipes 风格） */
function ibeta(x, a, b) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lbeta) / a;
  // Lentz 连分式
  let f = 1, c = 1, d = 0;
  for (let i = 0; i <= 200; i++) {
    const m = Math.floor(i / 2);
    let num;
    if (i === 0) num = 1;
    else if (i % 2 === 0) num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    else num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + num * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    d = 1 / d;
    c = 1 + num / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    const cd = c * d;
    f *= cd;
    if (Math.abs(1 - cd) < 1e-10) break;
  }
  return front * (f - 1);
}

function lgamma(z) {
  const g = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1;
  let x = 0.99999999999980993;
  for (let i = 0; i < 8; i++) x += g[i] / (z + i + 1);
  const t = z + 7.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

/** χ² 抽样: 整数 ν 用正态平方和; 大 ν 用 Wilson–Hilferty; 其余 Marsaglia-Tsang */
function chi2Sample(nu, rng) {
  const k = Math.round(nu);
  if (Math.abs(nu - k) < 1e-9 && k >= 1 && k <= 128) {
    let v = 0;
    for (let i = 0; i < k; i++) {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v += z * z;
    }
    return v;
  }
  if (nu >= 30) {
    // Wilson–Hilferty: (χ²/ν)^{1/3} ≈ 1 - 2/(9ν) + z√(2/(9ν))
    const u = Math.max(1e-12, rng());
    const v = rng();
    const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    const x = 1 - 2 / (9 * nu) + z * Math.sqrt(2 / (9 * nu));
    return Math.max(1e-12, nu * x * x * x);
  }
  // Marsaglia-Tsang: χ²(ν) = 2 · Gamma(ν/2, 1),  d = ν/2 − 1/3
  const a = nu / 2;
  const d = a - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (let tries = 0; tries < 2000; tries++) {
    let x, v;
    do {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      x = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = Math.max(1e-12, rng());
    if (u < 1 - 0.0331 * x * x * x * x) return 2 * d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return 2 * d * v;
  }
  return Math.max(1e-12, nu);
}

/**
 * t-copula: 多元 t → 边际分位 → 离散状态
 * nu 越小尾部越厚（危机同跌更强）; nu→∞ ≈ 高斯
 */
function jointFromTCopula(marginals, corr, nu, nSamples = 8000, seed = 42) {
  const marg = normalizeMarginals(marginals);
  const n = marg.length;
  const L = cholesky(corr);
  if (!L) throw new Error("相关矩阵非正定，请先投影到 PSD");
  const cuts = marg.map((m) => marginalCuts(m.states));
  const dims = marg.map((m) => m.states.length);
  const total = dims.reduce((a, b) => a * b, 1);
  if (total > 400) throw new Error(`联合情景数 ${total} 过大（>400）`);
  const counts = new Array(total).fill(0);
  const rng = makeRng(seed);
  const eps = new Array(n).fill(0);
  const nPairs = Math.max(1, Math.floor(nSamples / 2));
  const nUsed = nPairs * 2;
  for (let s = 0; s < nPairs; s++) {
    for (let i = 0; i < n; i++) {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      const r = Math.sqrt(-2 * Math.log(u1));
      const th = 2 * Math.PI * u2;
      eps[i] = r * Math.cos(th);
    }
    for (const sign of [1, -1]) {
      // 多元 t: T = Z / sqrt(V/nu), V~chi2_nu
      const V = chi2Sample(nu, rng);
      const scale = Math.sqrt(nu / Math.max(V, 1e-12));
      let idx = 0;
      for (let k = 0; k < n; k++) {
        let zk = 0;
        for (let j = 0; j <= k; j++) zk += L[k][j] * (sign * eps[j]);
        const tk = zk * scale;
        const u = tCdf(tk, nu);
        const st = pickState(u, cuts[k]);
        idx = idx * dims[k] + st;
      }
      counts[idx] += 1;
    }
  }
  const rows = [];
  for (let idx = 0; idx < total; idx++) {
    const p = counts[idx] / nUsed;
    if (p <= 0) continue;
    let t = idx;
    const tmp = [];
    for (let k = n - 1; k >= 0; k--) {
      tmp[k] = t % dims[k];
      t = Math.floor(t / dims[k]);
    }
    const sa = tmp.slice();
    const ra = sa.map((st, k) => marg[k].states[st].r);
    rows.push({
      p,
      states: sa,
      returns: ra,
      label: sa
        .map((st, k) => {
          const nm = marg[k].name;
          return nm + (marg[k].states.length === 2 ? (st ? "涨" : "跌") : "档" + (st + 1));
        })
        .join("·"),
    });
  }
  return rows;
}

/**
 * 高斯 copula：各资产状态数可不同 + 相关矩阵 → 联合概率。
 * marginals[k] = { name?, states: [{p, r}, ...] }
 * 兼容旧的 { pHigh, rLow, rHigh } 两状态写法。
 */
function normalizeMarginals(marginals) {
  return marginals.map((m, k) => {
    if (m.states && m.states.length) {
      return {
        name: m.name || "S" + (k + 1),
        states: m.states.map((s) => ({ p: +s.p, r: +s.r })),
      };
    }
    // 旧格式两状态
    return {
      name: m.name || "S" + (k + 1),
      states: [
        { p: 1 - m.pHigh, r: m.rLow },
        { p: m.pHigh, r: m.rHigh },
      ],
    };
  });
}

function jointFromCopula(marginals, corr, nSamples = 8000, seed = 42) {
  const marg = normalizeMarginals(marginals);
  const n = marg.length;
  // ρ 全为 0（单位阵）→ 精确独立，避免 MC 噪声使 q* 与独立列不一致
  let isIdentity = true;
  for (let i = 0; i < n && isIdentity; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && Math.abs(corr[i][j]) > 1e-12) {
        isIdentity = false;
        break;
      }
    }
  }
  if (isIdentity) return jointIndependent(marginals);

  const L = cholesky(corr);
  if (!L) throw new Error("相关矩阵非正定，请先投影到 PSD");
  const cuts = marg.map((m) => marginalCuts(m.states));
  const dims = marg.map((m) => m.states.length);
  const total = dims.reduce((a, b) => a * b, 1);
  if (total > 400) {
    throw new Error(`联合情景数 ${total} 过大（>400），请减少状态数`);
  }
  const counts = new Array(total).fill(0);
  const rng = makeRng(seed);
  const eps = new Array(n).fill(0);
  // 对偶变量 (antithetic): ε 与 −ε 成对，降低 MC 方差
  const nPairs = Math.max(1, Math.floor(nSamples / 2));
  const nUsed = nPairs * 2;
  for (let s = 0; s < nPairs; s++) {
    for (let i = 0; i < n; i++) {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      const r = Math.sqrt(-2 * Math.log(u1));
      const th = 2 * Math.PI * u2;
      eps[i] = r * Math.cos(th);
    }
    for (const sign of [1, -1]) {
      let idx = 0;
      for (let k = 0; k < n; k++) {
        let zk = 0;
        for (let j = 0; j <= k; j++) zk += L[k][j] * (sign * eps[j]);
        const u = normCdf(zk);
        const st = pickState(u, cuts[k]);
        idx = idx * dims[k] + st;
      }
      counts[idx] += 1;
    }
  }
  const rows = [];
  for (let idx = 0; idx < total; idx++) {
    const p = counts[idx] / nUsed;
    if (p <= 0) continue;
    const sa = [];
    const ra = [];
    let t = idx;
    // mixed radix 解码（与编码顺序一致：先资产 0 为高位）
    const tmp = [];
    for (let k = n - 1; k >= 0; k--) {
      tmp[k] = t % dims[k];
      t = Math.floor(t / dims[k]);
    }
    for (let k = 0; k < n; k++) {
      sa.push(tmp[k]);
      ra.push(marg[k].states[tmp[k]].r);
    }
    rows.push({
      p,
      states: sa,
      returns: ra,
      label: sa.map((st, k) => {
        const nm = marg[k].name;
        return nm + (marg[k].states.length === 2 ? (st ? "涨" : "跌") : "档" + (st + 1));
      }).join("·"),
    });
  }
  return rows;
}

/** 独立联合（对照）—— 各资产状态数可不同 */
function jointIndependent(marginals) {
  const marg = normalizeMarginals(marginals);
  const n = marg.length;
  const dims = marg.map((m) => m.states.length);
  const total = dims.reduce((a, b) => a * b, 1);
  if (total > 400) throw new Error(`联合情景数 ${total} 过大（>400）`);
  const rows = [];
  for (let idx = 0; idx < total; idx++) {
    let p = 1;
    const sa = [];
    const ra = [];
    let t = idx;
    const tmp = [];
    for (let k = n - 1; k >= 0; k--) {
      tmp[k] = t % dims[k];
      t = Math.floor(t / dims[k]);
    }
    for (let k = 0; k < n; k++) {
      p *= marg[k].states[tmp[k]].p;
      sa.push(tmp[k]);
      ra.push(marg[k].states[tmp[k]].r);
    }
    rows.push({
      p,
      states: sa,
      returns: ra,
      label: sa.map((st, k) => {
        const nm = marg[k].name;
        return nm + (marg[k].states.length === 2 ? (st ? "涨" : "跌") : "档" + (st + 1));
      }).join("·"),
    });
  }
  return rows;
}

/* ===================== 增值熵优化 ===================== */

function portReturns(q, returnsRows, r0) {
  return portReturnsFull(q, returnsRows, r0, r0, 0);
}

/**
 * 组合收益 R-1（含贷款 r_loan / 借券 r_borrow）
 *   现金 q0=1-Σq 抵押模型:
 *   W-1 = cashLocked·r_cost + Σ q_k r_k − Σ|q_k<0|·r_borrow
 *   cashLocked = 1 − Σ q_long − Σ |q_short|
 *   r_cost = r0 (cash≥0) 或 r_loan (cash&lt;0 透支)
 */
function portReturnsFull(qv, returnsRows, r0, rLoan, rBorrow) {
  if (rLoan == null) rLoan = r0;
  if (rBorrow == null) rBorrow = 0;
  const qs = Array.isArray(qv) ? qv : [qv];
  let longSum = 0, shortSum = 0;
  qs.forEach((x) => {
    if (x > 0) longSum += x;
    else if (x < 0) shortSum += -x;
  });
  const cashLocked = 1 - longSum - shortSum;
  const rCost = cashLocked >= 0 ? r0 : rLoan;
  return returnsRows.map((row) => {
    let ret = cashLocked * rCost - shortSum * rBorrow;
    for (let k = 0; k < qs.length; k++) {
      ret += qs[k] * row.returns[k];
    }
    return ret;
  });
}

function growthEntropy(probs, returns) {
  let H = 0;
  for (let i = 0; i < probs.length; i++) {
    const R = 1 + returns[i];
    if (R <= 0) return -Infinity;
    if (probs[i] > 0) H += probs[i] * log2(R);
  }
  return H;
}

function optimizeCorrelated(rows, r0, allowShort, allowLev, maxMultiple = 1, rLoan, rBorrow) {
  return solveCorrelated(rows, r0, allowShort, allowLev, maxMultiple, "grad", rLoan, rBorrow);
}

/**
 * 统一求解入口。method:
 *   hill  坐标爬山
 *   grad  解析梯度精修（从零点）
 *   warm  §3.8 抛物线初值 + 梯度
 * rLoan / rBorrow: 贷款利率、借券费率（可选, 默认 rLoan=r0, rBorrow=0）
 */
function solveCorrelated(rows, r0, allowShort, allowLev, maxMultiple = 1, method = "grad", rLoan, rBorrow) {
  if (rLoan == null) rLoan = r0;
  if (rBorrow == null) rBorrow = 0;
  const W = rows.length;
  const N = rows[0].returns.length;
  let P = rows.map((r) => r.p);
  const s = P.reduce((a, b) => a + b, 0) || 1;
  P = P.map((p) => p / s);
  const lo = allowShort ? -maxMultiple : 0;
  const hi = allowLev ? maxMultiple : 1;
  const sumMax = allowLev ? maxMultiple : 1;

  function H_of(qa) {
    const qs = qa.reduce((a, b) => a + b, 0);
    if (qs > sumMax + 1e-12) return -Infinity;
    if (qa.some((x) => x < -1e-12) && !allowShort) return -Infinity;
    const rets = portReturnsFull(qa, rows, r0, rLoan, rBorrow);
    let H = 0;
    for (let i = 0; i < W; i++) {
      const R = 1 + rets[i];
      if (R <= 0) return -Infinity;
      H += P[i] * log2(R);
    }
    return H;
  }
  function project(q) {
    q = q.map((x) => Math.max(lo, Math.min(hi, x)));
    const qs = q.reduce((a, b) => a + b, 0);
    if (qs > sumMax && qs > 0) {
      const k = sumMax / qs;
      q = q.map((x) => Math.max(lo, Math.min(hi, x * k)));
    }
    return q;
  }
  function grad(q) {
    const g = new Array(N).fill(0);
    let longSum = 0, shortSum = 0;
    q.forEach((x) => {
      if (x > 0) longSum += x;
      else if (x < 0) shortSum += -x;
    });
    const cashLocked = 1 - longSum - shortSum;
    const rCost = cashLocked >= 0 ? r0 : rLoan;
    for (let i = 0; i < W; i++) {
      const rets = portReturnsFull(q, [rows[i]], r0, rLoan, rBorrow);
      const R = 1 + rets[0];
      if (R <= 0) return null;
      for (let k = 0; k < N; k++) {
        // ∂cash/∂q_k = −sign(q_k); 卖空时再 −r_borrow
        const sg = q[k] >= 0 ? -1 : 1;
        const shortPen = q[k] < 0 ? rBorrow : 0;
        const d = sg * rCost + rows[i].returns[k] + shortPen;
        g[k] += (P[i] * d) / (R * Math.LN2);
      }
    }
    return g;
  }
  function hillClimb(start) {
    // 多起点: 0、单资产扫描最优、给定 start、若干随机点 → 再坐标爬山
    const starts = [project(start ? start.slice() : new Array(N).fill(0))];
    let scanBest = new Array(N).fill(0);
    let scanH = H_of(scanBest);
    for (let k = 0; k < N; k++) {
      for (let t = 0; t <= 30; t++) {
        const q = new Array(N).fill(0);
        q[k] = lo + (hi - lo) * (t / 30);
        const h = H_of(q);
        if (h > scanH) {
          scanH = h;
          scanBest = q.slice();
        }
      }
    }
    starts.push(project(scanBest));
    // 等权可行点
    const eq = new Array(N).fill(0).map(() => Math.min(hi, sumMax / N));
    starts.push(project(eq));
    // 伪随机多起点 (LCG, 可复现) — 覆盖单纯形内部与边界
    let s = 12345;
    const rnd = () => {
      s = (1664525 * s + 1013904223) >>> 0;
      return s / 4294967296;
    };
    for (let t = 0; t < 24; t++) {
      const q = new Array(N).fill(0).map(() => lo + rnd() * (hi - lo));
      const scale = 0.25 + 0.75 * rnd();
      let qs = q.reduce((a, b) => a + b, 0);
      if (qs > 1e-12) {
        const target = sumMax * scale;
        for (let i = 0; i < N; i++) q[i] *= target / qs;
      }
      starts.push(project(q));
    }

    let best = starts[0];
    let bestH = H_of(best);
    for (const st of starts) {
      let q = project(st.slice());
      let h = H_of(q);
      let step = 0.15;
      for (let it = 0; it < 150; it++) {
        let improved = false;
        for (let k = 0; k < N; k++) {
          for (const dir of [step, -step, step * 0.5, -step * 0.5]) {
            const cand = q.slice();
            cand[k] = Math.max(lo, Math.min(hi, cand[k] + dir));
            const qs = cand.reduce((a, b) => a + b, 0);
            if (qs > sumMax + 1e-12) continue;
            const hh = H_of(cand);
            if (hh > h + 1e-12) {
              h = hh;
              q = cand;
              improved = true;
            }
          }
        }
        if (!improved) {
          step *= 0.5;
          if (step < 1e-4) break;
        }
      }
      if (h > bestH) {
        bestH = h;
        best = q.slice();
      }
    }
    // 收尾抛光，避免卡在非峰点
    const pol = polish(best);
    if (pol.H > bestH) {
      bestH = pol.H;
      best = pol.q;
    }
    return { q: best, H: bestH };
  }
  /** 数值梯度短程抛光（供 hill 收尾，保证凹问题到峰） */
  function polish(q0) {
    let q = project(q0.slice());
    let h = H_of(q);
    let step = Math.max(0.4, hi - lo) * 0.5;
    for (let it = 0; it < 80; it++) {
      const g = new Array(N).fill(0);
      let ok = true;
      for (let k = 0; k < N; k++) {
        const e = 1e-4;
        const qp = q.slice(), qm = q.slice();
        qp[k] = Math.min(hi, q[k] + e);
        qm[k] = Math.max(lo, q[k] - e);
        const hp = H_of(qp), hm = H_of(qm);
        if (!isFinite(hp) || !isFinite(hm)) { ok = false; break; }
        g[k] = (hp - hm) / (2 * e);
      }
      if (!ok) break;
      const gmax = Math.max(...g.map(Math.abs), 1e-15);
      let improved = false;
      for (const sc of [step, step * 0.4, step * 0.16]) {
        const cand = project(q.map((x, k) => x + sc * g[k] / gmax));
        const hh = H_of(cand);
        if (hh > h + 1e-13) { q = cand; h = hh; improved = true; break; }
      }
      if (!improved) { step *= 0.5; if (step < 1e-5) break; }
    }
    return { q, H: h };
  }
  function gradRefine(start) {
    // 多起点: 给定点 / 零点 / 单资产极点 / 等权 / 随机 — 避免停在同一假点
    const starts = [
      project(start ? start.slice() : new Array(N).fill(0)),
      project(new Array(N).fill(0)),
      project(new Array(N).fill(0).map(() => sumMax / N)),
    ];
    for (let k = 0; k < N; k++) {
      const qk = new Array(N).fill(0);
      qk[k] = hi;
      starts.push(project(qk));
    }
    let s0 = 99;
    const rnd = () => {
      s0 = (1664525 * s0 + 1013904223) >>> 0;
      return s0 / 4294967296;
    };
    for (let t = 0; t < 8; t++) {
      const q = new Array(N).fill(0).map(() => lo + rnd() * (hi - lo));
      let qs = q.reduce((a, b) => a + b, 0);
      if (qs > 1e-12) {
        const target = sumMax * (0.3 + 0.7 * rnd());
        for (let i = 0; i < N; i++) q[i] *= target / qs;
      }
      starts.push(project(q));
    }

    let best = starts[0];
    let bestH = H_of(best);
    let bestNit = 0;
    for (const st of starts) {
      let q = project(st.slice());
      let h = H_of(q);
      let step = Math.max(0.5, hi - lo);
      let nit = 0;
      for (; nit < 200; nit++) {
        const g = grad(q);
        if (!g) break;
        const gmax = Math.max(...g.map(Math.abs), 1e-15);
        const dir = g.map((x) => x / gmax);
        let improved = false;
        for (const sc of [step, step * 0.5, step * 0.25, step * 0.125, step * 0.0625]) {
          const cand = project(q.map((x, k) => x + sc * dir[k]));
          const hh = H_of(cand);
          if (hh > h + 1e-14) {
            q = cand;
            h = hh;
            improved = true;
            break;
          }
        }
        if (!improved) {
          step *= 0.5;
          if (step < 1e-5) break;
        }
      }
      // 沿边界微调 (Σq 贴约束时，比例才是关键)
      for (let t = 0; t < 40; t++) {
        let moved = false;
        for (let a = 0; a < N; a++) {
          for (let b = 0; b < N; b++) {
            if (a === b) continue;
            for (const d of [0.02, -0.02, 0.005, -0.005]) {
              const cand = q.slice();
              cand[a] += d;
              cand[b] -= d;
              const cq = project(cand);
              const hh = H_of(cq);
              if (hh > h + 1e-14) {
                q = cq;
                h = hh;
                moved = true;
              }
            }
          }
        }
        if (!moved) break;
      }
      if (h > bestH) {
        bestH = h;
        best = q.slice();
        bestNit = nit;
      }
    }
    return { q: best, H: bestH, nit: bestNit };
  }
  /** §3.8 单资产抛物线 → 初值 */
  function approxWarmstart(t = 0.5) {
    const qInit = new Array(N).fill(0);
    for (let k = 0; k < N; k++) {
      const one = (x) => {
        const q = new Array(N).fill(0);
        q[k] = x;
        return H_of(q);
      };
      const ts = [t, Math.min(hi, 1), Math.min(hi, Math.max(t, hi / 2))];
      let bestQ = 0;
      let bestOne = one(0);
      for (const tt of [...new Set(ts)]) {
        if (!(tt > 0)) continue;
        const H0 = one(0), Ht2 = one(tt / 2), Ht = one(tt);
        if (!(isFinite(H0) && isFinite(Ht2) && isFinite(Ht))) continue;
        const a = 2 * (Ht - 2 * Ht2 + H0) / (tt * tt);
        const b = (4 * Ht2 - Ht - 3 * H0) / tt;
        let x = a === 0 ? 0 : -b / (2 * a);
        x = Math.max(lo, Math.min(hi, x));
        const h = one(x);
        if (h > bestOne) {
          bestOne = h;
          bestQ = x;
        }
      }
      qInit[k] = bestQ;
    }
    return project(qInit);
  }

  let out;
  let label = method;
  if (method === "hill") {
    const h = hillClimb(new Array(N).fill(0));
    out = h;
    label = "坐标爬山";
  } else if (method === "warm") {
    const x0 = approxWarmstart(0.5);
    const g = gradRefine(x0);
    out = g;
    label = "§3.8初值+梯度";
    out.warmstart = x0;
  } else {
    const g = gradRefine(new Array(N).fill(0));
    out = g;
    label = "解析梯度精修";
  }

  const qs = out.q.reduce((a, b) => a + b, 0);
  const portRs = portReturnsFull(out.q, rows, r0, rLoan, rBorrow);
  const H = growthEntropy(P, portRs);
  let logRg = 0;
  let ok = true;
  for (let i = 0; i < W; i++) {
    if (1 + portRs[i] <= 0) {
      ok = false;
      break;
    }
    logRg += P[i] * Math.log(1 + portRs[i]);
  }
  const sp = splitPosition(out.q);
  return {
    weights: out.q.slice(),
    cash: sp.cash,
    debt: sp.debt,
    assetSum: sp.longSum,
    shortSum: sp.shortSum,
    cashRaw: sp.cashRaw,
    H,
    rg: ok ? Math.exp(logRg) - 1 : -1,
    ra: P.reduce((a, p, i) => a + p * portRs[i], 0),
    portRs,
    method: label,
    nit: out.nit != null ? out.nit : "—",
    warmstart: out.warmstart,
  };
}

/** 三方法对比（含耗时 ms） */
function compareCorrelated(rows, r0, allowShort, allowLev, maxMultiple = 1, rLoan, rBorrow) {
  const methods = ["hill", "grad", "warm"];
  const results = methods.map((m) => {
    const t0 = (typeof performance !== "undefined" ? performance.now() : Date.now());
    const res = solveCorrelated(rows, r0, allowShort, allowLev, maxMultiple, m, rLoan, rBorrow);
    const t1 = (typeof performance !== "undefined" ? performance.now() : Date.now());
    res.elapsedMs = t1 - t0;
    return res;
  });
  const ranked = results.slice().sort((a, b) => (b.H || -Infinity) - (a.H || -Infinity));
  return { results, ranking: ranked.map((r) => r.method), best: ranked[0].method };
}

/* ===================== 单证券闭式 §3.2 ===================== */

/**
 * 单证券两状态闭式 (§3.2 / §3.3):
 *   q'  = -(P1Δ1+P2Δ2)/(Δ1Δ2)·R0        自有资金区 q∈[0,1]
 *   q'' = -(P1Δ1'+P2Δ2')/(Δ1'Δ2')·R0'   透支区 q>1
 *   卖空区: Δ_- = r + r0 + r_b (借券费), 现金 (1-|q|) 得 r0
 */
function optimalSingleClosed(probs, returns, r0, allowLev, allowShort, M, rLoan, rBorrow) {
  if (probs.length !== 2 || returns.length !== 2) {
    throw new Error("闭式仅适用于两种可能收益");
  }
  if (rLoan == null) rLoan = r0;
  if (rBorrow == null) rBorrow = 0;
  let P1 = probs[0], P2 = probs[1];
  let r1 = returns[0], r2 = returns[1];
  let D1 = r1 - r0, D2 = r2 - r0;
  if (D1 > D2) {
    let t = D1; D1 = D2; D2 = t;
    t = P1; P1 = P2; P2 = t;
    t = r1; r1 = r2; r2 = t;
  }
  const R0 = 1 + r0;
  const RL = 1 + rLoan;
  const D1p = r1 - rLoan, D2p = r2 - rLoan;
  // 卖空区超常: Δ_- = r + r0 + r_b  (抵押得 r0, 借券付 r_b)
  const Dm1 = r1 + r0 + rBorrow, Dm2 = r2 + r0 + rBorrow;
  const Eex = P1 * D1 + P2 * D2;

  let qRaw, region = "", notes = [];
  let qPick = 0;
  // 组合收益 R-1
  // 多头: (1-q)r0 + q r
  // 透支 q>1: -（q-1）r0' + q r
  // 卖空 -1≤q<0: (1+q) r0 + q r - |q| r_b   （书中 1+(1-|q|)r0+qr，再扣借券）
  // 深度卖空 q<-1: 上式再减 (|q|-1) r0'
  const portR2 = (qq) => returns.map((r) => {
    if (qq > 1) return -(qq - 1) * rLoan + qq * r;
    if (qq >= 0) return (1 - qq) * r0 + qq * r;
    const absq = -qq;
    let ret = (1 + qq) * r0 + qq * r - absq * rBorrow;
    if (qq < -1) ret -= (absq - 1) * rLoan;
    return ret;
  });

  if (D1 >= 0) {
    qRaw = allowLev ? M : 1;
    qPick = qRaw;
    region = "只赢不亏";
    notes.push("只赢不亏 → " + (allowLev ? "上限 M" : "满仓"));
  } else if (D2 <= 0) {
    // 只亏: 允许卖空则做空 (亏的资产跌了才赚? 两状态都是亏→不碰; 若 r 均为负, 卖空可赚)
    // 书中: 只亏不赢且不许卖空 → 空仓; 允许卖空时可做空
    if (allowShort && (Dm1 > 0 || Dm2 > 0)) {
      // 卖空等价于换符号的多头
      const q_s = -((P1 * Dm1 + P2 * Dm2) / (Dm1 * Dm2)) * R0;
      qRaw = q_s;
      qPick = Math.max(-M, Math.min(0, q_s));
      region = "卖空区";
      notes.push("两状态皆亏 → 卖空闭式 q₋=" + fmt(q_s, 6) + "（r_b=" + fmt(rBorrow, 4) + "）");
    } else {
      qRaw = 0;
      qPick = 0;
      region = "空仓";
      notes.push("只亏不赢 → 空仓" + (allowShort ? "（借券后仍不划算）" : "（不许卖空）"));
    }
  } else if (Eex <= 0 && !allowShort) {
    qRaw = 0;
    qPick = 0;
    region = "空仓";
    notes.push("期望超常收益≤0 → 空仓");
  } else {
    const q_p = -((P1 * D1 + P2 * D2) / (D1 * D2)) * R0;
    const q_pp = (D1p === 0 || D2p === 0) ? NaN : -((P1 * D1p + P2 * D2p) / (D1p * D2p)) * RL;
    const q_s = (Dm1 === 0 || Dm2 === 0) ? NaN : -((P1 * Dm1 + P2 * Dm2) / (Dm1 * Dm2)) * R0;
    qRaw = q_p;
    notes.push("闭式 q′（不含贷款/借券成本）=" + fmt(q_p, 6));
    if (isFinite(q_s) && q_s < 0) {
      notes.push("卖空闭式 q₋=" + fmt(q_s, 6) + "（借券 r_b=" + fmt(rBorrow, 4) + "）");
    }
    qPick = q_p;
    if (isFinite(q_pp) && q_pp >= 1 && allowLev && q_p > 1) {
      region = "透支区";
      notes.push("透支闭式 q″=" + fmt(q_pp, 6) + "（贷款 r₀′=" + fmt(rLoan, 4) + "）");
    } else if (allowLev && q_p > 1) {
      region = "满仓边界";
      notes.push("透支区无内点（r₀′=" + fmt(rLoan, 4) + "）→ 边界/数值");
    } else {
      region = q_p > 1 ? "满仓" : q_p < 0 ? "卖空" : "多头";
    }
    // 候选点数值择优 (含卖空)
    const lo = allowShort ? -M : 0;
    const hi = allowLev ? M : 1;
    const cands = [q_p, q_pp, q_s, 0, 1, -1, lo, hi];
    let bestQ = q_p, bestH = -Infinity;
    for (const cand of cands) {
      if (!isFinite(cand)) continue;
      const qq = Math.min(hi, Math.max(lo, cand));
      const h = growthEntropy(probs, portR2(qq));
      if (h > bestH) { bestH = h; bestQ = qq; }
    }
    // 若无卖空且多头更优，不要选负的
    if (!allowShort && bestQ < 0) bestQ = Math.min(hi, Math.max(0, q_p));
    qPick = bestQ;
    if (Math.abs(bestQ - q_p) > 1e-3 && Math.abs(bestQ - q_pp) > 1e-3 && Math.abs(bestQ - q_s) > 1e-3) {
      notes.push("候选择优 → " + fmt(bestQ, 4));
    }
  }
  const lo = allowShort ? -M : 0;
  const hi = allowLev ? M : 1;
  const q = Math.min(hi, Math.max(lo, qPick));
  if (qPick > hi + 1e-12) notes.push("触及上限 " + hi);
  if (qPick < lo - 1e-12) notes.push("触及下限 " + lo);

  const rs = portR2(q);
  const H = growthEntropy(probs, rs);
  let logRg = 0, ok = true;
  for (let i = 0; i < probs.length; i++) {
    if (1 + rs[i] <= 0) { ok = false; break; }
    logRg += probs[i] * Math.log(1 + rs[i]);
  }
  // 现金/负债/卖空: 用 splitPosition 口径
  const sp = splitPosition([q]);
  return {
    qRaw, q,
    cashRaw: sp.cashRaw,
    cash: sp.cash,
    debt: sp.debt,
    H, rg: ok ? Math.exp(logRg) - 1 : -1,
    ra: probs[0] * rs[0] + probs[1] * rs[1],
    region, rLoan, rBorrow,
    notes,
  };
}

function drawOneHCurve(probs, returns, r0, qStar, rLoan) {
  if (rLoan == null) rLoan = r0;
  const canvas = document.getElementById("o-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 48, r: 16, t: 18, b: 34 };
  ctx.clearRect(0, 0, W, H);
  const qs = [], Hs = [];
  let hMin = Infinity, hMax = -Infinity;
  for (let i = 0; i <= 200; i++) {
    const q = i / 200;
    const rs = returns.map((r) => {
      const R = q > 1 ? -(q - 1) * (1 + rLoan) + q * (1 + r) : (1 - q) * (1 + r0) + q * (1 + r);
      return R - 1;
    });
    const Hq = growthEntropy(probs, rs);
    qs.push(q); Hs.push(Hq);
    if (isFinite(Hq)) { hMin = Math.min(hMin, Hq); hMax = Math.max(hMax, Hq); }
  }
  if (!isFinite(hMin)) { hMin = -0.5; hMax = 0.5; }
  if (hMax - hMin < 1e-6) hMax = hMin + 1;
  const x = (q) => pad.l + q * (W - pad.l - pad.r);
  const y = (h) => pad.t + (1 - (h - hMin) / (hMax - hMin)) * (H - pad.t - pad.b);
  ctx.strokeStyle = "#e4ddd2";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, H - pad.b); ctx.lineTo(W - pad.r, H - pad.b);
  ctx.stroke();
  ctx.fillStyle = "#6b6560"; ctx.font = "11px sans-serif";
  ctx.fillText("q=0", pad.l, H - pad.b + 16);
  ctx.fillText("q=1", W - pad.r - 24, H - pad.b + 16);
  ctx.strokeStyle = "#0f5c4c"; ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < qs.length; i++) {
    if (!isFinite(Hs[i])) { started = false; continue; }
    if (!started) { ctx.moveTo(x(qs[i]), y(Hs[i])); started = true; }
    else ctx.lineTo(x(qs[i]), y(Hs[i]));
  }
  ctx.stroke();
  if (isFinite(qStar) && qStar >= 0 && qStar <= 1) {
    const rs = returns.map((r) => (1 - qStar) * (1 + r0) + qStar * (1 + r) - 1);
    const hOpt = growthEntropy(probs, rs);
    if (isFinite(hOpt)) {
      ctx.fillStyle = "#8a3b12";
      ctx.beginPath(); ctx.arc(x(qStar), y(hOpt), 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillText("q*=" + fmt(qStar, 3), x(qStar) + 8, y(hOpt) - 8);
    }
  }
}

function runOne() {
  let p1 = num("o-p1");
  let p2 = num("o-p2");
  if (Math.abs(p1 + p2 - 1) > 1e-9 && p1 >= 0 && p1 <= 1) p2 = 1 - p1;
  const r1 = num("o-r1"), r2 = num("o-r2"), r0 = num("o-r0");
  const lev = document.getElementById("o-lev").checked;
  const sh = document.getElementById("o-short").checked;
  const M = num("o-M") || 2;
  const rl = num("o-rl");
  const rLoan = isFinite(rl) ? rl : r0;
  const rb = num("o-rb");
  const rBorrow = isFinite(rb) ? rb : 0;
  const cap = num("o-cap") || 0;
  const res = optimalSingleClosed([p1, p2], [r1, r2], r0, lev, sh, M, rLoan, rBorrow);
  document.getElementById("o-qraw").textContent = fmt(res.qRaw, 4);
  document.getElementById("o-q").textContent = pct(res.q);
  document.getElementById("o-cash").textContent = pct(res.cash);
  document.getElementById("o-debt").textContent = pct(res.debt);
  document.getElementById("o-rg").textContent = pct(res.rg);
  document.getElementById("o-ra").textContent = pct(res.ra);
  document.getElementById("o-H").textContent = isFinite(res.H) ? fmt(res.H, 4) : "−∞";
  document.getElementById("o-amt").textContent = (res.q * cap).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  document.getElementById("o-amt0").textContent = (res.cash * cap).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  document.getElementById("o-amtd").textContent = (res.debt * cap).toLocaleString("zh-CN", { maximumFractionDigits: 0 });
  document.getElementById("o-qhint").textContent = res.region || "";
  const msgs = res.notes.slice();
  if (lev || sh) {
    msgs.push(
      "资金成本：存款 r₀=" + fmt(r0, 4) +
      "，贷款 r₀′=" + fmt(rLoan, 4) +
      "，借券 r_b=" + fmt(rBorrow, 4) +
      (res.q > 1 ? "（透支用 r₀′）" : res.q < 0 ? "（卖空用 r_b）" : "")
    );
  }
  if (1 + r1 <= 0 || r1 <= -(1 + r0) * 0.999) {
    msgs.push("存在接近/达到 100% 亏损：仓位不应超过 1−P₁ = " + pct(1 - p1, 1));
  }
  const note = document.getElementById("o-note");
  note.style.display = msgs.length ? "block" : "none";
  note.innerHTML = msgs.filter(Boolean).join("<br>");
  fillWeightTable(document.getElementById("o-weights"), [
    { name: "证券 A", q: res.q },
  ], { cashRaw: 1 - res.q });
  drawOneHCurve([p1, p2], [r1, r2], r0, res.q, rLoan);
}

document.getElementById("o-run").addEventListener("click", runOne);
// 单证券输入不自动求解
["o-p1", "o-p2", "o-r1", "o-r2", "o-r0", "o-M", "o-cap", "o-rl", "o-rb"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener("change", () => {});
    el.addEventListener("input", () => {});
  }
});
document.getElementById("o-lev").addEventListener("change", () => {});
document.getElementById("o-short").addEventListener("change", () => {});
document.querySelectorAll("[data-one]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const e = {
      coin: { p1: 0.5, r1: -1, r2: 2, r0: 0, lev: false, short: false, M: 2 },
      stock: { p1: 0.5, r1: -0.3, r2: 0.8, r0: 0.1, lev: false, short: false, M: 2 },
      future: { p1: 0.5, r1: -1, r2: 3, r0: 0, lev: false, short: false, M: 2 },
      opt: { p1: 0.7, r1: -1, r2: 3, r0: 0, lev: false, short: false, M: 2 },
      dice: { p1: 1 / 3, r1: -1, r2: 1, r0: 0, lev: false, short: false, M: 2 },
      lev: { p1: 0.5, r1: 0.08, r2: -0.05, r0: 0, lev: true, short: false, M: 5, rb: 0 },
      // 卖空例: 资产可能 +5% / −8%；默认 r_b=0
      short: { p1: 0.5, r1: 0.05, r2: -0.08, r0: 0, lev: false, short: true, M: 2, rb: 0 },
    }[btn.getAttribute("data-one")];
    document.getElementById("o-p1").value = e.p1;
    document.getElementById("o-p2").value = 1 - e.p1;
    document.getElementById("o-r1").value = e.r1;
    document.getElementById("o-r2").value = e.r2;
    document.getElementById("o-r0").value = e.r0;
    document.getElementById("o-lev").checked = e.lev;
    document.getElementById("o-short").checked = e.short;
    document.getElementById("o-M").value = e.M;
    if (e.rb != null) document.getElementById("o-rb").value = e.rb;
    // 预设只填输入，不自动求解
  });
});

/* ===================== UI 多证券 runMulti ===================== */

/* ===================== UI ===================== */

/**
 * 与单证券一致的结果指标卡: q* / 现金 / 负债 / 几何 / 期望 / H / 金额
 */
function fillResultCards(el, opts) {
  const {
    qItems = [],   // [{name, q}]
    cash, debt, rg, ra, H,
    capital = 0,
    extra = [],    // [{k, v, warn?}]
  } = opts;
  const assetSum = qItems.reduce((a, b) => a + b.q, 0);
  const amt = assetSum * capital;
  let html = "";
  qItems.forEach((it) => {
    html += `<div class="metric"><div class="k">最优 ${it.name} q*</div><div class="v">${fmt(it.q, 4)}</div><div class="hint">${pct(it.q)}</div></div>`;
  });
  html += `<div class="metric"><div class="k">现金</div><div class="v">${pct(cash)}</div></div>`;
  html += `<div class="metric"><div class="k">负债</div><div class="v ${debt > 1e-9 ? "warn" : ""}">${pct(debt)}</div></div>`;
  html += `<div class="metric"><div class="k">几何平均</div><div class="v">${pct(rg)}</div></div>`;
  html += `<div class="metric"><div class="k">期望收益</div><div class="v">${pct(ra)}</div></div>`;
  html += `<div class="metric"><div class="k">增值熵 H</div><div class="v">${isFinite(H) ? fmt(H, 4) : "−∞"}</div><div class="hint">bit / 期</div></div>`;
  extra.forEach((e) => {
    html += `<div class="metric"><div class="k">${e.k}</div><div class="v ${e.warn ? "warn" : ""}">${e.v}</div></div>`;
  });
  if (capital > 0) {
    html += `<div class="metric"><div class="k">投入金额</div><div class="v">${(amt).toLocaleString("zh-CN", { maximumFractionDigits: 0 })}</div></div>`;
    html += `<div class="metric"><div class="k">现金金额</div><div class="v">${(cash * capital).toLocaleString("zh-CN", { maximumFractionDigits: 0 })}</div></div>`;
    html += `<div class="metric"><div class="k">负债金额</div><div class="v">${(debt * capital).toLocaleString("zh-CN", { maximumFractionDigits: 0 })}</div></div>`;
  }
  el.innerHTML = html;
}

function fillJointTable(tableEl, rows, n) {
  // 概率从大到小显示
  const sorted = rows.slice().sort((a, b) => b.p - a.p);
  let th = "<thead><tr><th>情景</th><th>概率 P</th>";
  for (let k = 0; k < n; k++) th += `<th>${k === 0 ? "A" : k === 1 ? "B" : "S" + k} 原始收益</th>`;
  th += "<th>等权组合收益</th><th>组合收益(q*)</th></tr></thead><tbody>";
  sorted.forEach((r) => {
    const label = r.label || r.states.join("·");
    const eq = r.returns.reduce((a, b) => a + b, 0) / (r.returns.length || 1);
    th += `<tr><td>${label}</td><td>${fmt(r.p, 4)}</td>`;
    r.returns.forEach((x) => (th += `<td>${pct(x, 2)}</td>`));
    th += `<td>${pct(eq, 2)}</td>`;
    th += `<td>${r.portR != null ? pct(r.portR, 2) : "—"}</td></tr>`;
  });
  th += "</tbody>";
  tableEl.innerHTML = th;
}

function runTwo() {
  const pA = num("t-pA");
  const pB = num("t-pB");
  const rAL = num("t-rAL"), rAH = num("t-rAH");
  const rBL = num("t-rBL"), rBH = num("t-rBH");
  const rho = num("t-rho");
  const r0 = num("t-r0") || 0;
  const lev = document.getElementById("t-lev").checked;
  const sh = document.getElementById("t-short").checked;
  const M = num("t-M") || 1;
  const rl = num("t-rl");
  const rb = num("t-rb");
  const rLoan = isFinite(rl) ? rl : r0;
  const rBorrow = isFinite(rb) ? rb : 0;

  const rows = joint2x2(pA, pB, rho, rAL, rAH, rBL, rBH);
  const res = optimizeCorrelated(rows, r0, sh, lev, M, rLoan, rBorrow);
  rows.forEach((row, i) => {
    row.portR = res.portRs[i];
  });
  fillJointTable(document.getElementById("t-joint"), rows, 2);

  const out = document.getElementById("t-out");
  const capT = num("t-cap") || 0;
  fillResultCards(out, {
    qItems: [
      { name: "证券 A", q: res.weights[0] },
      { name: "证券 B", q: res.weights[1] },
    ],
    cash: res.cash,
    debt: res.debt,
    rg: res.rg,
    ra: res.ra != null ? res.ra : 0,
    H: res.H,
    capital: capT,
    extra: [
      { k: "标的合计", v: pct(res.assetSum) },
      { k: "ρ", v: fmt(rho, 2) },
    ],
  });

  fillWeightTable(document.getElementById("t-weights"), [
    { name: "证券 A", q: res.weights[0] },
    { name: "证券 B", q: res.weights[1] },
  ], { cashRaw: res.cashRaw });

  const sA = Math.sqrt(Math.max(0, pA * (1 - pA) * pB * (1 - pB)));
  const p11raw = pA * pB + rho * sA;
  document.getElementById("t-note").innerHTML =
    `ρ=${fmt(rho, 2)} → P(A涨·B涨)=${fmt(rows[3].p, 4)}，` +
    `qA*=${fmt(res.weights[0], 4)}，qB*=${fmt(res.weights[1], 4)}，H*=${fmt(res.H, 4)} bit。` +
    `改变 ρ 会改变联合概率与 H*；在边际对称时 q* 可能几乎不变（H* 仍变）。` +
    (p11raw < Math.max(0, pA + pB - 1) - 1e-12 || p11raw > Math.min(pA, pB) + 1e-12
      ? `<br>当前 ρ 超出可行范围，已截断到 Frechet 边界。`
      : "");

  drawRhoCurve(pA, pB, rAL, rAH, rBL, rBH, r0, sh, lev, M, rho);
}

function drawRhoCurve(pA, pB, rAL, rAH, rBL, rBH, r0, sh, lev, M, markRho) {
  const canvas = document.getElementById("t-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 48, r: 16, t: 18, b: 34 };
  ctx.clearRect(0, 0, W, H);
  const xs = [], hH = [], qA = [], qB = [];
  for (let i = 0; i <= 40; i++) {
    const rho = -1 + (2 * i) / 40;
    const rows = joint2x2(pA, pB, rho, rAL, rAH, rBL, rBH);
    const res = optimizeCorrelated(rows, r0, sh, lev, M);
    xs.push(rho);
    hH.push(res.H);
    qA.push(res.weights[0]);
    qB.push(res.weights[1]);
  }
  const yMin = Math.min(...hH.filter(isFinite), 0);
  const yMax = Math.max(...hH.filter(isFinite), 0.05);
  const x = (rho) => pad.l + ((rho + 1) / 2) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);
  ctx.strokeStyle = "#e4ddd2";
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, H - pad.b);
  ctx.lineTo(W - pad.r, H - pad.b);
  ctx.stroke();
  ctx.fillStyle = "#6b6560";
  ctx.font = "11px sans-serif";
  ctx.fillText("ρ=-1", pad.l, H - pad.b + 16);
  ctx.fillText("ρ=0", x(0) - 10, H - pad.b + 16);
  ctx.fillText("ρ=1", W - pad.r - 28, H - pad.b + 16);
  // H curve
  ctx.strokeStyle = "#0f5c4c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  xs.forEach((rho, i) => {
    const px = x(rho), py = y(hH[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  // mark
  if (markRho >= -1 && markRho <= 1) {
    const rows = joint2x2(pA, pB, markRho, rAL, rAH, rBL, rBH);
    const res = optimizeCorrelated(rows, r0, sh, lev, M);
    ctx.fillStyle = "#8a3b12";
    ctx.beginPath();
    ctx.arc(x(markRho), y(res.H), 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(`ρ=${markRho}  H=${fmt(res.H, 3)}`, x(markRho) + 8, y(res.H) - 8);
  }
  ctx.fillStyle = "#0f5c4c";
  ctx.fillText("H*(ρ)", pad.l + 4, pad.t + 12);
}

/* ---- 多证券（各资产状态数可不同） ---- */
let margState = [];
let corrState = [];

function defaultMarg2() {
  return {
    name: "A",
    states: [
      { p: 0.5, r: -0.1 },
      { p: 0.5, r: 0.3 },
    ],
  };
}

function initMulti(n) {
  n = Math.max(2, Math.min(6, n || 3));
  margState = Array.from({ length: n }, (_, k) => ({
    name: "S" + (k + 1),
    states: [
      { p: 0.5, r: -0.1 - 0.02 * k },
      { p: 0.5, r: 0.2 + 0.05 * k },
    ],
  }));
  corrState = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : Math.pow(0.5, Math.abs(i - j))))
  );
  document.getElementById("n-count").value = String(n);
  renderMultiTables();
}

function loadExample2x3() {
  margState = [
    {
      name: "A",
      states: [
        { p: 0.5, r: -0.1 },
        { p: 0.5, r: 0.3 },
      ],
    },
    {
      name: "B",
      states: [
        { p: 0.3, r: -0.25 },
        { p: 0.4, r: 0.08 },
        { p: 0.3, r: 0.5 },
      ],
    },
  ];
  corrState = [
    [1, 0.4],
    [0.4, 1],
  ];
  document.getElementById("n-count").value = "2";
  renderMultiTables();
}

/** 约束 Σp=1，并按 p 从大到小排序（仅在用户要求时调用） */
function normalizeAndSortStates(states) {
  let sum = 0;
  states.forEach((s) => {
    s.p = Math.max(0, parseFloat(s.p) || 0);
    sum += s.p;
  });
  if (sum > 0) {
    states.forEach((s) => {
      s.p = s.p / sum;
    });
  } else if (states.length) {
    states.forEach((s) => {
      s.p = 1 / states.length;
    });
  }
  states.sort((a, b) => b.p - a.p || b.r - a.r);
  return states;
}

/** 仅排序，不改数值 */
function sortStatesByP(states) {
  states.sort((a, b) => (b.p - a.p) || (b.r - a.r));
  return states;
}

function renderMultiTables() {
  const n = margState.length;
  const mb = document.getElementById("n-marg");
  let mh = "";
  margState.forEach((m, k) => {
    const sumP = m.states.reduce((a, b) => a + (parseFloat(b.p) || 0), 0);
    const badSum = Math.abs(sumP - 1) > 1e-6;
    mh += `<div class="card" style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <strong>${m.name}（${m.states.length} 态）Σp=<span style="${badSum ? "color:#8a3b12" : ""}">${fmt(sumP, 4)}</span>${badSum ? "　⚠ 请调到 1.0000，或点归一化" : ""}</strong>
        <span>
          <button class="ghost" data-madd="${k}">＋ 状态</button>
          ${m.states.length > 2 ? `<button class="ghost" data-mdelst="${k}">－ 状态</button>` : ""}
          <button class="ghost" data-mnorm="${k}" title="所有 p 除以 Σp，并按大小排序">归一化</button>
        </span>
      </div>`;
    mh += `<div class="multi-row" style="font-size:12px;color:var(--muted)">
      <div>状态</div><div>概率 p</div><div>收益率 r</div><div></div></div>`;
    m.states.forEach((st, i) => {
      mh += `<div class="multi-row">
        <div>#${i + 1}</div>
        <div><input data-mg="${k},${i},p" value="${st.p}" /></div>
        <div><input data-mg="${k},${i},r" value="${st.r}" /></div>
        <div></div>
      </div>`;
    });
    mh += `</div>`;
  });
  mb.innerHTML = mh;
  mb.querySelectorAll("[data-mg]").forEach((inp) => {
    inp.addEventListener("change", () => {
      const [k, i, key] = inp.getAttribute("data-mg").split(",");
      margState[+k].states[+i][key] = parseFloat(inp.value);
      // 不强制改数: 只刷新 Σp 提示; 失焦后按 p 大小排序
      if (key === "p") sortStatesByP(margState[+k].states);
      renderMultiTables();
    });
  });
  mb.querySelectorAll("[data-madd]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = +btn.getAttribute("data-madd");
      margState[k].states.push({ p: 0, r: 0.1 });
      sortStatesByP(margState[k].states);
      renderMultiTables();
    });
  });
  mb.querySelectorAll("[data-mdelst]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = +btn.getAttribute("data-mdelst");
      margState[k].states.pop();
      renderMultiTables();
    });
  });
  mb.querySelectorAll("[data-mnorm]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = +btn.getAttribute("data-mnorm");
      normalizeAndSortStates(margState[k].states);
      renderMultiTables();
    });
  });

  const cb = document.getElementById("n-corr");
  let ch = `<table class="corr-table"><thead><tr><th></th>`;
  margState.forEach((m) => (ch += `<th>${m.name}</th>`));
  ch += "</tr></thead><tbody>";
  for (let i = 0; i < n; i++) {
    ch += `<tr><td><strong>${margState[i].name}</strong></td>`;
    for (let j = 0; j < n; j++) {
      if (i === j) ch += `<td><input disabled value="1" /></td>`;
      else
        ch += `<td><input data-corr="${i},${j}" value="${fmt(corrState[i][j], 3)}" /></td>`;
    }
    ch += "</tr>";
  }
  ch += "</tbody></table>";
  cb.innerHTML = ch;
  cb.querySelectorAll("[data-corr]").forEach((inp) => {
    inp.addEventListener("change", () => {
      const [i, j] = inp.getAttribute("data-corr").split(",").map(Number);
      const v = Math.max(-0.999, Math.min(0.999, parseFloat(inp.value) || 0));
      corrState[i][j] = v;
      corrState[j][i] = v;
      renderMultiTables();
    });
  });
}

function runMulti() {
  try {
    const r0 = num("n-r0") || 0;
    const lev = document.getElementById("n-lev").checked;
    const sh = document.getElementById("n-short").checked;
    const M = num("n-M") || 1;
    const method = document.getElementById("n-method").value || "grad";
    const n = margState.length;
    const corrRaw = projectCorrPSD(corrState, n);
    const corr = projectCorrPSD(corrRaw, n);
    corrState = corrRaw.map((r) => r.slice());
    renderMultiTables();
    const cop = document.getElementById("n-copula").value || "gauss";
    const nu = Math.max(2, num("n-nu") || 5);
    const joint = cop === "t"
      ? jointFromTCopula(margState, corr, nu, 8000, 42)
      : jointFromCopula(margState, corr, 8000, 42);
    const ind = jointIndependent(margState);
    const rl = num("n-rl");
    const rb = num("n-rb");
    const rLoan = isFinite(rl) ? rl : r0;
    const rBorrow = isFinite(rb) ? rb : 0;
    const res = solveCorrelated(joint, r0, sh, lev, M, method, rLoan, rBorrow);
    const resInd = solveCorrelated(ind, r0, sh, lev, M, method, rLoan, rBorrow);
    joint.forEach((row, i) => {
      row.portR = res.portRs[i];
    });

    const dims = margState.map((m) => m.states.length);
    const capN = num("n-cap") || 0;
    fillResultCards(document.getElementById("n-out"), {
      qItems: margState.map((m, k) => ({ name: m.name, q: res.weights[k] })),
      cash: res.cash,
      debt: res.debt,
      rg: res.rg,
      ra: res.ra != null ? res.ra : 0,
      H: res.H,
      capital: capN,
      extra: [
        { k: "H*（独立对照）", v: fmt(resInd.H, 4) },
        { k: "标的合计", v: pct(res.assetSum) },
        { k: "联合情景", v: String(joint.length) },
        { k: "copula", v: cop === "t" ? "t ν=" + fmt(nu, 0) : "高斯" },
      ],
    });

    fillWeightCompareTable(document.getElementById("n-weights"), margState.map((m, k) => ({
      name: m.name + "（" + m.states.length + " 态）",
      q: res.weights[k],
      qInd: resInd.weights[k],
    })));

    const top = joint.slice(0, 16);
    fillJointTable(document.getElementById("n-joint"), top, n);
    document.getElementById("n-compare-wrap").style.display = "none";

    const dH = res.H - resInd.H;
    document.getElementById("n-note").innerHTML =
      `状态数 ${dims.join("×")} = ${dims.reduce((a, b) => a * b, 1)} 格联合；` +
      (cop === "t"
        ? `t-copula ν=${fmt(nu, 0)}（尾部同跌更强）`
        : `高斯 copula`) +
      `，有效情景 ${joint.length}。方法：<b>${res.method}</b>。` +
      `相对独立：ΔH = ${fmt(dH, 4)} bit` +
      (dH < -1e-4
        ? " —— 正相关/厚尾削弱分散红利。"
        : dH > 1e-4
        ? " —— 负相关增强对冲，几何增长更好。"
        : " —— 与独立接近。");
  } catch (e) {
    document.getElementById("n-note").textContent = "错误: " + e.message;
  }
}

function runMultiCompare() {
  try {
    const r0 = num("n-r0") || 0;
    const lev = document.getElementById("n-lev").checked;
    const sh = document.getElementById("n-short").checked;
    const M = num("n-M") || 1;
    const n = margState.length;
    const corrRaw = projectCorrPSD(corrState, n);
    const corr = projectCorrPSD(corrRaw, n);
    corrState = corrRaw.map((r) => r.slice());
    renderMultiTables();
    const cop = document.getElementById("n-copula").value || "gauss";
    const nu = Math.max(2, num("n-nu") || 5);
    const joint = cop === "t"
      ? jointFromTCopula(margState, corr, nu, 8000, 42)
      : jointFromCopula(margState, corr, 8000, 42);
    const rl2 = num("n-rl");
    const rb2 = num("n-rb");
    const cmp = compareCorrelated(joint, r0, sh, lev, M,
      isFinite(rl2) ? rl2 : r0, isFinite(rb2) ? rb2 : 0);

    let th = "<thead><tr><th>方法</th><th>H (bit)</th><th>耗时 (ms)</th>";
    margState.forEach((m) => (th += `<th>${m.name}</th>`));
    th += "<th>现金</th><th>负债</th><th>ΔH</th></tr></thead><tbody>";
    const bestH = cmp.results[0].H;
    const sorted = cmp.results.slice().sort((a, b) => (b.H || -Infinity) - (a.H || -Infinity));
    sorted.forEach((r) => {
      th += `<tr><td>${r.method}</td><td>${fmt(r.H, 6)}</td><td>${fmt(r.elapsedMs != null ? r.elapsedMs : 0, 2)}</td>`;
      r.weights.forEach((q) => (th += `<td>${fmt(q, 4)}</td>`));
      th += `<td>${fmt(r.cash, 4)}</td><td>${fmt(r.debt, 4)}</td>`;
      th += `<td>${fmt(bestH - r.H, 6)}</td></tr>`;
    });
    th += "</tbody>";
    document.getElementById("n-compare-table").innerHTML = th;
    document.getElementById("n-compare-wrap").style.display = "block";
    const fastest = sorted.reduce((a, b) => ((a.elapsedMs || 1e9) <= (b.elapsedMs || 1e9) ? a : b));
    document.getElementById("n-note").innerHTML =
      `凸问题下三种方法应接近或相同；最优为 <b>${sorted[0].method}</b>（H=${fmt(sorted[0].H, 6)}），` +
      `最快为 <b>${fastest.method}</b>（${fmt(fastest.elapsedMs, 2)} ms）。`;
    // 同步展示最优
    const best = sorted[0];
    document.getElementById("n-out").innerHTML = `
      <div class="metric"><div class="k">H*（对比最优）</div><div class="v">${fmt(best.H, 4)}</div></div>
      <div class="metric"><div class="k">几何平均</div><div class="v">${pct(best.rg)}</div></div>
      <div class="metric"><div class="k">现金</div><div class="v">${pct(best.cash)}</div></div>
      <div class="metric"><div class="k">标的合计</div><div class="v">${pct(best.assetSum)}</div></div>
      <div class="metric"><div class="k">负债</div><div class="v ${best.debt > 1e-9 ? "warn" : ""}">${pct(best.debt)}</div></div>
      <div class="metric"><div class="k">方法</div><div class="v" style="font-size:14px">${best.method}</div></div>`;
    fillWeightCompareTable(document.getElementById("n-weights"), margState.map((m, k) => ({
      name: m.name, q: best.weights[k], qInd: best.weights[k],
    })));
  } catch (e) {
    document.getElementById("n-note").textContent = "错误: " + e.message;
  }
}

/* ---- ρ 扫描（仅按钮触发；两证券 / 多证券） ---- */
let scanSource = { kind: "two", note: "两证券（默认）" };

function runRhoScan() {
  try {
    const rl = num("t-rl");
    const rb = num("t-rb");
    const two = {
      pA: num("t-pA"), pB: num("t-pB"),
      rAL: num("t-rAL"), rAH: num("t-rAH"),
      rBL: num("t-rBL"), rBH: num("t-rBH"),
      r0: num("t-r0") || 0,
      sh: document.getElementById("t-short").checked,
      lev: document.getElementById("t-lev").checked,
      M: num("t-M") || 1,
    };
    const rLoan = isFinite(rl) ? rl : two.r0;
    const rBorrow = isFinite(rb) ? rb : 0;

    const canvas = document.getElementById("s-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const pad = { l: 48, r: 16, t: 18, b: 36 };
    ctx.clearRect(0, 0, W, H);

    const data = [];
    const useMulti = scanSource.kind === "multi" && scanSource.marg;
    const nStep = useMulti ? 15 : 40;
    for (let i = 0; i <= nStep; i++) {
      const rho = -1 + (2 * i) / nStep;
      let res;
      if (useMulti) {
        const n = scanSource.marg.length;
        let corr = scanSource.struct.map((row, a) =>
          row.map((v, b) => (a === b ? 1 : Math.max(-0.98, Math.min(0.98, rho * v))))
        );
        corr = projectCorrPSD(corr, n);
        if (scanSource.compareCopulas) {
          // 高斯实线 + t(ν=2) 虚线: 两套 q 都存进 d.qs / d.qsT
          const jG = jointFromCopula(scanSource.marg, corr, 2500, 42 + i);
          const resG = solveCorrelated(jG, scanSource.r0, scanSource.sh, scanSource.lev, scanSource.M,
            "grad", scanSource.rLoan, scanSource.rBorrow);
          const jT = jointFromTCopula(scanSource.marg, corr, 2, 2500, 42 + i);
          const resT = solveCorrelated(jT, scanSource.r0, scanSource.sh, scanSource.lev, scanSource.M,
            "grad", scanSource.rLoan, scanSource.rBorrow);
          data.push({
            rho, H: resG.H, qs: resG.weights.slice(), rg: resG.rg,
            qsT: resT.weights.slice(), HT: resT.H,
          });
          continue;
        }
        const joint = scanSource.cop === "t"
          ? jointFromTCopula(scanSource.marg, corr, scanSource.nu, 2500, 42 + i)
          : jointFromCopula(scanSource.marg, corr, 2500, 42 + i);
        res = solveCorrelated(joint, scanSource.r0, scanSource.sh, scanSource.lev, scanSource.M,
          "grad", scanSource.rLoan, scanSource.rBorrow);
      } else {
        const rows = joint2x2(two.pA, two.pB, rho, two.rAL, two.rAH, two.rBL, two.rBH);
        res = optimizeCorrelated(rows, two.r0, two.sh, two.lev, two.M, rLoan, rBorrow);
      }
      data.push({ rho, H: res.H, qs: res.weights.slice(), rg: res.rg });
    }
    const Hs = data.map((d) => d.H);
    const yMin = Math.min(...Hs, 0), yMax = Math.max(...Hs, 0.05);
    const x = (rho) => pad.l + ((rho + 1) / 2) * (W - pad.l - pad.r);
    const y = (v) => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - pad.t - pad.b);
    ctx.strokeStyle = "#e4ddd2";
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, H - pad.b);
    ctx.lineTo(W - pad.r, H - pad.b);
    ctx.stroke();
    // 仅画各资产 q_k*(ρ)，不画 H；compareCopulas 时 t 走虚线
    const nAsset = data[0].qs.length;
    const qColors = ["#0f5c4c", "#8a3b12", "#1f6fb2", "#6b4c9a", "#b8860b", "#2e7d32", "#c2185b"];
    const yScale = 0.85;
    function plotQ(getQ, dashed) {
      for (let k = 0; k < nAsset; k++) {
        ctx.strokeStyle = qColors[k % qColors.length];
        ctx.lineWidth = 2;
        ctx.setLineDash(dashed ? [6, 4] : []);
        ctx.beginPath();
        data.forEach((d, i) => {
          const px = x(d.rho), py = y(getQ(d, k) * (yMax - yMin) * yScale + yMin);
          i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        });
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }
    const hasT = data.some((d) => d.qsT);
    plotQ((d, k) => d.qs[k], false);
    if (hasT) plotQ((d, k) => (d.qsT ? d.qsT[k] : d.qs[k]), true);
    ctx.fillStyle = "#6b6560";
    ctx.fillText("ρ=-1", pad.l, H - pad.b + 16);
    ctx.fillText("ρ=1", W - pad.r - 24, H - pad.b + 16);
    let lx = pad.l + 6;
    for (let k = 0; k < nAsset; k++) {
      ctx.fillStyle = qColors[k % qColors.length];
      ctx.fillText(`q${k + 1}*(ρ)${hasT ? " 实=高斯" : ""}`, lx, pad.t + 12);
      lx += 72;
    }
    if (hasT) {
      ctx.fillStyle = "#6b6560";
      ctx.fillText("虚线 = t-copula ν=2", pad.l + 6, pad.t + 28);
    }

    let th = "<thead><tr><th>ρ</th><th>H*</th>";
    for (let k = 0; k < nAsset; k++) th += `<th>q${k + 1}*</th>`;
    if (hasT) {
      for (let k = 0; k < nAsset; k++) th += `<th>q${k + 1}* (t)</th>`;
    }
    th += "<th>r_g</th></tr></thead><tbody>";
    const stride = Math.max(1, Math.ceil(data.length / 12));
    for (let i = 0; i < data.length; i += stride) {
      const d = data[i];
      th += `<tr><td>${fmt(d.rho, 2)}</td><td>${fmt(d.H, 4)}</td>`;
      d.qs.forEach((q) => (th += `<td>${fmt(q, 4)}</td>`));
      if (hasT && d.qsT) d.qsT.forEach((q) => (th += `<td>${fmt(q, 4)}</td>`));
      th += `<td>${pct(d.rg)}</td></tr>`;
    }
    th += "</tbody>";
    document.getElementById("s-table").innerHTML = th;
    const best = data.reduce((a, b) => (b.H > a.H ? b : a), data[0]);
    const worst = data.reduce((a, b) => (b.H < a.H ? b : a), data[0]);
    document.getElementById("s-note").textContent =
      `${scanSource.note}。H* 在 ρ=${fmt(best.rho, 2)} 达到最大 ${fmt(best.H, 4)}；` +
      `在 ρ=${fmt(worst.rho, 2)} 最小 ${fmt(worst.H, 4)}。一般：|ρ| 越低，分散越有效。`;
  } catch (e) {
    document.getElementById("s-note").textContent = "错误: " + e.message;
  }
}

/* ---- tabs（切换不自动计算，节省资源） ---- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.getAttribute("data-tab")).classList.add("active");
  });
});

/** 绑定输入框：只记录数值，不自动求解 */
function bindLazy(id, fn) {
  const el = document.getElementById(id);
  if (!el) return;
  el.addEventListener("change", fn || (() => {}));
  el.addEventListener("input", fn || (() => {}));
}

document.getElementById("t-run").addEventListener("click", runTwo);
["t-pA", "t-pB", "t-rAL", "t-rAH", "t-rBL", "t-rBH", "t-rho", "t-r0", "t-M", "t-rl", "t-rb", "t-cap"].forEach((id) => bindLazy(id));
document.getElementById("t-lev").addEventListener("change", () => {});
document.getElementById("t-short").addEventListener("change", () => {});
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = btn.getAttribute("data-preset");
    document.getElementById("t-rho").value = p === "ind" ? "0" : p === "pos" ? "0.8" : "-0.8";
    // 预设只改输入，不自动算
  });
});

document.getElementById("n-run").addEventListener("click", runMulti);
document.getElementById("n-run-2").addEventListener("click", runMulti);
document.getElementById("n-compare").addEventListener("click", runMultiCompare);
["n-r0", "n-M", "n-rl", "n-rb", "n-cap", "n-nu"].forEach((id) => bindLazy(id));
["n-lev", "n-short", "n-method", "n-copula"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener("change", () => {
    if (id === "n-copula") {
      const wrap = document.getElementById("n-nu-wrap");
      if (wrap) wrap.style.display = el.value === "t" ? "block" : "none";
    }
    // 不自动求解
  });
});
// 初始: 高斯时不显示 ν
(function initCopulaVis() {
  const sel = document.getElementById("n-copula");
  const wrap = document.getElementById("n-nu-wrap");
  if (sel && wrap) wrap.style.display = sel.value === "t" ? "block" : "none";
})();

/** 所有非对角 ρ 同步 ±step，并自动求解 */
function stepAllRho(delta) {
  const n = margState.length;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      // 只改上三角再镜像，避免 (i,j)(j,i) 各加一次
      const v = Math.max(-0.99, Math.min(0.99, corrState[i][j] + delta));
      corrState[i][j] = v;
      corrState[j][i] = v;
    }
  }
  corrState = projectCorrPSD(corrState, n);
  renderMultiTables();
  runMulti();
}
document.getElementById("n-rho-up").addEventListener("click", () => stepAllRho(0.1));
document.getElementById("n-rho-down").addEventListener("click", () => stepAllRho(-0.1));
document.getElementById("n-add").addEventListener("click", () => {
  initMulti(margState.length + 1);
  runMulti();
});
document.getElementById("n-del").addEventListener("click", () => {
  initMulti(margState.length - 1);
  runMulti();
});
document.getElementById("n-count").addEventListener("change", () => {
  initMulti(parseInt(document.getElementById("n-count").value, 10));
  runMulti();
});
document.getElementById("n-reset").addEventListener("click", () => {
  initMulti(3);
  runMulti();
});
document.getElementById("n-example-2x3").addEventListener("click", () => {
  loadExample2x3();
  runMulti();
});
document.getElementById("n-corr-zero").addEventListener("click", () => {
  const n = margState.length;
  corrState = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );
  renderMultiTables();
});
document.getElementById("n-corr-pos").addEventListener("click", () => {
  const n = margState.length;
  corrState = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0.5))
  );
  renderMultiTables();
});
document.getElementById("n-corr-toeplitz").addEventListener("click", () => {
  const n = margState.length;
  corrState = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : Math.pow(0.5, Math.abs(i - j))))
  );
  renderMultiTables();
});
document.getElementById("n-psd").addEventListener("click", () => {
  corrState = projectCorrPSD(corrState, margState.length);
  renderMultiTables();
  document.getElementById("n-note").textContent = "已投影到最近合法相关矩阵（对称 + 可 Cholesky）。";
});

document.getElementById("s-run").addEventListener("click", runRhoScan);
document.getElementById("s-copy-two").addEventListener("click", () => {
  scanSource = {
    kind: "two",
    note: `两证券: A(p涨=${num("t-pA")}) B(p涨=${num("t-pB")})`,
  };
  document.getElementById("s-note").textContent =
    "已读取「两证券」输入。请点「开始扫描 ρ」计算。";
});
function loadMultiScan(useToeplitz, compareCopulas) {
  const marg = margState.map((m) => ({
    name: m.name,
    states: m.states.map((s) => ({ p: s.p, r: s.r })),
  }));
  const n = marg.length;
  const struct = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      if (i === j) return 1;
      return useToeplitz ? Math.pow(0.5, Math.abs(i - j)) : 1;
    })
  );
  const rl = num("n-rl"), rb = num("n-rb");
  scanSource = {
    kind: "multi",
    marg,
    struct,
    compareCopulas: !!compareCopulas,
    r0: num("n-r0") || 0,
    sh: document.getElementById("n-short").checked,
    lev: document.getElementById("n-lev").checked,
    M: num("n-M") || 1,
    rLoan: isFinite(rl) ? rl : (num("n-r0") || 0),
    rBorrow: isFinite(rb) ? rb : 0,
    cop: document.getElementById("n-copula").value || "gauss",
    nu: Math.max(2, num("n-nu") || 5),
    note: `多证券 ${marg.map((m) => m.name + "(" + m.states.length + "态)").join(" / ")}`
      + (useToeplitz ? " · 近强远弱结构" : " · 等权 ρ")
      + (compareCopulas ? " · 对比高斯(实线) vs t-copula ν=2(虚线)"
        : (document.getElementById("n-copula").value === "t" ? " · t-copula" : " · 高斯")),
  };
  document.getElementById("s-note").textContent =
    "已读取「多证券」输入（" + scanSource.note + "）。请点「开始扫描 ρ」计算。";
}
document.getElementById("s-copy-multi").addEventListener("click", () => loadMultiScan(false, false));
document.getElementById("s-copy-multi-t").addEventListener("click", () => loadMultiScan(false, true));
document.getElementById("s-copy-multi-toeplitz").addEventListener("click", () => loadMultiScan(true, false));

/* init: 只准备输入区，不自动求解（点按钮才算） */
initMulti(3);
renderMultiTables();
document.getElementById("s-note").textContent =
  "请选择来源（两证券 / 多证券），再点「开始扫描 ρ」。";
["o-note", "t-note", "n-note"].forEach((id) => {
  const el = document.getElementById(id);
  if (el && !el.textContent) el.textContent = "请填写参数后点击计算按钮。";
});
