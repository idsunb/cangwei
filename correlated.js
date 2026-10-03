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
    { p: p00, sa: 0, sb: 0, ra: rAL, rb: rBL, returns: [rAL, rBL], label: "LL" },
    { p: p01, sa: 0, sb: 1, ra: rAL, rb: rBH, returns: [rAL, rBH], label: "LH" },
    { p: p10, sa: 1, sb: 0, ra: rAH, rb: rBL, returns: [rAH, rBL], label: "HL" },
    { p: p11c, sa: 1, sb: 1, ra: rAH, rb: rBH, returns: [rAH, rBH], label: "HH" },
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
 * 边际 → 分位阈值。states = [{p,r}, ...]，p 之和≈1。
 * 返回 cut[i] 使得 state j ⇔ cut[j] < U ≤ cut[j+1]
 */
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
      label: sa.join(","),
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
    rows.push({ p, states: sa, returns: ra, label: sa.join(",") });
  }
  return rows;
}

/* ===================== 增值熵优化 ===================== */

function portReturns(q, returnsRows, r0) {
  const R0 = 1 + r0;
  return returnsRows.map((row) => {
    // row.returns: 资产收益率
    const qs = q.reduce((a, b) => a + b, 0);
    const q0 = 1 - qs;
    let R = q0 * R0;
    for (let k = 0; k < q.length; k++) R += q[k] * (1 + row.returns[k]);
    return R - 1;
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

function optimizeCorrelated(rows, r0, allowShort, allowLev, maxMultiple = 1) {
  return solveCorrelated(rows, r0, allowShort, allowLev, maxMultiple, "grad");
}

/**
 * 统一求解入口。method:
 *   hill  坐标爬山
 *   grad  解析梯度精修（从零点）
 *   warm  §3.8 抛物线初值 + 梯度
 */
function solveCorrelated(rows, r0, allowShort, allowLev, maxMultiple = 1, method = "grad") {
  const W = rows.length;
  const N = rows[0].returns.length;
  let P = rows.map((r) => r.p);
  const s = P.reduce((a, b) => a + b, 0) || 1;
  P = P.map((p) => p / s);
  const R0 = 1 + r0;
  const lo = allowShort ? -maxMultiple : 0;
  const hi = allowLev ? maxMultiple : 1;
  const sumMax = allowLev ? maxMultiple : 1;

  function H_of(qa) {
    const qs = qa.reduce((a, b) => a + b, 0);
    const q0 = 1 - qs;
    if (!allowLev && q0 < -1e-12) return -Infinity;
    if (qs > sumMax + 1e-12) return -Infinity;
    if (qa.some((x) => x < -1e-12) && !allowShort) return -Infinity;
    let H = 0;
    for (let i = 0; i < W; i++) {
      let portR = q0 * R0;
      for (let k = 0; k < N; k++) portR += qa[k] * (1 + rows[i].returns[k]);
      if (portR <= 0) return -Infinity;
      H += P[i] * log2(portR);
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
    for (let i = 0; i < W; i++) {
      let portR = (1 - q.reduce((a, b) => a + b, 0)) * R0;
      for (let k = 0; k < N; k++) portR += q[k] * (1 + rows[i].returns[k]);
      if (portR <= 0) return null;
      for (let k = 0; k < N; k++) {
        const D = 1 + rows[i].returns[k] - R0;
        g[k] += (P[i] * D) / (portR * Math.LN2);
      }
    }
    return g;
  }
  function hillClimb(start) {
    let best = project(start ? start.slice() : new Array(N).fill(0));
    let bestH = H_of(best);
    // 单资产粗扫
    for (let k = 0; k < N; k++) {
      for (let t = 0; t <= 30; t++) {
        const q = new Array(N).fill(0);
        q[k] = lo + (hi - lo) * (t / 30);
        const h = H_of(q);
        if (h > bestH) {
          bestH = h;
          best = q.slice();
        }
      }
    }
    let step = 0.12;
    for (let it = 0; it < 120; it++) {
      let improved = false;
      for (let k = 0; k < N; k++) {
        for (const dir of [step, -step]) {
          const q = best.slice();
          q[k] = Math.max(lo, Math.min(hi, q[k] + dir));
          const qs = q.reduce((a, b) => a + b, 0);
          if (qs > sumMax + 1e-12) continue;
          const h = H_of(q);
          if (h > bestH + 1e-12) {
            bestH = h;
            best = q;
            improved = true;
          }
        }
      }
      if (!improved) {
        step *= 0.5;
        if (step < 1e-4) break;
      }
    }
    return { q: best, H: bestH };
  }
  function gradRefine(start) {
    let q = project(start ? start.slice() : new Array(N).fill(0));
    let best = q.slice();
    let bestH = H_of(q);
    let step = Math.max(0.5, hi - lo);
    let nit = 0;
    for (; nit < 400; nit++) {
      const g = grad(q);
      if (!g) break;
      const gmax = Math.max(...g.map(Math.abs), 1e-15);
      const dir = g.map((x) => x / gmax);
      let improved = false;
      for (const sc of [step, step * 0.5, step * 0.25, step * 0.125]) {
        const cand = project(q.map((x, k) => x + sc * dir[k]));
        const h = H_of(cand);
        if (h > bestH + 1e-14) {
          q = cand;
          best = cand.slice();
          bestH = h;
          improved = true;
          break;
        }
      }
      if (!improved) {
        step *= 0.5;
        if (step < 1e-5) break;
      }
    }
    return { q: best, H: bestH, nit };
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
  const q0 = 1 - qs;
  const portRs = [];
  for (let i = 0; i < W; i++) {
    let R = q0 * R0;
    for (let k = 0; k < N; k++) R += out.q[k] * (1 + rows[i].returns[k]);
    portRs.push(R - 1);
  }
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
  return {
    weights: out.q.slice(),
    cash: Math.max(0, q0),
    debt: Math.max(0, -q0),
    assetSum: qs,
    cashRaw: q0,
    H,
    rg: ok ? Math.exp(logRg) - 1 : -1,
    portRs,
    method: label,
    nit: out.nit != null ? out.nit : "—",
    warmstart: out.warmstart,
  };
}

/** 三方法对比 */
function compareCorrelated(rows, r0, allowShort, allowLev, maxMultiple = 1) {
  const methods = ["hill", "grad", "warm"];
  const results = methods.map((m) => solveCorrelated(rows, r0, allowShort, allowLev, maxMultiple, m));
  const ranked = results.slice().sort((a, b) => (b.H || -Infinity) - (a.H || -Infinity));
  return { results, ranking: ranked.map((r) => r.method), best: ranked[0].method };
}

/* ===================== UI 多证券 runMulti ===================== */

/* ===================== UI ===================== */

function fillJointTable(tableEl, rows, n) {
  let th = "<thead><tr><th>情景</th><th>P</th>";
  for (let k = 0; k < n; k++) th += `<th>r${k + 1}</th>`;
  th += "<th>组合收益(q*)</th></tr></thead><tbody>";
  rows.forEach((r, i) => {
    th += `<tr><td>${r.label}</td><td>${fmt(r.p, 4)}</td>`;
    r.returns.forEach((x) => (th += `<td>${pct(x, 2)}</td>`));
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

  const rows = joint2x2(pA, pB, rho, rAL, rAH, rBL, rBH);
  const res = optimizeCorrelated(rows, r0, sh, lev, M);
  rows.forEach((row, i) => {
    row.portR = res.portRs[i];
  });
  fillJointTable(document.getElementById("t-joint"), rows, 2);

  const out = document.getElementById("t-out");
  out.innerHTML = `
    <div class="metric"><div class="k">H*</div><div class="v">${fmt(res.H, 4)}</div></div>
    <div class="metric"><div class="k">几何平均</div><div class="v">${pct(res.rg)}</div></div>
    <div class="metric"><div class="k">现金</div><div class="v">${pct(res.cash)}</div></div>
    <div class="metric"><div class="k">标的合计</div><div class="v">${pct(res.assetSum)}</div></div>
    <div class="metric"><div class="k">负债</div><div class="v ${res.debt > 1e-9 ? "warn" : ""}">${pct(res.debt)}</div></div>
    <div class="metric"><div class="k">ρ</div><div class="v">${fmt(rho, 2)}</div></div>`;

  const wt = document.getElementById("t-weights");
  wt.innerHTML =
    `<thead><tr><th>项目</th><th>q*</th><th>占比</th></tr></thead><tbody>
     <tr><td>现金</td><td>${fmt(res.cash, 4)}</td><td>${pct(res.cash)}</td></tr>` +
    (res.debt > 1e-9
      ? `<tr><td>负债</td><td>${fmt(res.debt, 4)}</td><td>${pct(res.debt)}</td></tr>`
      : "") +
    `<tr><td>证券 A</td><td>${fmt(res.weights[0], 4)}</td><td>${pct(res.weights[0])}</td></tr>
     <tr><td>证券 B</td><td>${fmt(res.weights[1], 4)}</td><td>${pct(res.weights[1])}</td></tr>
     </tbody>`;

  // ρ 有效范围提示
  const sA = Math.sqrt(pA * (1 - pA) * pB * (1 - pB));
  const p11raw = pA * pB + rho * sA;
  const clipped = p11raw !== rows[3].p * (rows.reduce((a, b) => a + b.p, 0) || 1);
  document.getElementById("t-note").innerHTML =
    `P(HH)=${fmt(rows[3].p, 4)}；相关使「同涨同跌」概率 ${
      rho > 0 ? "升高" : rho < 0 ? "降低" : "不变"
    }，几何平均与最优仓位随之变化。` +
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

function renderMultiTables() {
  const n = margState.length;
  const mb = document.getElementById("n-marg");
  let mh = "";
  margState.forEach((m, k) => {
    const sumP = m.states.reduce((a, b) => a + (parseFloat(b.p) || 0), 0);
    mh += `<div class="card" style="margin-bottom:10px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
        <strong>${m.name}（${m.states.length} 态，Σp=${fmt(sumP, 4)}）</strong>
        <span>
          <button class="ghost" data-madd="${k}">＋ 状态</button>
          ${m.states.length > 2 ? `<button class="ghost" data-mdelst="${k}">－ 状态</button>` : ""}
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
    });
  });
  mb.querySelectorAll("[data-madd]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const k = +btn.getAttribute("data-madd");
      margState[k].states.push({ p: 0, r: 0.1 });
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
    const corr = projectCorrPSD(corrState, n);
    corrState = corr.map((r) => r.slice());
    renderMultiTables();

    const joint = jointFromCopula(margState, corr, 8000, 42);
    const ind = jointIndependent(margState);
    const res = solveCorrelated(joint, r0, sh, lev, M, method);
    const resInd = solveCorrelated(ind, r0, sh, lev, M, method);
    joint.forEach((row, i) => {
      row.portR = res.portRs[i];
    });

    const dims = margState.map((m) => m.states.length);
    document.getElementById("n-out").innerHTML = `
      <div class="metric"><div class="k">H*（相关）</div><div class="v">${fmt(res.H, 4)}</div></div>
      <div class="metric"><div class="k">H*（独立对照）</div><div class="v">${fmt(resInd.H, 4)}</div></div>
      <div class="metric"><div class="k">几何平均</div><div class="v">${pct(res.rg)}</div></div>
      <div class="metric"><div class="k">现金</div><div class="v">${pct(res.cash)}</div></div>
      <div class="metric"><div class="k">标的合计</div><div class="v">${pct(res.assetSum)}</div></div>
      <div class="metric"><div class="k">负债</div><div class="v ${res.debt > 1e-9 ? "warn" : ""}">${pct(res.debt)}</div></div>
      <div class="metric"><div class="k">方法</div><div class="v" style="font-size:14px">${res.method}</div></div>`;

    let th = "<thead><tr><th>证券</th><th>状态数</th><th>q*（相关）</th><th>q*（独立）</th></tr></thead><tbody>";
    margState.forEach((m, k) => {
      th += `<tr><td>${m.name}</td><td>${m.states.length}</td><td>${fmt(res.weights[k], 4)}</td><td>${fmt(resInd.weights[k], 4)}</td></tr>`;
    });
    th += `<tr><td>现金</td><td></td><td>${fmt(res.cash, 4)}</td><td>${fmt(resInd.cash, 4)}</td></tr>`;
    th += `<tr><td>负债</td><td></td><td>${fmt(res.debt, 4)}</td><td>${fmt(resInd.debt, 4)}</td></tr>`;
    th += "</tbody>";
    document.getElementById("n-weights").innerHTML = th;

    const top = joint.slice(0, 16);
    fillJointTable(document.getElementById("n-joint"), top, n);
    document.getElementById("n-compare-wrap").style.display = "none";

    const dH = res.H - resInd.H;
    document.getElementById("n-note").innerHTML =
      `状态数 ${dims.join("×")} = ${dims.reduce((a, b) => a * b, 1)} 格联合；相关 copula 有效情景 ${joint.length}。` +
      `方法：<b>${res.method}</b>。相对独立：ΔH = ${fmt(dH, 4)} bit` +
      (dH < -1e-4
        ? " —— 正相关削弱分散红利。"
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
    const corr = projectCorrPSD(corrState, n);
    corrState = corr.map((r) => r.slice());
    renderMultiTables();
    const joint = jointFromCopula(margState, corr, 8000, 42);
    const cmp = compareCorrelated(joint, r0, sh, lev, M);

    let th = "<thead><tr><th>方法</th><th>H (bit)</th>";
    margState.forEach((m) => (th += `<th>${m.name}</th>`));
    th += "<th>现金</th><th>负债</th><th>ΔH</th></tr></thead><tbody>";
    const bestH = cmp.results[0].H;
    const sorted = cmp.results.slice().sort((a, b) => (b.H || -Infinity) - (a.H || -Infinity));
    sorted.forEach((r) => {
      th += `<tr><td>${r.method}</td><td>${fmt(r.H, 6)}</td>`;
      r.weights.forEach((q) => (th += `<td>${fmt(q, 4)}</td>`));
      th += `<td>${fmt(r.cash, 4)}</td><td>${fmt(r.debt, 4)}</td>`;
      th += `<td>${fmt(bestH - r.H, 6)}</td></tr>`;
    });
    th += "</tbody>";
    document.getElementById("n-compare-table").innerHTML = th;
    document.getElementById("n-compare-wrap").style.display = "block";
    document.getElementById("n-note").innerHTML =
      `凸问题下三种方法应接近或相同；最优为 <b>${sorted[0].method}</b>（H=${fmt(sorted[0].H, 6)}）。`;
    // 同步展示最优
    const best = sorted[0];
    document.getElementById("n-out").innerHTML = `
      <div class="metric"><div class="k">H*（对比最优）</div><div class="v">${fmt(best.H, 4)}</div></div>
      <div class="metric"><div class="k">几何平均</div><div class="v">${pct(best.rg)}</div></div>
      <div class="metric"><div class="k">现金</div><div class="v">${pct(best.cash)}</div></div>
      <div class="metric"><div class="k">标的合计</div><div class="v">${pct(best.assetSum)}</div></div>
      <div class="metric"><div class="k">负债</div><div class="v ${best.debt > 1e-9 ? "warn" : ""}">${pct(best.debt)}</div></div>
      <div class="metric"><div class="k">方法</div><div class="v" style="font-size:14px">${best.method}</div></div>`;
    let th2 = "<thead><tr><th>证券</th><th>q*</th></tr></thead><tbody>";
    margState.forEach((m, k) => {
      th2 += `<tr><td>${m.name}</td><td>${fmt(best.weights[k], 4)}</td></tr>`;
    });
    th2 += `<tr><td>现金</td><td>${fmt(best.cash, 4)}</td></tr>`;
    th2 += `<tr><td>负债</td><td>${fmt(best.debt, 4)}</td></tr></tbody>`;
    document.getElementById("n-weights").innerHTML = th2;
  } catch (e) {
    document.getElementById("n-note").textContent = "错误: " + e.message;
  }
}

/* ---- ρ 扫描 ---- */
function runRhoScan() {
  const pA = num("t-pA"), pB = num("t-pB");
  const rAL = num("t-rAL"), rAH = num("t-rAH");
  const rBL = num("t-rBL"), rBH = num("t-rBH");
  const r0 = num("t-r0") || 0;
  const sh = document.getElementById("t-short").checked;
  const lev = document.getElementById("t-lev").checked;
  const M = num("t-M") || 1;

  const canvas = document.getElementById("s-canvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const pad = { l: 48, r: 16, t: 18, b: 36 };
  ctx.clearRect(0, 0, W, H);

  const data = [];
  for (let i = 0; i <= 40; i++) {
    const rho = -1 + (2 * i) / 40;
    const rows = joint2x2(pA, pB, rho, rAL, rAH, rBL, rBH);
    const res = optimizeCorrelated(rows, r0, sh, lev, M);
    data.push({ rho, H: res.H, qA: res.weights[0], qB: res.weights[1], rg: res.rg });
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
  // H
  ctx.strokeStyle = "#0f5c4c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((d, i) => {
    const px = x(d.rho), py = y(d.H);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
  // qA
  ctx.strokeStyle = "#8a3b12";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  data.forEach((d, i) => {
    const px = x(d.rho), py = y(d.qA * (yMax - yMin) * 0.3 + yMin);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.fillStyle = "#6b6560";
  ctx.fillText("ρ=-1", pad.l, H - pad.b + 16);
  ctx.fillText("ρ=1", W - pad.r - 24, H - pad.b + 16);
  ctx.fillStyle = "#0f5c4c";
  ctx.fillText("H*(ρ)", pad.l + 4, pad.t + 12);
  ctx.fillStyle = "#8a3b12";
  ctx.fillText("qA*(ρ)（缩放）", pad.l + 4, pad.t + 26);

  let th = "<thead><tr><th>ρ</th><th>H*</th><th>qA*</th><th>qB*</th><th>r_g</th></tr></thead><tbody>";
  for (let i = 0; i < data.length; i += 4) {
    const d = data[i];
    th += `<tr><td>${fmt(d.rho, 2)}</td><td>${fmt(d.H, 4)}</td><td>${fmt(d.qA, 4)}</td><td>${fmt(d.qB, 4)}</td><td>${pct(d.rg)}</td></tr>`;
  }
  th += "</tbody>";
  document.getElementById("s-table").innerHTML = th;
  const best = data.reduce((a, b) => (b.H > a.H ? b : a), data[0]);
  const worst = data.reduce((a, b) => (b.H < a.H ? b : a), data[0]);
  document.getElementById("s-note").textContent =
    `H* 在 ρ=${fmt(best.rho, 2)} 达到最大 ${fmt(best.H, 4)}；` +
    `在 ρ=${fmt(worst.rho, 2)} 最小 ${fmt(worst.H, 4)}。` +
    `一般：|ρ| 越低，分散越有效，H* 越高。`;
}

/* ---- tabs ---- */
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.getAttribute("data-tab")).classList.add("active");
    if (tab.getAttribute("data-tab") === "rho") runRhoScan();
    if (tab.getAttribute("data-tab") === "two") runTwo();
    if (tab.getAttribute("data-tab") === "multi") runMulti();
  });
});

document.getElementById("t-run").addEventListener("click", runTwo);
["t-pA", "t-pB", "t-rAL", "t-rAH", "t-rBL", "t-rBH", "t-rho", "t-r0", "t-M"].forEach((id) => {
  document.getElementById(id).addEventListener("change", runTwo);
});
document.getElementById("t-lev").addEventListener("change", runTwo);
document.getElementById("t-short").addEventListener("change", runTwo);
document.querySelectorAll("[data-preset]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const p = btn.getAttribute("data-preset");
    document.getElementById("t-rho").value = p === "ind" ? "0" : p === "pos" ? "0.8" : "-0.8";
    runTwo();
  });
});

document.getElementById("n-run").addEventListener("click", runMulti);
document.getElementById("n-compare").addEventListener("click", runMultiCompare);
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
document.getElementById("s-copy-two").addEventListener("click", runRhoScan);

/* init */
initMulti(3);
runTwo();
runMulti();
