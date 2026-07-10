/**
 * model.js — data generation + two tiny models (linear vs. 1-hidden-layer ReLU MLP)
 *
 * Pure math, no DOM dependency, so it can be unit-tested in Node and then
 * reused as-is in the browser (loaded as a plain <script>, exposes window.RD).
 */
(function (global) {
  'use strict';

  // ---------------- Seeded RNG (mulberry32) + Gaussian (Box-Muller) ----------------
  function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
      s |= 0;
      s = (s + 0x6d2b79f5) | 0;
      let t = Math.imul(s ^ (s >>> 15), 1 | s);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function makeGaussian(rand) {
    let spare = null;
    return function gaussian() {
      if (spare !== null) {
        const s = spare;
        spare = null;
        return s;
      }
      let u = 0, v = 0;
      while (u === 0) u = rand();
      while (v === 0) v = rand();
      const mag = Math.sqrt(-2.0 * Math.log(u));
      spare = mag * Math.sin(2.0 * Math.PI * v);
      return mag * Math.cos(2.0 * Math.PI * v);
    };
  }

  // ---------------- Data: two concentric noisy rings ----------------
  // inner ring -> label 0, outer ring -> label 1. Not linearly separable.
  function generateRings(opts) {
    opts = opts || {};
    const nPerClass = opts.nPerClass || 150;
    const innerR = opts.innerR != null ? opts.innerR : 1.2;
    const outerR = opts.outerR != null ? opts.outerR : 2.8;
    const noise = opts.noise != null ? opts.noise : 0.18;
    const seed = opts.seed != null ? opts.seed : 42;

    const rand = mulberry32(seed);
    const gauss = makeGaussian(rand);
    const points = [];

    for (let i = 0; i < nPerClass; i++) {
      const angle = rand() * Math.PI * 2;
      const r = innerR + gauss() * noise;
      points.push({ x: r * Math.cos(angle), y: r * Math.sin(angle), label: 0 });
    }
    for (let i = 0; i < nPerClass; i++) {
      const angle = rand() * Math.PI * 2;
      const r = outerR + gauss() * noise;
      points.push({ x: r * Math.cos(angle), y: r * Math.sin(angle), label: 1 });
    }
    // Fisher-Yates shuffle (deterministic, same seed stream)
    for (let i = points.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      const tmp = points[i];
      points[i] = points[j];
      points[j] = tmp;
    }
    return points;
  }

  // ---------------- Math ----------------
  function sigmoid(z) {
    if (z >= 0) {
      const e = Math.exp(-z);
      return 1 / (1 + e);
    }
    const e = Math.exp(z);
    return e / (1 + e);
  }

  // ---------------- Adam optimizer (operates on a flat Float64Array) ----------------
  function createAdam(n, opts) {
    opts = opts || {};
    return {
      m: new Float64Array(n),
      v: new Float64Array(n),
      t: 0,
      lr: opts.lr || 0.05,
      b1: opts.b1 || 0.9,
      b2: opts.b2 || 0.999,
      eps: opts.eps || 1e-8,
    };
  }
  function adamStep(state, params, grads) {
    state.t++;
    const { m, v, b1, b2, eps, lr, t } = state;
    const b1t = 1 - Math.pow(b1, t);
    const b2t = 1 - Math.pow(b2, t);
    for (let i = 0; i < params.length; i++) {
      m[i] = b1 * m[i] + (1 - b1) * grads[i];
      v[i] = b2 * v[i] + (1 - b2) * grads[i] * grads[i];
      const mHat = m[i] / b1t;
      const vHat = v[i] / b2t;
      params[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
    }
  }

  // ---------------- Model A: single linear layer + sigmoid (logistic regression) ----------------
  // params = [w1, w2, b]   z = w1*x + w2*y + b   p = sigmoid(z)
  function createLinearModel(seed) {
    const rand = mulberry32(seed || 7);
    const params = new Float64Array([(rand() - 0.5) * 0.2, (rand() - 0.5) * 0.2, 0]);
    return { params, adam: createAdam(3, { lr: 0.1 }) };
  }
  function linearForward(params, x, y) {
    const z = params[0] * x + params[1] * y + params[2];
    return sigmoid(z);
  }
  function linearTrainStep(model, points) {
    const grads = new Float64Array(3);
    let loss = 0;
    const n = points.length;
    const eps = 1e-9;
    for (const p of points) {
      const pred = linearForward(model.params, p.x, p.y);
      const dz = pred - p.label; // dL/dz for sigmoid+BCE
      grads[0] += dz * p.x;
      grads[1] += dz * p.y;
      grads[2] += dz;
      loss += -(p.label * Math.log(pred + eps) + (1 - p.label) * Math.log(1 - pred + eps));
    }
    for (let i = 0; i < 3; i++) grads[i] /= n;
    adamStep(model.adam, model.params, grads);
    return loss / n;
  }

  // ---------------- Model B: 2 -> H (ReLU) -> 1 (sigmoid) ----------------
  // params layout: [ W1(2*H) | b1(H) | W2(H) | b2(1) ]
  function createMLPModel(hidden, seed) {
    const H = hidden || 8;
    const rand = mulberry32(seed || 11);
    const gauss = makeGaussian(rand);
    const n = 4 * H + 1;
    const params = new Float64Array(n);
    const w2Offset = 3 * H;
    const b2Offset = 4 * H;
    for (let j = 0; j < H; j++) {
      params[j * 2 + 0] = gauss() * 0.7; // He-ish init for 2 inputs
      params[j * 2 + 1] = gauss() * 0.7;
      params[2 * H + j] = 0; // b1
      params[w2Offset + j] = gauss() * 0.3;
    }
    params[b2Offset] = 0;
    return { params, H, adam: createAdam(n, { lr: 0.05 }) };
  }
  function mlpForward(params, H, x, y) {
    const w2Offset = 3 * H;
    const b2Offset = 4 * H;
    let z2 = params[b2Offset];
    for (let j = 0; j < H; j++) {
      const w1 = params[j * 2 + 0];
      const w2 = params[j * 2 + 1];
      const b1 = params[2 * H + j];
      const z1 = w1 * x + w2 * y + b1;
      const h = z1 > 0 ? z1 : 0;
      z2 += params[w2Offset + j] * h;
    }
    return sigmoid(z2);
  }
  function mlpTrainStep(model, points) {
    const { params, H } = model;
    const grads = new Float64Array(params.length);
    const w2Offset = 3 * H;
    const b2Offset = 4 * H;
    const z1 = new Float64Array(H);
    const h = new Float64Array(H);
    let loss = 0;
    const n = points.length;
    const eps = 1e-9;

    for (const pt of points) {
      for (let j = 0; j < H; j++) {
        const w1 = params[j * 2 + 0];
        const w2 = params[j * 2 + 1];
        const b1 = params[2 * H + j];
        const z = w1 * pt.x + w2 * pt.y + b1;
        z1[j] = z;
        h[j] = z > 0 ? z : 0;
      }
      let z2 = params[b2Offset];
      for (let j = 0; j < H; j++) z2 += params[w2Offset + j] * h[j];
      const p = sigmoid(z2);
      loss += -(pt.label * Math.log(p + eps) + (1 - pt.label) * Math.log(1 - p + eps));

      const dz2 = p - pt.label;
      grads[b2Offset] += dz2;
      for (let j = 0; j < H; j++) {
        grads[w2Offset + j] += dz2 * h[j];
        const dh = dz2 * params[w2Offset + j];
        const dz1 = z1[j] > 0 ? dh : 0;
        grads[j * 2 + 0] += dz1 * pt.x;
        grads[j * 2 + 1] += dz1 * pt.y;
        grads[2 * H + j] += dz1;
      }
    }
    for (let i = 0; i < grads.length; i++) grads[i] /= n;
    adamStep(model.adam, params, grads);
    return loss / n;
  }

  function accuracy(predictFn, points) {
    let correct = 0;
    for (const p of points) {
      const pred = predictFn(p.x, p.y);
      const cls = pred >= 0.5 ? 1 : 0;
      if (cls === p.label) correct++;
    }
    return correct / points.length;
  }

  const api = {
    mulberry32,
    makeGaussian,
    generateRings,
    sigmoid,
    createAdam,
    adamStep,
    createLinearModel,
    linearForward,
    linearTrainStep,
    createMLPModel,
    mlpForward,
    mlpTrainStep,
    accuracy,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
  global.RD = api;
})(typeof window !== 'undefined' ? window : globalThis);
