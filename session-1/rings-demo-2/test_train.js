// Prototype/test of ring-task training logic before porting to HTML artifact.
function zeros(rows, cols) { return Array.from({length: rows}, () => new Array(cols).fill(0)); }
function randInit(rows, cols, scale) {
  return Array.from({length: rows}, () => Array.from({length: cols}, () => (Math.random() * 2 - 1) * scale));
}
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }

function makeLayer(inDim, outDim, relu) {
  return {
    W: randInit(inDim, outDim, Math.sqrt(2 / inDim)),
    b: new Array(outDim).fill(0),
    relu,
    mW: zeros(inDim, outDim), vW: zeros(inDim, outDim),
    mb: new Array(outDim).fill(0), vb: new Array(outDim).fill(0),
  };
}

function forward(layers, x) {
  let a = x;
  const cache = [];
  for (const layer of layers) {
    const { W, b, relu } = layer;
    const inDim = a.length, outDim = b.length;
    const z = new Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let s = b[j];
      for (let i = 0; i < inDim; i++) s += a[i] * W[i][j];
      z[j] = s;
    }
    const out = relu ? z.map(v => Math.max(0, v)) : z.slice();
    cache.push({ input: a, z, relu });
    a = out;
  }
  return { output: a, cache };
}

function backward(layers, cache, dOutInit) {
  let dA = dOutInit;
  const grads = new Array(layers.length);
  for (let l = layers.length - 1; l >= 0; l--) {
    const layer = layers[l];
    const { input, z, relu } = cache[l];
    const outDim = z.length, inDim = input.length;
    const dz = new Array(outDim);
    for (let j = 0; j < outDim; j++) dz[j] = relu ? (z[j] > 0 ? dA[j] : 0) : dA[j];
    const dW = zeros(inDim, outDim);
    const db = new Array(outDim).fill(0);
    const dInput = new Array(inDim).fill(0);
    for (let j = 0; j < outDim; j++) {
      db[j] = dz[j];
      for (let i = 0; i < inDim; i++) {
        dW[i][j] = input[i] * dz[j];
        dInput[i] += layer.W[i][j] * dz[j];
      }
    }
    grads[l] = { dW, db };
    dA = dInput;
  }
  return grads;
}

function adamStep(layer, grad, t, lr) {
  const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
  const { W, b, mW, vW, mb, vb } = layer;
  for (let i = 0; i < W.length; i++) {
    for (let j = 0; j < W[0].length; j++) {
      mW[i][j] = beta1 * mW[i][j] + (1 - beta1) * grad.dW[i][j];
      vW[i][j] = beta2 * vW[i][j] + (1 - beta2) * grad.dW[i][j] ** 2;
      const mHat = mW[i][j] / (1 - beta1 ** t);
      const vHat = vW[i][j] / (1 - beta2 ** t);
      W[i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
    }
  }
  for (let j = 0; j < b.length; j++) {
    mb[j] = beta1 * mb[j] + (1 - beta1) * grad.db[j];
    vb[j] = beta2 * vb[j] + (1 - beta2) * grad.db[j] ** 2;
    const mHat = mb[j] / (1 - beta1 ** t);
    const vHat = vb[j] / (1 - beta2 ** t);
    b[j] -= lr * mHat / (Math.sqrt(vHat) + eps);
  }
}

function trainOneEpoch(layers, X, Y, t0, lr) {
  const idx = X.map((_, i) => i);
  for (let i = idx.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [idx[i], idx[j]] = [idx[j], idx[i]];
  }
  let t = t0;
  for (const i of idx) {
    t++;
    const { output, cache } = forward(layers, X[i]);
    const p = sigmoid(output[0]);
    const dOut = [p - Y[i]];
    const grads = backward(layers, cache, dOut);
    for (let l = 0; l < layers.length; l++) adamStep(layers[l], grads[l], t, lr);
  }
  return t;
}

function accuracy(layers, X, Y) {
  let correct = 0;
  for (let i = 0; i < X.length; i++) {
    const { output } = forward(layers, X[i]);
    const pred = output[0] > 0 ? 1 : 0;
    if (pred === Y[i]) correct++;
  }
  return correct / X.length;
}

function genRingData(nPerClass) {
  const X = [], Y = [];
  for (let i = 0; i < nPerClass; i++) {
    const r = Math.sqrt(Math.random()) * 0.32;
    const th = Math.random() * 2 * Math.PI;
    X.push([r * Math.cos(th) + (Math.random()-0.5)*0.05, r * Math.sin(th) + (Math.random()-0.5)*0.05]);
    Y.push(0);
  }
  for (let i = 0; i < nPerClass; i++) {
    const r = 0.62 + Math.random() * 0.3;
    const th = Math.random() * 2 * Math.PI;
    X.push([r * Math.cos(th) + (Math.random()-0.5)*0.05, r * Math.sin(th) + (Math.random()-0.5)*0.05]);
    Y.push(1);
  }
  return { X, Y };
}

// matrix helpers for collapse
function matMul(A, B) {
  const rows = A.length, k = B.length, cols = B[0].length;
  const out = zeros(rows, cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
      out[i][j] = s;
    }
  return out;
}
function vecMatMul(v, B) {
  const k = v.length, cols = B[0].length;
  const out = new Array(cols).fill(0);
  for (let j = 0; j < cols; j++) {
    let s = 0;
    for (let p = 0; p < k; p++) s += v[p] * B[p][j];
    out[j] = s;
  }
  return out;
}
function vecAdd(a, b) { return a.map((v, i) => v + b[i]); }

function collapseLinear(layers) {
  // layers must all be non-relu, affine composition
  let Wc = layers[0].W;
  let bc = layers[0].b;
  for (let l = 1; l < layers.length; l++) {
    Wc = matMul(Wc, layers[l].W);
    bc = vecAdd(vecMatMul(bc, layers[l].W), layers[l].b);
  }
  return { W: Wc, b: bc };
}

// ---- run experiment ----
const { X, Y } = genRingData(150);
const H = 6; // hidden width
const EPOCHS = 400;
const LR = 0.01;

const modelA = [makeLayer(2, 1, false)];
const modelB = [makeLayer(2, H, false), makeLayer(H, H, false), makeLayer(H, H, false), makeLayer(H, H, false), makeLayer(H, 1, false)];
const modelC = [makeLayer(2, H, true), makeLayer(H, H, true), makeLayer(H, H, true), makeLayer(H, H, true), makeLayer(H, 1, false)];

let t = 0;
for (let e = 0; e < EPOCHS; e++) t = trainOneEpoch(modelA, X, Y, t, LR);
console.log('Model A (1 linear layer) acc:', accuracy(modelA, X, Y).toFixed(3));

t = 0;
for (let e = 0; e < EPOCHS; e++) t = trainOneEpoch(modelB, X, Y, t, LR);
console.log('Model B (5 linear layers, no relu) acc:', accuracy(modelB, X, Y).toFixed(3));

t = 0;
for (let e = 0; e < EPOCHS; e++) t = trainOneEpoch(modelC, X, Y, t, LR);
console.log('Model C (5 layers + ReLU) acc:', accuracy(modelC, X, Y).toFixed(3));

// collapse test
const collapsed = collapseLinear(modelB);
console.log('Collapsed W (2x1):', collapsed.W);
console.log('Collapsed b:', collapsed.b);

// verify equivalence on a few points
console.log('\nVerification (full 5-layer forward vs collapsed matrix):');
for (let i = 0; i < 5; i++) {
  const x = X[i];
  const full = forward(modelB, x).output[0];
  const coll = vecMatMul(x, collapsed.W)[0] + collapsed.b[0];
  console.log(`x=[${x[0].toFixed(3)},${x[1].toFixed(3)}] full=${full.toFixed(6)} collapsed=${coll.toFixed(6)} diff=${Math.abs(full-coll).toExponential(2)}`);
}

// accuracy using collapsed matrix directly
let correctColl = 0;
for (let i = 0; i < X.length; i++) {
  const logit = vecMatMul(X[i], collapsed.W)[0] + collapsed.b[0];
  const pred = logit > 0 ? 1 : 0;
  if (pred === Y[i]) correctColl++;
}
console.log('Accuracy using collapsed single matrix:', (correctColl / X.length).toFixed(3));
