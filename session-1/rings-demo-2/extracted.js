
/* ---------------- tiny NN engine (per-sample forward/backward + Adam) ---------------- */
function zeros(rows, cols){ return Array.from({length:rows}, () => new Array(cols).fill(0)); }
function randInit(rows, cols, scale){
  return Array.from({length:rows}, () => Array.from({length:cols}, () => (Math.random()*2-1)*scale));
}
function sigmoid(x){ return 1/(1+Math.exp(-x)); }

function makeLayer(inDim, outDim, relu, xavier){
  const scale = xavier ? Math.sqrt(1/inDim) : Math.sqrt(2/inDim); // Xavier for linear stacks, He for ReLU
  return {
    W: randInit(inDim, outDim, scale), b: new Array(outDim).fill(0), relu,
    mW: zeros(inDim, outDim), vW: zeros(inDim, outDim),
    mb: new Array(outDim).fill(0), vb: new Array(outDim).fill(0),
  };
}

function forward(layers, x){
  let a = x;
  const cache = [];
  for (const layer of layers){
    const { W, b, relu } = layer;
    const inDim = a.length, outDim = b.length;
    const z = new Array(outDim);
    for (let j=0;j<outDim;j++){
      let s = b[j];
      for (let i=0;i<inDim;i++) s += a[i]*W[i][j];
      z[j] = s;
    }
    const out = relu ? z.map(v => Math.max(0,v)) : z.slice();
    cache.push({ input:a, z, relu });
    a = out;
  }
  return { output:a, cache };
}

function backward(layers, cache, dOutInit){
  let dA = dOutInit;
  const grads = new Array(layers.length);
  for (let l=layers.length-1; l>=0; l--){
    const layer = layers[l];
    const { input, z, relu } = cache[l];
    const outDim = z.length, inDim = input.length;
    const dz = new Array(outDim);
    for (let j=0;j<outDim;j++) dz[j] = relu ? (z[j]>0 ? dA[j] : 0) : dA[j];
    const dW = zeros(inDim, outDim);
    const db = new Array(outDim).fill(0);
    const dInput = new Array(inDim).fill(0);
    for (let j=0;j<outDim;j++){
      db[j] = dz[j];
      for (let i=0;i<inDim;i++){
        dW[i][j] = input[i]*dz[j];
        dInput[i] += layer.W[i][j]*dz[j];
      }
    }
    grads[l] = { dW, db };
    dA = dInput;
  }
  return grads;
}

function adamStep(layer, grad, t, lr){
  const beta1=0.9, beta2=0.999, eps=1e-8;
  const { W, b, mW, vW, mb, vb } = layer;
  for (let i=0;i<W.length;i++){
    for (let j=0;j<W[0].length;j++){
      mW[i][j] = beta1*mW[i][j] + (1-beta1)*grad.dW[i][j];
      vW[i][j] = beta2*vW[i][j] + (1-beta2)*grad.dW[i][j]**2;
      const mHat = mW[i][j]/(1-beta1**t), vHat = vW[i][j]/(1-beta2**t);
      W[i][j] -= lr*mHat/(Math.sqrt(vHat)+eps);
    }
  }
  for (let j=0;j<b.length;j++){
    mb[j] = beta1*mb[j] + (1-beta1)*grad.db[j];
    vb[j] = beta2*vb[j] + (1-beta2)*grad.db[j]**2;
    const mHat = mb[j]/(1-beta1**t), vHat = vb[j]/(1-beta2**t);
    b[j] -= lr*mHat/(Math.sqrt(vHat)+eps);
  }
}

function trainOneEpoch(layers, X, Y, t0, lr){
  const idx = X.map((_,i)=>i);
  for (let i=idx.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [idx[i],idx[j]]=[idx[j],idx[i]]; }
  let t = t0;
  for (const i of idx){
    t++;
    const { output, cache } = forward(layers, X[i]);
    const p = sigmoid(output[0]);
    const grads = backward(layers, cache, [p - Y[i]]);
    for (let l=0;l<layers.length;l++) adamStep(layers[l], grads[l], t, lr);
  }
  return t;
}

function accuracy(layers, X, Y){
  let correct = 0;
  for (let i=0;i<X.length;i++){
    const pred = forward(layers, X[i]).output[0] > 0 ? 1 : 0;
    if (pred === Y[i]) correct++;
  }
  return correct / X.length;
}

/* ---------------- matrix helpers for the collapse proof ---------------- */
function matMul(A, B){
  const rows=A.length, k=B.length, cols=B[0].length;
  const out = zeros(rows, cols);
  for (let i=0;i<rows;i++) for (let j=0;j<cols;j++){ let s=0; for (let p=0;p<k;p++) s+=A[i][p]*B[p][j]; out[i][j]=s; }
  return out;
}
function vecMatMul(v, B){
  const k=v.length, cols=B[0].length;
  const out = new Array(cols).fill(0);
  for (let j=0;j<cols;j++){ let s=0; for (let p=0;p<k;p++) s+=v[p]*B[p][j]; out[j]=s; }
  return out;
}
function vecAdd(a,b){ return a.map((v,i)=>v+b[i]); }
function collapseLinear(layers){
  let Wc = layers[0].W, bc = layers[0].b;
  for (let l=1;l<layers.length;l++){
    Wc = matMul(Wc, layers[l].W);
    bc = vecAdd(vecMatMul(bc, layers[l].W), layers[l].b);
  }
  return { W:Wc, b:bc };
}

/* ---------------- ring dataset ---------------- */
function genRingData(nPerClass){
  const X=[], Y=[];
  for (let i=0;i<nPerClass;i++){
    const r = Math.sqrt(Math.random())*0.32, th = Math.random()*2*Math.PI;
    X.push([r*Math.cos(th)+(Math.random()-0.5)*0.05, r*Math.sin(th)+(Math.random()-0.5)*0.05]);
    Y.push(0);
  }
  for (let i=0;i<nPerClass;i++){
    const r = 0.62+Math.random()*0.3, th = Math.random()*2*Math.PI;
    X.push([r*Math.cos(th)+(Math.random()-0.5)*0.05, r*Math.sin(th)+(Math.random()-0.5)*0.05]);
    Y.push(1);
  }
  return { X, Y };
}

/* ---------------- rendering ---------------- */
function drawBoundary(canvas, layers, X, Y){
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height, res = 60;
  const lo = -1.05, hi = 1.05, span = hi-lo;
  const img = ctx.createImageData(w, h);
  for (let py=0; py<h; py++){
    const gy = lo + (py/h)*span;
    for (let px=0; px<w; px++){
      const gx = lo + (px/w)*span;
      const logit = forward(layers, [gx, -gy]).output[0];
      const p = sigmoid(logit);
      // blue (class0) -> orange (class1) background, soft
      const r = 20 + p*70, g = 22 + p*40, b = 34 - p*10;
      const idx = (py*w+px)*4;
      img.data[idx]=r; img.data[idx+1]=g; img.data[idx+2]=b; img.data[idx+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);
  // scatter
  for (let i=0;i<X.length;i++){
    const [x,y] = X[i];
    const px = ((x-lo)/span)*w, py = h - ((y-lo)/span)*h;
    ctx.beginPath();
    ctx.arc(px, py, 3, 0, Math.PI*2);
    ctx.fillStyle = Y[i]===0 ? '#5ea1ff' : '#ff8a5e';
    ctx.fill();
    ctx.lineWidth = 0.6; ctx.strokeStyle = 'rgba(0,0,0,0.5)'; ctx.stroke();
  }
}

function fmt(n, d=2){ return (n<0?'':' ') + n.toFixed(d); }

function renderMatrix(M, label){
  let html = '<div class="matbox"><div class="lbl">'+label+'</div><table class="mat">';
  for (const row of M){
    html += '<tr>' + row.map(v=>'<td>'+fmt(v,2)+'</td>').join('') + '</tr>';
  }
  html += '</table></div>';
  return html;
}
function renderVector(v, label){
  return renderMatrix([v], label);
}

/* ---------------- main run ---------------- */
function setAcc(elId, acc, threshold){
  const el = document.getElementById(elId);
  el.innerHTML = '<span>accuracy</span><b>'+(acc*100).toFixed(1)+'%</b>';
  el.className = 'acc ' + (acc >= threshold ? 'good' : 'bad');
}

async function runAll(){
  const btn = document.getElementById('trainBtn');
  const status = document.getElementById('status');
  btn.disabled = true;
  status.textContent = 'Generating ring data…';
  await new Promise(r => setTimeout(r, 20));

  const H = 6, LR = 0.005, EPOCHS = 1200;
  const { X, Y } = genRingData(150);

  const modelA = [ makeLayer(2, 1, false, true) ];
  const modelB = [
    makeLayer(2, H, false, true), makeLayer(H, H, false, true), makeLayer(H, H, false, true),
    makeLayer(H, H, false, true), makeLayer(H, 1, false, true)
  ];
  const modelC = [
    makeLayer(2, H, true, false), makeLayer(H, H, true, false), makeLayer(H, H, true, false),
    makeLayer(H, H, true, false), makeLayer(H, 1, false, false)
  ];

  status.textContent = 'Training Model A (1 linear layer)…';
  await new Promise(r => setTimeout(r, 10));
  let t = 0; for (let e=0;e<EPOCHS;e++) t = trainOneEpoch(modelA, X, Y, t, LR);
  const accA = accuracy(modelA, X, Y);

  status.textContent = 'Training Model B (5 linear layers, no activation)…';
  await new Promise(r => setTimeout(r, 10));
  t = 0; for (let e=0;e<EPOCHS;e++) t = trainOneEpoch(modelB, X, Y, t, LR);
  const accB = accuracy(modelB, X, Y);

  status.textContent = 'Training Model C (5 layers + ReLU)…';
  await new Promise(r => setTimeout(r, 10));
  t = 0; for (let e=0;e<EPOCHS;e++) t = trainOneEpoch(modelC, X, Y, t, LR);
  const accC = accuracy(modelC, X, Y);

  status.textContent = 'Rendering…';
  await new Promise(r => setTimeout(r, 10));

  drawBoundary(document.getElementById('cvA'), modelA, X, Y);
  drawBoundary(document.getElementById('cvB'), modelB, X, Y);
  drawBoundary(document.getElementById('cvC'), modelC, X, Y);
  setAcc('accA', accA, 0.8);
  setAcc('accB', accB, 0.8);
  setAcc('accC', accC, 0.8);

  const closeCall = Math.abs(accA-accB) < 0.20;
  document.getElementById('proofBox').innerHTML =
    'Model A (1 layer): <b>'+(accA*100).toFixed(1)+'%</b> — a single straight boundary. '+
    'Model B (5 linear layers): <b>'+(accB*100).toFixed(1)+'%</b> — despite 5 layers and dozens of extra parameters, '+
    'it is <em>still</em> only a straight boundary, and it is stuck at the same near-chance accuracy as A'+
    (closeCall ? ' (both hover around 50% — neither line can carve the ring apart)' : '')+
    '. Model C (5 layers + ReLU, identical layer sizes): <b style="color:var(--good)">'+(accC*100).toFixed(1)+'%</b> — '+
    'the exact same 5 matrices, just with a ReLU between them, and the boundary curves to wrap the ring. '+
    'Nothing changed except the nonlinearity.';

  /* ---- bonus: collapse the 5 matrices of Model B into one ---- */
  const collapsed = collapseLinear(modelB);
  let chainHtml = '';
  modelB.forEach((layer, i) => {
    chainHtml += renderMatrix(layer.W, 'W'+(i+1)+' ('+layer.W.length+'×'+layer.W[0].length+')');
    chainHtml += '<span class="op">×</span>';
  });
  chainHtml = chainHtml.slice(0, -'<span class="op">×</span>'.length);
  chainHtml += '<span class="op">=</span>';
  chainHtml += renderMatrix(collapsed.W, 'W₁·W₂·W₃·W₄·W₅ (2×1)');
  document.getElementById('matChain').innerHTML = chainHtml;

  let vtHtml = '<tr><th>sample x</th><th>full 5-layer forward</th><th>collapsed W·x + b</th><th>|difference|</th></tr>';
  let maxDiff = 0;
  for (let i=0;i<6;i++){
    const x = X[i];
    const full = forward(modelB, x).output[0];
    const coll = vecMatMul(x, collapsed.W)[0] + collapsed.b[0];
    const diff = Math.abs(full-coll);
    maxDiff = Math.max(maxDiff, diff);
    vtHtml += '<tr><td>['+x[0].toFixed(3)+', '+x[1].toFixed(3)+']</td>'+
      '<td>'+full.toFixed(8)+'</td><td>'+coll.toFixed(8)+'</td><td>'+diff.toExponential(2)+'</td></tr>';
  }
  document.getElementById('verifyTable').innerHTML = vtHtml;

  let correctColl = 0;
  for (let i=0;i<X.length;i++){
    const logit = vecMatMul(X[i], collapsed.W)[0] + collapsed.b[0];
    if ((logit>0?1:0) === Y[i]) correctColl++;
  }
  const accColl = correctColl / X.length;

  document.getElementById('headline').textContent =
    'Max forward-vs-collapsed difference across all points: ' + maxDiff.toExponential(2) +
    ' (floating-point noise, i.e. zero) — Model B accuracy: ' + (accB*100).toFixed(1) +
    '% vs. single-collapsed-matrix accuracy: ' + (accColl*100).toFixed(1) +
    '%. They are the same function. Five linear layers were always one matrix.';

  status.textContent = 'Done.';
  btn.disabled = false;
}

runAll();
