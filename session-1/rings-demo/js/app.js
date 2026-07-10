/**
 * app.js — UI glue: canvas rendering + animation loop for the rings demo.
 * Depends on window.RD from model.js.
 */
(function () {
  'use strict';

  const PLOT_RANGE = 3.6; // plane spans [-PLOT_RANGE, PLOT_RANGE] on both axes
  const GRID = 90; // resolution of the offscreen probability heatmap
  const MAX_EPOCHS = 4000; // auto-stop training once both models have settled

  const els = {
    canvasLinear: document.getElementById('canvasLinear'),
    canvasMLP: document.getElementById('canvasMLP'),
    linAcc: document.getElementById('linAcc'),
    linLoss: document.getElementById('linLoss'),
    mlpAcc: document.getElementById('mlpAcc'),
    mlpLoss: document.getElementById('mlpLoss'),
    statusLine: document.getElementById('statusLine'),
    toggleBtn: document.getElementById('toggleBtn'),
    resetBtn: document.getElementById('resetBtn'),
    newDataBtn: document.getElementById('newDataBtn'),
    speedRange: document.getElementById('speedRange'),
  };

  // Offscreen low-res canvases used to build the smooth heatmap cheaply.
  const heatLinear = document.createElement('canvas');
  const heatMLP = document.createElement('canvas');
  heatLinear.width = heatLinear.height = GRID;
  heatMLP.width = heatMLP.height = GRID;

  let points = [];
  let dataSeed = 42;
  let linear, mlp;
  let epoch = 0;
  let running = true;
  let rafId = null;

  function toPixel(v) {
    // maps a data coordinate in [-PLOT_RANGE, PLOT_RANGE] to [0,1]
    return (v + PLOT_RANGE) / (2 * PLOT_RANGE);
  }
  function toData(u) {
    return u * (2 * PLOT_RANGE) - PLOT_RANGE;
  }

  // Diverging colormap: p=0 -> blue, p=0.5 -> white, p=1 -> orange.
  const BLUE = [59, 130, 246];
  const ORANGE = [249, 115, 22];
  const WHITE = [255, 255, 255];
  function colorFor(p) {
    let r, g, b;
    if (p < 0.5) {
      const a = 1 - p / 0.5; // 1 at p=0 (full blue), 0 at p=0.5 (white)
      r = WHITE[0] + (BLUE[0] - WHITE[0]) * a;
      g = WHITE[1] + (BLUE[1] - WHITE[1]) * a;
      b = WHITE[2] + (BLUE[2] - WHITE[2]) * a;
    } else {
      const bnd = (p - 0.5) / 0.5; // 0 at p=0.5 (white), 1 at p=1 (full orange)
      r = WHITE[0] + (ORANGE[0] - WHITE[0]) * bnd;
      g = WHITE[1] + (ORANGE[1] - WHITE[1]) * bnd;
      b = WHITE[2] + (ORANGE[2] - WHITE[2]) * bnd;
    }
    return [r | 0, g | 0, b | 0];
  }

  function renderHeatmap(offscreen, predictFn) {
    const ctx = offscreen.getContext('2d');
    const img = ctx.createImageData(GRID, GRID);
    for (let row = 0; row < GRID; row++) {
      // row 0 = top of canvas = highest y value
      const y = toData(1 - row / (GRID - 1));
      for (let col = 0; col < GRID; col++) {
        const x = toData(col / (GRID - 1));
        const p = predictFn(x, y);
        const [r, g, b] = colorFor(p);
        const idx = (row * GRID + col) * 4;
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function drawPanel(canvas, offscreen) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    ctx.imageSmoothingEnabled = true;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(offscreen, 0, 0, GRID, GRID, 0, 0, W, H);

    // scatter the training points on top
    const R = Math.max(3, W * 0.014);
    for (const p of points) {
      const px = toPixel(p.x) * W;
      const py = (1 - toPixel(p.y)) * H;
      ctx.beginPath();
      ctx.arc(px, py, R, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 0 ? '#3b82f6' : '#f97316';
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(0,0,0,0.55)';
      ctx.stroke();
    }
  }

  function setAccColor(el, acc) {
    el.classList.remove('acc-good', 'acc-bad');
    if (acc >= 0.9) el.classList.add('acc-good');
    else if (acc < 0.65) el.classList.add('acc-bad');
  }

  function newModels() {
    linear = RD.createLinearModel(7);
    mlp = RD.createMLPModel(8, 11);
    epoch = 0;
  }

  function newData(seed) {
    dataSeed = seed;
    points = RD.generateRings({ nPerClass: 150, seed: dataSeed, innerR: 1.2, outerR: 2.8, noise: 0.18 });
  }

  function stepsPerFrame() {
    // speed slider 1..12 -> epochs per animation frame
    return Math.round(Number(els.speedRange.value));
  }

  function trainAndRender() {
    const steps = stepsPerFrame();
    let linLoss = 0, mlpLoss = 0;
    if (epoch < MAX_EPOCHS) {
      for (let i = 0; i < steps; i++) {
        linLoss = RD.linearTrainStep(linear, points);
        mlpLoss = RD.mlpTrainStep(mlp, points);
        epoch++;
        if (epoch >= MAX_EPOCHS) break;
      }
    }

    const linPredict = (x, y) => RD.linearForward(linear.params, x, y);
    const mlpPredict = (x, y) => RD.mlpForward(mlp.params, mlp.H, x, y);

    renderHeatmap(heatLinear, linPredict);
    renderHeatmap(heatMLP, mlpPredict);
    drawPanel(els.canvasLinear, heatLinear);
    drawPanel(els.canvasMLP, heatMLP);

    const linAcc = RD.accuracy(linPredict, points);
    const mlpAcc = RD.accuracy(mlpPredict, points);

    els.linAcc.textContent = (linAcc * 100).toFixed(1) + '%';
    els.mlpAcc.textContent = (mlpAcc * 100).toFixed(1) + '%';
    els.linLoss.textContent = linLoss.toFixed(3);
    els.mlpLoss.textContent = mlpLoss.toFixed(3);
    setAccColor(els.linAcc, linAcc);
    setAccColor(els.mlpAcc, mlpAcc);

    els.statusLine.textContent =
      'epoch ' + epoch + (epoch >= MAX_EPOCHS ? ' — converged, training stopped' : '') +
      ' · ' + steps + ' epoch(s)/frame';
  }

  function loop() {
    // Stop doing work once training has converged and settled — no need to
    // keep re-rendering an identical frame forever.
    if (running && epoch < MAX_EPOCHS) {
      trainAndRender();
    }
    rafId = requestAnimationFrame(loop);
  }

  function resetWeights() {
    newModels();
    trainAndRender();
  }

  function regenerateData() {
    newData(Math.floor(Math.random() * 1e9));
    newModels();
    trainAndRender();
  }

  els.toggleBtn.addEventListener('click', () => {
    running = !running;
    els.toggleBtn.textContent = running ? 'Pause' : 'Resume';
    els.toggleBtn.classList.toggle('primary', running);
  });
  els.resetBtn.addEventListener('click', resetWeights);
  els.newDataBtn.addEventListener('click', regenerateData);

  // init
  newData(dataSeed);
  newModels();
  trainAndRender();
  loop();
})();
