/* =============================================================
   ANN Digit Recognition — frontend
   Sections:
     1. State
     2. Drawing canvas (280×280 → 28×28 pixels)
     3. Network visualisation canvas
     4. Animation engine (phase state machine)
     5. Probability bars
     6. API & status polling
     7. Init
   ============================================================= */

// =============================================================
// 1. STATE
// =============================================================
const state = {
  trained: false,
  isBusy: false,
  pixels: new Float32Array(784).fill(0),
  lastResult: null,
};

// =============================================================
// 2. DRAWING CANVAS
// =============================================================

let drawCanvas, drawCtx;
let isDrawing = false;
let lastX = 0, lastY = 0;

function setupDrawCanvas() {
  drawCanvas = document.getElementById('draw-canvas');
  drawCtx = drawCanvas.getContext('2d');
  clearDrawCanvas();

  // Mouse events
  drawCanvas.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const { x, y } = getDrawPos(e);
    lastX = x; lastY = y;
    drawDot(x, y);
  });
  drawCanvas.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const { x, y } = getDrawPos(e);
    drawLine(lastX, lastY, x, y);
    lastX = x; lastY = y;
  });
  drawCanvas.addEventListener('mouseup', () => { isDrawing = false; });
  drawCanvas.addEventListener('mouseleave', () => { isDrawing = false; });

  // Touch events (mobile)
  drawCanvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    isDrawing = true;
    const { x, y } = getDrawPos(e.touches[0]);
    lastX = x; lastY = y;
    drawDot(x, y);
  }, { passive: false });
  drawCanvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!isDrawing) return;
    const { x, y } = getDrawPos(e.touches[0]);
    drawLine(lastX, lastY, x, y);
    lastX = x; lastY = y;
  }, { passive: false });
  drawCanvas.addEventListener('touchend', () => { isDrawing = false; });
}

function getDrawPos(e) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

function drawDot(x, y) {
  drawCtx.beginPath();
  drawCtx.arc(x, y, 12, 0, Math.PI * 2);
  drawCtx.fillStyle = '#fff';
  drawCtx.fill();
}

function drawLine(x1, y1, x2, y2) {
  drawCtx.strokeStyle = '#fff';
  drawCtx.lineWidth = 24;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  drawCtx.beginPath();
  drawCtx.moveTo(x1, y1);
  drawCtx.lineTo(x2, y2);
  drawCtx.stroke();
}

function clearDrawCanvas() {
  drawCtx.fillStyle = '#000';
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function getPixels() {
  // Downsample 280×280 to 28×28
  const hidden = document.createElement('canvas');
  hidden.width = 28;
  hidden.height = 28;
  const hCtx = hidden.getContext('2d');
  hCtx.drawImage(drawCanvas, 0, 0, 28, 28);
  const data = hCtx.getImageData(0, 0, 28, 28).data;
  const pixels = new Float32Array(784);
  for (let i = 0; i < 784; i++) {
    // Use red channel (grayscale: R=G=B). Black bg = 0, white digit = 1.
    pixels[i] = data[i * 4] / 255.0;
  }

  // Centre by centre-of-mass — MNIST digits are preprocessed this way.
  // Without this, a 7 drawn slightly off-centre looks nothing like a
  // training example and gets misclassified as 2 or 3.
  const centred = centreByMass(pixels);

  state.pixels = centred;
  return Array.from(centred);
}

/**
 * Shift the 28×28 grid so the centre of mass of pixel values sits at (14,14),
 * matching the MNIST preprocessing convention.
 */
function centreByMass(pixels) {
  let mass = 0, cx = 0, cy = 0;
  for (let i = 0; i < 784; i++) {
    const v = pixels[i];
    const row = Math.floor(i / 28);
    const col = i % 28;
    mass += v;
    cx += col * v;
    cy += row * v;
  }
  if (mass < 0.5) return pixels; // blank canvas — nothing to shift

  const dx = Math.round(13.5 - cx / mass);
  const dy = Math.round(13.5 - cy / mass);
  if (dx === 0 && dy === 0) return pixels;

  const shifted = new Float32Array(784).fill(0);
  for (let i = 0; i < 784; i++) {
    const row = Math.floor(i / 28);
    const col = i % 28;
    const nr = row + dy;
    const nc = col + dx;
    if (nr >= 0 && nr < 28 && nc >= 0 && nc < 28) {
      shifted[nr * 28 + nc] = pixels[i];
    }
  }
  return shifted;
}

// =============================================================
// 3. NETWORK VISUALISATION CANVAS
// =============================================================

const VIZ = {
  canvas: null,
  ctx: null,
  width: 0,
  height: 0,

  // Node layout (computed in layoutNodes)
  inputGrid: { x: 0, y: 0, cellSize: 4 }, // 28*4 = 112px grid
  h1Nodes: [],    // 32 nodes: [{x, y, activation, lit}]
  h2Nodes: [],    // 16 nodes
  outputNodes: [],// 10 nodes: [{x, y, activation, label, lit}]

  // Animation phase state machine
  phase: 'idle',
  phaseStart: 0,
  waveX: 0,

  // Latest API result
  activations: null,
  probabilities: null,
  prediction: null,

  animId: null,
};

function setupVizCanvas() {
  VIZ.canvas = document.getElementById('viz-canvas');
  resizeVizCanvas();
  window.addEventListener('resize', resizeVizCanvas);
}

function resizeVizCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const panel = document.querySelector('.viz-panel');
  const cssWidth = panel.clientWidth - 40; // account for padding
  const cssHeight = Math.max(420, Math.min(cssWidth * 0.65, 520));

  VIZ.canvas.style.width = cssWidth + 'px';
  VIZ.canvas.style.height = cssHeight + 'px';
  VIZ.canvas.width = cssWidth * dpr;
  VIZ.canvas.height = cssHeight * dpr;

  VIZ.ctx = VIZ.canvas.getContext('2d');
  VIZ.ctx.scale(dpr, dpr);
  VIZ.width = cssWidth;
  VIZ.height = cssHeight;

  layoutNodes();
  drawIdleState();
}

function layoutNodes() {
  const W = VIZ.width;
  const H = VIZ.height;
  const pad = 24;

  // Input grid: small 28×4=112px square, vertically centred
  const cellSize = Math.max(3, Math.floor((H - 2 * pad) / 28));
  const gridW = 28 * cellSize;
  const gridH = 28 * cellSize;
  VIZ.inputGrid = {
    x: pad,
    y: (H - gridH) / 2,
    cellSize,
    w: gridW,
    h: gridH,
  };

  const inputRight = pad + gridW;

  // Column X positions
  const h1X = inputRight + (W - inputRight) * 0.28;
  const h2X = inputRight + (W - inputRight) * 0.58;
  const outX = inputRight + (W - inputRight) * 0.88;

  const vPad = 16;

  // 32 H1 nodes
  VIZ.h1Nodes = Array.from({ length: 32 }, (_, i) => ({
    x: h1X,
    y: vPad + (i / 31) * (H - 2 * vPad),
    activation: 0,
    lit: false,
  }));

  // 16 H2 nodes
  VIZ.h2Nodes = Array.from({ length: 16 }, (_, i) => ({
    x: h2X,
    y: vPad + 10 + (i / 15) * (H - 2 * vPad - 20),
    activation: 0,
    lit: false,
  }));

  // 10 output nodes
  VIZ.outputNodes = Array.from({ length: 10 }, (_, i) => ({
    x: outX,
    y: vPad + 14 + (i / 9) * (H - 2 * vPad - 28),
    activation: 0,
    label: String(i),
    lit: false,
  }));
}

// Map activation value [0,1] → CSS color string
function activationColor(t, lit) {
  if (!lit) return 'hsl(215, 20%, 18%)'; // dark unlit
  const clamped = Math.max(0, Math.min(1, t));
  const hue = 220 - clamped * 190;       // 220 (blue) → 30 (orange)
  const sat = 65 + clamped * 30;
  const lig = 28 + clamped * 38;
  return `hsl(${hue}, ${sat}%, ${lig}%)`;
}

function nodeRadius(isOutput) {
  return isOutput ? 14 : 6;
}

function drawNode(ctx, node, isOutput, pulseFactor = 1) {
  const r = nodeRadius(isOutput) * pulseFactor;
  const color = activationColor(node.activation, node.lit);

  // Glow for lit nodes
  if (node.lit && node.activation > 0.15) {
    ctx.shadowBlur = 12 * node.activation;
    ctx.shadowColor = color;
  } else {
    ctx.shadowBlur = 0;
  }

  ctx.beginPath();
  ctx.arc(node.x, node.y, r, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.shadowBlur = 0;

  // Output node labels
  if (isOutput) {
    ctx.fillStyle = node.lit ? '#e6edf3' : '#4a5568';
    ctx.font = `bold ${Math.round(r * 0.95)}px Inter, system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(node.label, node.x, node.y);
  }
}

function drawConnections(ctx, fromNodes, toNodes, progress = 1) {
  // Draw all connections; brighter when both ends are lit
  for (const from of fromNodes) {
    for (const to of toNodes) {
      const bothLit = from.lit && to.lit;
      const base = bothLit
        ? 0.05 + 0.35 * Math.min(from.activation, to.activation)
        : 0.025;
      const alpha = base * Math.min(progress, 1);
      ctx.strokeStyle = `rgba(88, 166, 255, ${alpha})`;
      ctx.lineWidth = bothLit ? 0.6 : 0.3;
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
    }
  }
}

function drawInputGrid(ctx) {
  const { x, y, cellSize, w, h } = VIZ.inputGrid;
  const pixels = state.pixels;
  for (let i = 0; i < 784; i++) {
    const row = Math.floor(i / 28);
    const col = i % 28;
    const b = Math.round(pixels[i] * 255);
    ctx.fillStyle = `rgb(${b},${b},${b})`;
    ctx.fillRect(x + col * cellSize, y + row * cellSize, cellSize, cellSize);
  }
  // Border
  ctx.strokeStyle = '#30363d';
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, w, h);

  // Label
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Input (28×28)', x + w / 2, y + h + 4);
}

function drawLayerLabel(ctx, x, label, y) {
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px Inter, system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText(label, x, y);
}

function drawIdleState() {
  const ctx = VIZ.ctx;
  if (!ctx) return;
  ctx.clearRect(0, 0, VIZ.width, VIZ.height);
  drawInputGrid(ctx);
  drawConnections(ctx, VIZ.h1Nodes, VIZ.h2Nodes);
  drawConnections(ctx, VIZ.h2Nodes, VIZ.outputNodes);
  VIZ.h1Nodes.forEach(n => drawNode(ctx, n, false));
  VIZ.h2Nodes.forEach(n => drawNode(ctx, n, false));
  VIZ.outputNodes.forEach(n => drawNode(ctx, n, true));
  drawLayerLabel(ctx, VIZ.h1Nodes[0].x, 'Hidden 1 (32)', VIZ.height - 16);
  drawLayerLabel(ctx, VIZ.h2Nodes[0].x, 'Hidden 2 (16)', VIZ.height - 16);
  drawLayerLabel(ctx, VIZ.outputNodes[0].x, 'Output (10)', VIZ.height - 16);
}

// =============================================================
// 4. ANIMATION ENGINE
// =============================================================

function resetNodes() {
  VIZ.h1Nodes.forEach(n => { n.lit = false; n.activation = 0; });
  VIZ.h2Nodes.forEach(n => { n.lit = false; n.activation = 0; });
  VIZ.outputNodes.forEach(n => { n.lit = false; n.activation = 0; });
}

function setPhase(phase, timestamp) {
  VIZ.phase = phase;
  VIZ.phaseStart = timestamp;
}

function startAnimation(apiResult) {
  state.lastResult = apiResult;
  VIZ.activations = apiResult.activations;
  VIZ.probabilities = apiResult.probabilities;
  VIZ.prediction = apiResult.prediction;

  resetNodes();
  if (VIZ.animId) cancelAnimationFrame(VIZ.animId);
  VIZ.phase = 'wave';
  VIZ.phaseStart = 0; // set properly in first frame
  VIZ.animId = requestAnimationFrame(animLoop);
}

function animLoop(ts) {
  if (VIZ.phaseStart === 0) VIZ.phaseStart = ts;
  const elapsed = ts - VIZ.phaseStart;
  const ctx = VIZ.ctx;

  ctx.clearRect(0, 0, VIZ.width, VIZ.height);

  // Always draw base scene
  drawInputGrid(ctx);
  drawConnections(ctx, VIZ.h1Nodes, VIZ.h2Nodes);
  drawConnections(ctx, VIZ.h2Nodes, VIZ.outputNodes);
  VIZ.h1Nodes.forEach(n => drawNode(ctx, n, false));
  VIZ.h2Nodes.forEach(n => drawNode(ctx, n, false));
  VIZ.outputNodes.forEach(n => drawNode(ctx, n, true));
  drawLayerLabel(ctx, VIZ.h1Nodes[0].x, 'Hidden 1 (32)', VIZ.height - 16);
  drawLayerLabel(ctx, VIZ.h2Nodes[0].x, 'Hidden 2 (16)', VIZ.height - 16);
  drawLayerLabel(ctx, VIZ.outputNodes[0].x, 'Output (10)', VIZ.height - 16);

  switch (VIZ.phase) {

    case 'wave': {
      // Glow wave sweeps from input grid right edge → H1
      const inputRight = VIZ.inputGrid.x + VIZ.inputGrid.w;
      const h1X = VIZ.h1Nodes[0].x;
      const t = Math.min(elapsed / 600, 1);
      const waveX = inputRight + (h1X - inputRight) * t;

      const grad = ctx.createLinearGradient(waveX - 30, 0, waveX + 30, 0);
      grad.addColorStop(0, 'rgba(88, 166, 255, 0)');
      grad.addColorStop(0.5, 'rgba(88, 166, 255, 0.55)');
      grad.addColorStop(1, 'rgba(88, 166, 255, 0)');
      ctx.fillStyle = grad;
      ctx.fillRect(waveX - 30, 0, 60, VIZ.height);

      if (elapsed >= 600) setPhase('h1', ts);
      break;
    }

    case 'h1': {
      const DURATION = 500;
      VIZ.h1Nodes.forEach((node, i) => {
        const delay = (i / 31) * (DURATION * 0.6);
        if (elapsed > delay) {
          node.lit = true;
          node.activation = VIZ.activations.hidden1[i];
        }
      });
      if (elapsed >= DURATION) setPhase('conn1', ts);
      break;
    }

    case 'conn1': {
      const DURATION = 450;
      const progress = elapsed / DURATION;
      // Redraw H1→H2 connections with animated progress alpha
      for (const from of VIZ.h1Nodes) {
        for (const to of VIZ.h2Nodes) {
          const alpha = (from.lit ? 0.08 + 0.2 * from.activation : 0.02) * Math.min(progress * 2, 1);
          ctx.strokeStyle = `rgba(88, 166, 255, ${alpha})`;
          ctx.lineWidth = 0.4;
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        }
      }
      if (elapsed >= DURATION) setPhase('h2', ts);
      break;
    }

    case 'h2': {
      const DURATION = 450;
      VIZ.h2Nodes.forEach((node, i) => {
        const delay = (i / 15) * (DURATION * 0.6);
        if (elapsed > delay) {
          node.lit = true;
          node.activation = VIZ.activations.hidden2[i];
        }
      });
      if (elapsed >= DURATION) setPhase('conn2', ts);
      break;
    }

    case 'conn2': {
      const DURATION = 400;
      const progress = elapsed / DURATION;
      for (const from of VIZ.h2Nodes) {
        for (const to of VIZ.outputNodes) {
          const alpha = (from.lit ? 0.1 + 0.3 * from.activation : 0.02) * Math.min(progress * 2, 1);
          ctx.strokeStyle = `rgba(88, 166, 255, ${alpha})`;
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        }
      }
      if (elapsed >= DURATION) setPhase('output', ts);
      break;
    }

    case 'output': {
      const DURATION = 550;
      VIZ.outputNodes.forEach((node, i) => {
        const delay = (i / 9) * (DURATION * 0.5);
        if (elapsed > delay) {
          node.lit = true;
          node.activation = VIZ.probabilities[i];
        }
      });

      // Pulse winner
      const winner = VIZ.outputNodes[VIZ.prediction];
      if (winner.lit) {
        const pulse = 1 + 0.25 * Math.sin(elapsed / 120 * Math.PI);
        drawNode(ctx, winner, true, pulse);
      }

      if (elapsed >= DURATION) {
        setPhase('done', ts);
        buildProbBars(VIZ.probabilities, VIZ.prediction);
        showProbBars();
        showFeedbackPrompt(VIZ.prediction);
      }
      break;
    }

    case 'done': {
      // Keep winner pulsing indefinitely
      const winner = VIZ.outputNodes[VIZ.prediction];
      const pulse = 1 + 0.2 * Math.sin(ts / 400);
      drawNode(ctx, winner, true, pulse);
      break;
    }
  }

  if (VIZ.phase !== 'idle') {
    VIZ.animId = requestAnimationFrame(animLoop);
  }
}

// =============================================================
// 5. PROBABILITY BARS
// =============================================================

function buildProbBars(probs, prediction) {
  const list = document.getElementById('prob-bars-list');
  list.innerHTML = '';
  probs.forEach((p, i) => {
    const isWinner = i === prediction;
    const item = document.createElement('div');
    item.className = 'prob-bar-item';
    item.innerHTML = `
      <span class="prob-bar-label ${isWinner ? 'winner' : ''}">${i}</span>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${isWinner ? 'winner' : ''}" id="bar-fill-${i}" style="width:0%"></div>
      </div>
      <span class="prob-bar-pct ${isWinner ? 'winner' : ''}" id="bar-pct-${i}">0%</span>
    `;
    list.appendChild(item);
  });

  // Animate bars after a short delay (allows repaint)
  setTimeout(() => {
    probs.forEach((p, i) => {
      const pct = (p * 100).toFixed(1);
      document.getElementById(`bar-fill-${i}`).style.width = pct + '%';
      document.getElementById(`bar-pct-${i}`).textContent = pct + '%';
    });
  }, 60);
}

function showProbBars() {
  document.getElementById('prob-bars').classList.remove('hidden');
}

// =============================================================
// 5b. FEEDBACK UI
// =============================================================

function showFeedbackPrompt(prediction) {
  const el = document.getElementById('feedback-area');
  el.innerHTML = `
    <div class="feedback-prompt">
      Was that right?
      <button class="fb-btn fb-correct" onclick="handleFeedbackCorrect()">✓ Yes</button>
      <button class="fb-btn fb-wrong" onclick="showDigitPicker(${prediction})">✗ No</button>
    </div>
  `;
  el.classList.remove('hidden');
}

function showDigitPicker(wrongPrediction) {
  const el = document.getElementById('feedback-area');
  const btns = Array.from({ length: 10 }, (_, i) => {
    const cls = i === wrongPrediction ? 'fb-digit-btn fb-wrong-digit' : 'fb-digit-btn';
    return `<button class="${cls}" onclick="handleFeedbackCorrection(${i})">${i}</button>`;
  }).join('');
  el.innerHTML = `
    <div class="feedback-picker">
      <span class="feedback-picker-label">It was actually:</span>
      <div class="digit-grid">${btns}</div>
    </div>
  `;
}

function handleFeedbackCorrect() {
  const el = document.getElementById('feedback-area');
  el.innerHTML = '<div class="feedback-thanks">Great!</div>';
  setTimeout(() => el.classList.add('hidden'), 1500);
}

async function handleFeedbackCorrection(correctLabel) {
  const el = document.getElementById('feedback-area');
  el.innerHTML = '<div class="feedback-thanks">Learning…</div>';

  try {
    const res = await fetch('/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pixels: Array.from(state.pixels),
        correct_label: correctLabel,
      }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    el.innerHTML = `<div class="feedback-thanks">Got it — I'll remember that ${correctLabel}!</div>`;
  } catch (err) {
    el.innerHTML = `<div class="feedback-thanks feedback-error">Update failed: ${err.message}</div>`;
    console.error(err);
  }

  setTimeout(() => el.classList.add('hidden'), 2500);
}

// =============================================================
// 6. API & STATUS POLLING
// =============================================================

async function pollStatus() {
  const overlay = document.getElementById('loading-overlay');
  const msg = document.getElementById('status-message');

  try {
    const res = await fetch('/status');
    const data = await res.json();

    if (data.error) {
      msg.textContent = `Error: ${data.error}`;
      return; // stop polling
    }

    if (data.trained) {
      overlay.classList.add('hidden');
      state.trained = true;
      document.getElementById('recognize-btn').disabled = false;
      const badge = document.getElementById('accuracy-badge');
      badge.textContent = `Test accuracy: ${(data.accuracy * 100).toFixed(1)}%`;
      badge.classList.remove('hidden');
      return; // stop polling
    }

    // Show training progress
    if (data.total_epochs > 0 && data.epoch > 0) {
      const pct = Math.round((data.epoch / data.total_epochs) * 100);
      msg.textContent = `Training… epoch ${data.epoch}/${data.total_epochs} (${pct}%)  acc=${(data.accuracy * 100).toFixed(1)}%`;
    } else {
      msg.textContent = data.training
        ? 'Loading MNIST dataset…'
        : 'Connecting to server…';
    }

    setTimeout(pollStatus, 2000);

  } catch {
    msg.textContent = 'Connecting to server…';
    setTimeout(pollStatus, 3000);
  }
}

async function handleRecognise() {
  if (state.isBusy || !state.trained) return;

  // Check the canvas isn't empty
  const pixels = getPixels();
  const hasContent = pixels.some(v => v > 0.1);
  if (!hasContent) {
    document.getElementById('result-label').textContent = 'Draw a digit first!';
    return;
  }

  state.isBusy = true;
  document.getElementById('recognize-btn').disabled = true;
  document.getElementById('result-label').textContent = '';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ pixels }),
    });
    const data = await res.json();

    if (data.error) {
      document.getElementById('result-label').textContent = data.error;
      return;
    }

    document.getElementById('result-label').textContent = `Prediction: ${data.prediction}`;
    startAnimation(data);

  } catch (err) {
    document.getElementById('result-label').textContent = 'Request failed — is the server running?';
    console.error(err);
  } finally {
    // Re-enable button after animation completes (~3.5s)
    setTimeout(() => {
      state.isBusy = false;
      document.getElementById('recognize-btn').disabled = false;
    }, 3500);
  }
}

// =============================================================
// 7. INIT
// =============================================================

document.addEventListener('DOMContentLoaded', () => {
  setupDrawCanvas();
  setupVizCanvas();

  document.getElementById('recognize-btn').addEventListener('click', handleRecognise);
  document.getElementById('clear-btn').addEventListener('click', () => {
    clearDrawCanvas();
    state.pixels = new Float32Array(784).fill(0);
    document.getElementById('result-label').textContent = '';
    document.getElementById('prob-bars').classList.add('hidden');
    document.getElementById('feedback-area').classList.add('hidden');
    resetNodes();

    if (VIZ.animId) {
      cancelAnimationFrame(VIZ.animId);
      VIZ.animId = null;
    }
    VIZ.phase = 'idle';
    drawIdleState();
  });

  pollStatus();
});
