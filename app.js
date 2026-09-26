import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

// Same order as training (and your confusion matrix)
const LABELS = ["surprise", "fear", "disgust", "happy", "sad", "anger", "neutral"];
const COLORS = { surprise: "#f0925a", fear: "#9a7fc0", disgust: "#7aa35f", happy: "#f2c14e", sad: "#6f94c4", anger: "#d9534f", neutral: "#b8bec6" };
const SIZE = 224, MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const PAD = 0.0;         // margin around the detected face; try 0 to 0.2
const THRESHOLD = 0.3;   // below this the app says "Unsure" (chance is about 0.14)
const SMOOTH = 0.6;      // higher = steadier label
const STEP_MS = 100;     // how often to classify
const TRACE_MS = 30000;  // length of the mood trace

const $ = (id) => document.getElementById(id);
const video = $("video"), overlay = $("overlay"), frame = $("frame"), startBtn = $("start");
const statusEl = $("status"), moodEl = $("mood"), sureEl = $("sure"), barsEl = $("bars"), trace = $("trace");
const octx = overlay.getContext("2d"), tctx = trace.getContext("2d");
const crop = document.createElement("canvas");
crop.width = crop.height = SIZE;
const cctx = crop.getContext("2d", { willReadFrequently: true });

const rows = LABELS.map((l) => {
  const li = document.createElement("li");
  li.innerHTML = `<span>${l}</span><div class="track"><div class="fill"></div></div><span class="pct">0%</span>`;
  barsEl.appendChild(li);
  return li;
});

let session, detector, smooth = null, lastRun = 0, history = [];
let mode = "live", running = false, loadPromise = null;

function ensureModelsLoaded() {
  if (!loadPromise) loadPromise = loadModels();
  return loadPromise;
}

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("on", b === btn));
    mode = btn.dataset.mode;
    $("live-panel").hidden = mode !== "live";
    $("game-panel").hidden = mode !== "game";
    if (mode !== "game") game.stop();
  });
});

async function loadModels() {
  ort.env.wasm.numThreads = 1;
  const base = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
  const [s, files] = await Promise.all([
    ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] }),
    FilesetResolver.forVisionTasks(base),
  ]);
  session = s;
  detector = await FaceDetector.createFromOptions(files, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite" },
    runningMode: "VIDEO",
    minDetectionConfidence: 0.6,
  });
}

const stopBtn = $("stop");

startBtn.addEventListener("click", async () => {
  startBtn.disabled = true;
  startBtn.textContent = "Loading...";
  statusEl.textContent = "Starting the camera and loading the model. The first load can take a few seconds.";
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } }, audio: false,
    });
    await video.play();
    frame.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    await ensureModelsLoaded();
    frame.classList.add("on");
    startBtn.disabled = false;
    startBtn.textContent = "Turn on camera";
    running = true;
    statusEl.textContent = "Running on this device. Nothing you show the camera is uploaded.";
    requestAnimationFrame(loop);
  } catch (e) {
    startBtn.disabled = false;
    startBtn.textContent = "Turn on camera";
    statusEl.textContent = e.name === "NotAllowedError"
      ? "Camera access was blocked. Allow the camera for this site in your browser settings, then try again."
      : "Could not start: " + e.message;
  }
});

stopBtn.addEventListener("click", () => {
  running = false;
  game.stop();
  video.srcObject?.getTracks().forEach((t) => t.stop());
  video.srcObject = null;
  frame.classList.remove("on");
  octx.clearRect(0, 0, overlay.width, overlay.height);
  smooth = null;
  history = [];
  statusEl.textContent = "Camera is off. Turn it back on any time — the model stays loaded.";
  moodEl.textContent = "Mood detector";
  sureEl.textContent = "Turn on the camera, then make an expression.";
  rows.forEach((li) => { li.classList.remove("top"); li.querySelector(".fill").style.width = "0%"; li.querySelector(".pct").textContent = "0%"; });
  tctx.clearRect(0, 0, trace.width, trace.height);
});

async function loop(now) {
  if (!running) return;
  if (now - lastRun >= STEP_MS) {
    lastRun = now;
    await step(now);
  }
  requestAnimationFrame(loop);
}

const area = (d) => d.boundingBox.width * d.boundingBox.height;

async function step(now) {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  const faces = detector.detectForVideo(video, now).detections;
  if (!faces.length) {
    smooth = null;
    return render(null, 0, false, now);
  }
  const b = faces.reduce((a, c) => (area(c) > area(a) ? c : a)).boundingBox;
  const x = Math.max(0, b.originX - b.width * PAD), y = Math.max(0, b.originY - b.height * PAD);
  const w = Math.min(video.videoWidth - x, b.width * (1 + 2 * PAD)), h = Math.min(video.videoHeight - y, b.height * (1 + 2 * PAD));
  cctx.drawImage(video, x, y, w, h, 0, 0, SIZE, SIZE);

  const probs = await classify();
  smooth = smooth ? smooth.map((v, i) => SMOOTH * v + (1 - SMOOTH) * probs[i]) : probs;
  const top = smooth.indexOf(Math.max(...smooth));
  const sure = smooth[top] >= THRESHOLD;
  drawBox(x, y, w, h, sure ? COLORS[LABELS[top]] : "#ffffff");
  render(smooth, top, sure, now);
}

async function classify() {
  const { data } = cctx.getImageData(0, 0, SIZE, SIZE);
  const n = SIZE * SIZE, input = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) input[c * n + i] = (data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", input, [1, 3, SIZE, SIZE]) });
  const logits = out[session.outputNames[0]].data, m = Math.max(...logits);
  const e = Array.from(logits, (v) => Math.exp(v - m)), sum = e.reduce((a, c) => a + c);
  return e.map((v) => v / sum);
}

function drawBox(x, y, w, h, color) {
  const L = Math.min(w, h) * 0.22;
  octx.strokeStyle = color;
  octx.lineWidth = Math.max(3, w / 60);
  octx.lineCap = "round";
  octx.lineJoin = "round";
  octx.beginPath();
  [[x, y, 1, 1], [x + w, y, -1, 1], [x, y + h, 1, -1], [x + w, y + h, -1, -1]].forEach(([cx, cy, dx, dy]) => {
    octx.moveTo(cx + dx * L, cy);
    octx.lineTo(cx, cy);
    octx.lineTo(cx, cy + dy * L);
  });
  octx.stroke();
}

function render(p, top, sure, now) {
  game.onFrame(p, top, sure);
  if (mode !== "live") return;
  const root = document.documentElement.style;
  if (!p) {
    moodEl.textContent = "No face";
    sureEl.textContent = "Move closer and face the camera.";
    root.setProperty("--mood", COLORS.neutral);
    rows.forEach((li) => { li.classList.remove("top"); li.querySelector(".fill").style.width = "0%"; li.querySelector(".pct").textContent = "0%"; });
  } else {
    moodEl.textContent = sure ? LABELS[top] : "Unsure";
    sureEl.textContent = sure ? `${Math.round(p[top] * 100)}% sure` : "Try a stronger expression.";
    root.setProperty("--mood", sure ? COLORS[LABELS[top]] : COLORS.neutral);
    rows.forEach((li, i) => {
      li.classList.toggle("top", sure && i === top);
      li.querySelector(".fill").style.width = p[i] * 100 + "%";
      li.querySelector(".pct").textContent = Math.round(p[i] * 100) + "%";
    });
    history.push({ t: now, i: sure ? top : -1 });
  }
  drawTrace(now);
}

// ---------------------------------------------------------------- Game: Emotion Streak
// Only these four are used — the model is much less reliable on fear/disgust.
const GAME_TARGETS = ["happy", "surprise", "neutral", "anger"];
const START_MS = 6000, MIN_MS = 2200, STEP_MS_DROP = 300, HOLD_MS = 700;
const BEST_KEY = "mood-streak-best";

const targetEl = $("target"), roundEl = $("round"), bestEl = $("best");
const timerFill = $("timer-fill"), gameMsg = $("game-msg"), gameStartBtn = $("game-start");
bestEl.textContent = localStorage.getItem(BEST_KEY) || 0;

const game = (() => {
  let active = false, streak = 0, target = null, roundMs = START_MS;
  let roundStart = 0, holdStart = null, raf = null, last = null;

  function pickTarget() {
    const pool = GAME_TARGETS.filter((l) => l !== last);
    return pool[Math.floor(Math.random() * pool.length)];
  }

  function nextRound() {
    target = last = pickTarget();
    holdStart = null;
    roundStart = performance.now();
    roundEl.textContent = streak;
    targetEl.textContent = `Make a ${target} face`;
    gameMsg.textContent = "Match the face above before time runs out.";
    gameMsg.className = "fine";
    tick();
  }

  function tick() {
    if (!active) return;
    const elapsed = performance.now() - roundStart;
    timerFill.style.transform = `scaleX(${Math.max(0, 1 - elapsed / roundMs)})`;
    if (elapsed >= roundMs) return fail();
    raf = requestAnimationFrame(tick);
  }

  function fail() {
    active = false;
    cancelAnimationFrame(raf);
    const best = Math.max(streak, +localStorage.getItem(BEST_KEY) || 0);
    localStorage.setItem(BEST_KEY, best);
    bestEl.textContent = best;
    targetEl.textContent = `Streak: ${streak}`;
    gameMsg.textContent = streak > 0 ? "Time's up — nice streak!" : "Time's up — give it another go.";
    gameMsg.className = "fine miss";
    timerFill.style.transform = "scaleX(0)";
    gameStartBtn.textContent = "Try again";
    gameStartBtn.hidden = false;
  }

  return {
    start() {
      if (!running) { gameMsg.textContent = "Turn the camera on first."; return; }
      active = true; streak = 0; roundMs = START_MS; last = null;
      gameStartBtn.hidden = true;
      nextRound();
    },
    stop() { active = false; cancelAnimationFrame(raf); },
    onFrame(p, top, sure) {
      if (!active || mode !== "game" || !target) return;
      const matched = sure && LABELS[top] === target;
      const now = performance.now();
      if (matched) {
        if (holdStart === null) holdStart = now;
        if (now - holdStart >= HOLD_MS) {
          streak++;
          roundMs = Math.max(MIN_MS, roundMs - STEP_MS_DROP);
          cancelAnimationFrame(raf);
          gameMsg.textContent = "Nailed it!";
          gameMsg.className = "fine hit";
          target = null; // prevent double-scoring before nextRound runs
          setTimeout(nextRound, 450);
        }
      } else {
        holdStart = null;
      }
    },
  };
})();

gameStartBtn.addEventListener("click", () => game.start());

function drawTrace(now) {
  const W = trace.width, H = trace.height, rh = H / LABELS.length;
  tctx.clearRect(0, 0, W, H);
  tctx.fillStyle = "rgba(28,32,41,.10)";
  for (let r = 0; r < LABELS.length; r++) tctx.fillRect(0, r * rh + rh / 2 - 0.5, W, 1);
  history = history.filter((e) => now - e.t < TRACE_MS);
  tctx.fillStyle = "#1c2029";
  for (const e of history) {
    if (e.i < 0) continue;
    tctx.beginPath();
    tctx.arc(W - ((now - e.t) / TRACE_MS) * W, e.i * rh + rh / 2, 5, 0, Math.PI * 2);
    tctx.fill();
  }
}
