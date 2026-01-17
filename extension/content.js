// ไฟล์นี้ใชเสำหรับ
// อ่าน / แก้ HTML ของเว็บ
//     เปลี่ยนข้อความ
// ใส่ปุ่มเพิ่มเข้าเว็บ
// ทำ dark mode
console.log("Content script loaded");

document.body.style.backgroundColor = "#f0f0f0";

console.log("math loaded:", typeof math);

// ================= WEBSOCKET =================
const ws = new WebSocket("ws://127.0.0.1:8000/gaze");

// ================= CONFIG =================
const CALIBRATION_TIME = 1500;
const GRID = 3;

// ================= STATE =================
let latestGaze = null;
let calibrationData = [];
let isCalibrating = false;
let affineMatrix = null;

// ================= DOT =================
const dot = document.createElement("div");
Object.assign(dot.style, {
  position: "fixed",
  width: "8px",
  height: "8px",
  borderRadius: "50%",
  background: "red",
  zIndex: 999999,
  pointerEvents: "none",
});
document.body.appendChild(dot);

// ================= TARGET =================
const target = document.createElement("div");
Object.assign(target.style, {
  position: "fixed",
  width: "20px",
  height: "20px",
  borderRadius: "50%",
  background: "lime",
  zIndex: 999998,
  display: "none",
});
document.body.appendChild(target);

// ================= DRAW =================
function drawDot(x, y) {
  dot.style.left = `${x * window.innerWidth}px`;
  dot.style.top = `${y * window.innerHeight}px`;
}

// ================= WEBSOCKET =================
ws.onmessage = (event) => {
  const g = JSON.parse(event.data);
  if (!g || g.confidence < 0.5) return;

  // normalize pupil → [0,1]
  const nx = g.px / g.eye_w;
  const ny = g.py / g.eye_h;

  latestGaze = { nx, ny };

  if (!isCalibrating && affineMatrix) {
    const p = applyAffine(nx, ny);
    drawDot(p.x, p.y);
  } else {
    // debug raw
    drawDot(nx, ny);
  }
};

// ================= CALIBRATION =================
function startCalibration() {
  calibrationData = [];
  isCalibrating = true;
  target.style.display = "block";
  runCalibrationPoint(0);
}

function runCalibrationPoint(index) {
  if (index >= GRID * GRID) {
    finishCalibration();
    return;
  }

  const row = Math.floor(index / GRID);
  const col = index % GRID;

  // 🔥 มุมจริง 100% (0.05 → 0.95)
  const margin = 0.05;
  const tx =
    GRID === 1
      ? 0.5
      : margin + (col / (GRID - 1)) * (1 - 2 * margin);
  const ty =
    GRID === 1
      ? 0.5
      : margin + (row / (GRID - 1)) * (1 - 2 * margin);

  target.style.left = `${tx * window.innerWidth - 10}px`;
  target.style.top = `${ty * window.innerHeight - 10}px`;

  const samples = [];

  const interval = setInterval(() => {
    if (latestGaze) samples.push({ ...latestGaze });
  }, 30);

  setTimeout(() => {
    clearInterval(interval);

    if (samples.length > 10) {
      const avg = average(samples);
      calibrationData.push({
        gaze: avg,
        screen: { x: tx, y: ty },
      });
    }

    runCalibrationPoint(index + 1);
  }, CALIBRATION_TIME);
}

function finishCalibration() {
  isCalibrating = false;
  target.style.display = "none";
  affineMatrix = computeAffine(calibrationData);
  console.log("✅ Calibration done", affineMatrix);
}

// ================= MATH =================
function average(samples) {
  const s = samples.reduce(
    (a, b) => ({ nx: a.nx + b.nx, ny: a.ny + b.ny }),
    { nx: 0, ny: 0 }
  );
  return {
    nx: s.nx / samples.length,
    ny: s.ny / samples.length,
  };
}

// screen = A * gaze
function computeAffine(data) {
  // X = [nx ny 1]
  const X = [];
  const Yx = [];
  const Yy = [];

  for (const d of data) {
    X.push([d.gaze.nx, d.gaze.ny, 1]);
    Yx.push(d.screen.x);
    Yy.push(d.screen.y);
  }

  const Xt = math.transpose(X);
  const XtX = math.multiply(Xt, X);
  const XtX_inv = math.inv(XtX);
  const pinv = math.multiply(XtX_inv, Xt);

  const ax = math.multiply(pinv, Yx);
  const ay = math.multiply(pinv, Yy);

  return { ax, ay };
}

function applyAffine(nx, ny) {
  const x =
    affineMatrix.ax[0] * nx +
    affineMatrix.ax[1] * ny +
    affineMatrix.ax[2];

  const y =
    affineMatrix.ay[0] * nx +
    affineMatrix.ay[1] * ny +
    affineMatrix.ay[2];

  return {
    x: Math.min(Math.max(x, 0), 1),
    y: Math.min(Math.max(y, 0), 1),
  };
}

// ================= KEY =================
window.addEventListener("keydown", (e) => {
  if (e.key === "c") {
    console.log("🟢 Start calibration");
    startCalibration();
  }
});
