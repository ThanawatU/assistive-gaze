// ไฟล์นี้ใชเสำหรับ
// อ่าน / แก้ HTML ของเว็บ
//     เปลี่ยนข้อความ
// ใส่ปุ่มเพิ่มเข้าเว็บ
// ทำ dark mode
console.log("Content script loaded");

document.body.style.backgroundColor = "#f0f0f0";

const ws = new WebSocket("ws://127.0.0.1:8000/gaze");

// ================= CONFIG =================
const CALIBRATION_TIME = 1500;
const GRID = 3;

// ================= STATE =================
let latestGaze = null;
let calibrationData = [];
let calibrationIndex = 0;
let isCalibrating = false;

// ================= DOT =================
const dot = document.createElement("div");
Object.assign(dot.style, {
  position: "fixed",
  width: "10px",
  height: "10px",
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
  width: "24px",
  height: "24px",
  borderRadius: "50%",
  background: "lime",
  zIndex: 999998,
  display: "none",
});
document.body.appendChild(target);

// ================= WEBSOCKET =================
ws.onmessage = (event) => {
  const g = JSON.parse(event.data);
  if (!g || g.confidence < 0.5) return;

  // ✅ normalize pupil → [0,1]
  const nx = g.px / g.eye_w;
  const ny = g.py / g.eye_h;

  latestGaze = { nx, ny };

  // วาด raw gaze ตลอด (debug)
  drawDot(nx, ny);
};

// ================= DRAW =================
function drawDot(x, y) {
  dot.style.left = `${x * window.innerWidth}px`;
  dot.style.top = `${y * window.innerHeight}px`;
}

// ================= CALIBRATION =================
function startCalibration() {
  calibrationData = [];
  calibrationIndex = 0;
  isCalibrating = true;
  target.style.display = "block";
  nextCalibrationPoint();
}

function nextCalibrationPoint() {
  if (calibrationIndex >= GRID * GRID) {
    finishCalibration();
    return;
  }

  const row = Math.floor(calibrationIndex / GRID);
  const col = calibrationIndex % GRID;

  const tx = (col + 0.5) / GRID;
  const ty = (row + 0.5) / GRID;

  target.style.left = `${tx * window.innerWidth}px`;
  target.style.top = `${ty * window.innerHeight}px`;

  const samples = [];

  const interval = setInterval(() => {
    if (latestGaze) samples.push({ ...latestGaze });
  }, 30);

  setTimeout(() => {
    clearInterval(interval);

    if (samples.length > 0) {
      const avg = average(samples);
      calibrationData.push({
        gaze: avg,
        screen: { x: tx, y: ty },
      });
    }

    calibrationIndex++;
    nextCalibrationPoint();
  }, CALIBRATION_TIME);
}

function finishCalibration() {
  isCalibrating = false;
  target.style.display = "none";
  console.log("✅ Calibration done:", calibrationData);
}

// ================= UTILS =================
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

// ================= KEY =================
window.addEventListener("keydown", (e) => {
  if (e.key === "c") {
    console.log("🟢 Start calibration");
    startCalibration();
  }
});
