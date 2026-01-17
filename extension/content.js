// ไฟล์นี้ใชเสำหรับ
// อ่าน / แก้ HTML ของเว็บ
//     เปลี่ยนข้อความ
// ใส่ปุ่มเพิ่มเข้าเว็บ
// ทำ dark mode
console.log("Content script loaded");

document.body.style.backgroundColor = "#f0f0f0";

const ws = new WebSocket("ws://127.0.0.1:8000/gaze");

// ================= CONFIG =================
const SMOOTHING = 0.15;
const CALIBRATION_TIME = 1500; // ms ต่อจุด
const GRID = 3;

// ================= STATE =================
let sx = 0.5, sy = 0.5;
let latestGaze = null;
let calibrationData = [];
let calibrationIndex = 0;
let isCalibrating = false;

// ================= DOT =================
const dot = document.createElement("div");
Object.assign(dot.style, {
  position: "fixed",
  width: "12px",
  height: "12px",
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
  const gaze = JSON.parse(event.data);
  latestGaze = gaze;

  if (!isCalibrating) {
    const { x, y } = applyMapping(gaze);
    drawDot(x, y);
  }
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
    if (latestGaze) {
      samples.push(latestGaze);
    }
  }, 50);

  setTimeout(() => {
    clearInterval(interval);

    const avg = average(samples);
    calibrationData.push({
      gaze: avg,
      screen: { x: tx, y: ty },
    });

    calibrationIndex++;
    nextCalibrationPoint();
  }, CALIBRATION_TIME);
}

function finishCalibration() {
  target.style.display = "none";
  isCalibrating = false;
  console.log("Calibration done:", calibrationData);
}

// ================= MAPPING =================
function applyMapping(gaze) {
  if (calibrationData.length < 4) {
    // fallback
    return {
      x: 0.5 + gaze.gx * 0.4,
      y: 0.5 + gaze.gy * 0.4,
    };
  }

  // simple nearest-neighbor (upgrade later)
  let best = calibrationData[0];
  let minDist = Infinity;

  for (const c of calibrationData) {
    const d =
      Math.pow(gaze.gx - c.gaze.gx, 2) +
      Math.pow(gaze.gy - c.gaze.gy, 2);
    if (d < minDist) {
      minDist = d;
      best = c;
    }
  }

  return best.screen;
}

// ================= UTILS =================
function average(samples) {
  const sum = samples.reduce(
    (a, b) => ({
      gx: a.gx + b.gx,
      gy: a.gy + b.gy,
      gz: a.gz + b.gz,
    }),
    { gx: 0, gy: 0, gz: 0 }
  );

  return {
    gx: sum.gx / samples.length,
    gy: sum.gy / samples.length,
    gz: sum.gz / samples.length,
  };
}

// ================= KEY BIND =================
window.addEventListener("keydown", (e) => {
  if (e.key === "c") {
    console.log("Start calibration");
    startCalibration();
  }
});
