import cv2
import numpy as np
from ultralytics import YOLO
import asyncio
import threading
from fastapi import FastAPI, WebSocket

# ---------------- CONFIG ----------------
MODEL_PATH = "models/yolov12n-face.pt"
CAMERA_ID = 1
EYE_W, EYE_H = 60, 36

# ---------------- GLOBAL STATE ----------------
latest_gaze = {
    "px": 0.0,
    "py": 0.0,
    "eye_w": EYE_W,
    "eye_h": EYE_H,
    "confidence": 0.0
}

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_ID)

# ---------------- FASTAPI ----------------
app = FastAPI()

@app.websocket("/gaze")
async def gaze_ws(ws: WebSocket):
    await ws.accept()
    while True:
        await ws.send_json(latest_gaze)
        await asyncio.sleep(0.03)

# ---------------- UTILS ----------------
def crop_eye(img, center, w=60, h=36):
    cx, cy = map(int, center)
    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(cx + w // 2, img.shape[1])
    y2 = min(cy + h // 2, img.shape[0])
    return img[y1:y2, x1:x2]


def find_pupil_center(eye_img):
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None

    px = M["m10"] / M["m00"]
    py = M["m01"] / M["m00"]
    return px, py

# ---------------- CAMERA LOOP ----------------
def camera_loop():
    global latest_gaze

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ✅ FIX: mirror camera
        frame = cv2.flip(frame, 1)

        results = model(frame, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # LEFT eye (heuristic)
                left_eye_center = (
                    x1 + int(0.3 * w),
                    y1 + int(0.35 * h)
                )

                left_eye = crop_eye(frame, left_eye_center)
                if left_eye.size == 0:
                    continue

                left_eye = cv2.resize(left_eye, (EYE_W, EYE_H))
                pupil = find_pupil_center(left_eye)

                if pupil:
                    px, py = pupil

                    cv2.circle(
                        left_eye,
                        (int(px), int(py)),
                        3,
                        (0, 0, 255),
                        -1
                    )

                    latest_gaze["px"] = float(px)
                    latest_gaze["py"] = float(py)
                    latest_gaze["confidence"] = 1.0

                    print(
                        f"Pupil (crop) -> "
                        f"x={px:.6f}, y={py:.6f}"
                    )

                cv2.imshow("Left Eye (Crop)", left_eye)

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import uvicorn

    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000),
        daemon=True
    )
    server_thread.start()

    camera_loop()
