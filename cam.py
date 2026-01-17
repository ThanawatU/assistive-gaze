import cv2
import dlib
import numpy as np
from ultralytics import YOLO
import os
import torch
import asyncio
import threading
from fastapi import FastAPI, WebSocket
from gaze_pred_model import GazeNN

# ---------------- CONFIG ----------------
MODEL_PATH = "models/yolov12n-face.pt"
CAMERA_ID = 1
EYE_W, EYE_H = 60, 36
LANDMARK_EVERY = 3
frame_id = 0

# ---------------- GLOBAL STATE (XYZ) ----------------
latest_gaze = {
    "gx": 0.0,
    "gy": 0.0,
    "gz": 1.0
}

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_ID)

gaze_model = GazeNN()
state_dict = torch.load("models/best_gaze_model.pth", map_location="cpu")
gaze_model.load_state_dict(state_dict)
gaze_model.eval()

model_face_landmark = "models/shape_predictor_68_face_landmarks.dat"
assert os.path.exists(model_face_landmark)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_face_landmark)

# ---------------- FASTAPI ----------------
app = FastAPI()

@app.websocket("/gaze")
async def gaze_ws(ws: WebSocket):
    await ws.accept()
    while True:
        await ws.send_json(latest_gaze)
        await asyncio.sleep(0.03)  # ~30 FPS

# ---------------- UTILS ----------------
def crop_eye(img, center, w=60, h=36):
    cx, cy = int(center[0]), int(center[1])
    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(cx + w // 2, img.shape[1])
    y2 = min(cy + h // 2, img.shape[0])
    return img[y1:y2, x1:x2]

def face_landmark(frame, face_box):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = face_box
    rect = dlib.rectangle(x1, y1, x2, y2)
    landmarks = predictor(gray, rect)

    for i in [36, 39, 42, 45]:
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

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

    return (
        int(M["m10"] / M["m00"]),
        int(M["m01"] / M["m00"])
    )

# ---------------- CAMERA LOOP ----------------
def camera_loop():
    global frame_id, latest_gaze

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=416, conf=0.5, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                if frame_id % LANDMARK_EVERY == 0:
                    face_landmark(frame, (x1, y1, x2, y2))

                    left_eye_center = (x1 + int(0.3 * w), y1 + int(0.35 * h))
                    right_eye_center = (x1 + int(0.7 * w), y1 + int(0.35 * h))

                    left_eye = crop_eye(frame, left_eye_center)
                    right_eye = crop_eye(frame, right_eye_center)

                    if left_eye.size == 0 or right_eye.size == 0:
                        continue

                    left_eye = cv2.resize(left_eye, (EYE_W, EYE_H))
                    right_eye = cv2.resize(right_eye, (EYE_W, EYE_H))

                    # ---------------- GAZE MODEL ----------------
                    gaze_input = torch.randn(1, 2, EYE_H, EYE_W)
                    head_pose = torch.randn(1, 3)

                    with torch.inference_mode():
                        gaze_output = gaze_model(gaze_input, head_pose)

                    # gaze_output = [x, y, z]
                    gx, gy, gz = gaze_output[0].tolist()

                    latest_gaze["gx"] = float(gx)
                    latest_gaze["gy"] = float(gy)
                    latest_gaze["gz"] = float(gz)

                    print("Gaze XYZ:", latest_gaze)

        cv2.imshow("Webcam", frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    import uvicorn

    # WebSocket server in background
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000),
        daemon=True
    )
    server_thread.start()

    # Camera + OpenCV MUST be main thread
    camera_loop()
