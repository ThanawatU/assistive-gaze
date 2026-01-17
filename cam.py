import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "models/yolov12n-face.pt"
CAMERA_ID = 1
EYE_W, EYE_H = 60, 36

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(CAMERA_ID)

# ---------------- UTILS ----------------
def crop_eye(img, center, w=60, h=36):
    cx, cy = map(int, center)

    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(cx + w // 2, img.shape[1])
    y2 = min(cy + h // 2, img.shape[0])

    return img[y1:y2, x1:x2]


def find_pupil_center(eye_img):
    """
    คืนค่า (x, y) แบบ FLOAT
    ตำแหน่งอยู่ในกรอบ eye crop เท่านั้น
    """
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

    # ✅ SUB-PIXEL (FLOAT)
    px = M["m10"] / M["m00"]
    py = M["m01"] / M["m00"]

    return px, py

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # ---- LEFT EYE CENTER (heuristic) ----
            left_eye_center = (
                x1 + int(0.3 * w),
                y1 + int(0.35 * h)
            )

            # ---- CROP LEFT EYE ----
            left_eye = crop_eye(frame, left_eye_center)
            if left_eye.size == 0:
                continue

            left_eye = cv2.resize(left_eye, (EYE_W, EYE_H))
            pupil = find_pupil_center(left_eye)

            if pupil:
                px, py = pupil  # FLOAT

                # วาดจุดแดง (ต้อง cast เป็น int แค่ตอนวาด)
                cv2.circle(
                    left_eye,
                    (int(px), int(py)),
                    3,
                    (0, 0, 255),
                    -1
                )

                # ✅ PRINT ตำแหน่งจุดแดงในกรอบเล็ก
                print(
                    f"Left eye pupil (crop): "
                    f"x = {px:.8f}, y = {py:.8f}"
                )

            cv2.imshow("Left Eye (Crop)", left_eye)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
