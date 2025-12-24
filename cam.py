import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/yolov12n-face.pt")
cap = cv2.VideoCapture(1)

EYE_W, EYE_H = 60, 36

def crop_eye(img, center, w=60, h=36):
    cx, cy = int(center[0]), int(center[1])
    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(cx + w // 2, img.shape[1])
    y2 = min(cy + h // 2, img.shape[0])
    crop = img[y1:y2, x1:x2]

    return crop, (x1, y1)

while True:
  ret, frame = cap.read()
  if not ret:
    break

  results = model(frame, conf=0.5, verbose=False)

  for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # Approximate eye positions inside face box
            left_eye_center = (x1 + int(0.3 * w), y1 + int(0.35 * h))
            right_eye_center = (x1 + int(0.7 * w), y1 + int(0.35 * h))


            left_eye = crop_eye(frame, left_eye_center)
            right_eye = crop_eye(frame, right_eye_center)

            #############################################################
            left_eye, (lx1, ly1) = crop_eye(frame, left_eye_center)
            right_eye, (rx1, ry1) = crop_eye(frame, right_eye_center)

            # center ในกรอบเล็ก
            left_eye_center_local = (
                left_eye_center[0] - lx1,
                left_eye_center[1] - ly1
            )

            right_eye_center_local = (
                right_eye_center[0] - rx1,
                right_eye_center[1] - ry1
            )

            print("Left eye center (local):", left_eye_center_local)
            print("Right eye center (local):", right_eye_center_local)
            #############################################################

            # Draw face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Draw eye centers
            cv2.circle(frame, left_eye_center, 3, (255,0,0), -1)
            cv2.circle(frame, right_eye_center, 3, (255,0,0), -1)

            if left_eye.size != 0:
                left_eye = cv2.resize(left_eye, (EYE_W, EYE_H))
                cv2.imshow("Left Eye", left_eye)

            if right_eye.size != 0:
                right_eye = cv2.resize(right_eye, (EYE_W, EYE_H))
                cv2.imshow("Right Eye", right_eye)

  cv2.imshow("Webcam", frame)
  if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
