import numpy as np
import dlib
import cv2
import os

# Correct paths
path_model = "models/shape_predictor_68_face_landmarks.dat"
path_image = "extension/icons/icon16.png"   # or any face image

# Safety check (important)
assert os.path.exists(path_model), "Model file not found!"
assert os.path.exists(path_image), "Image file not found!"

# Load dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(path_model)

# Load image
im = cv2.imread(path_image)
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = detector(gray)

# Landmark
landmarker = [37, 40, 43, 46, 49, 55 ,9, 1,17]  # Example: eyes corners

for face in faces:
    landmarks = predictor(gray, face)
    for i in landmarker:
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(im, (x, y), 2, (0, 255, 0), -1)

cv2.imshow("output", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
