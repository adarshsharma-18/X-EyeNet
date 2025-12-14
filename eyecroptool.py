import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

image = cv2.imread("C:/Users/Adarsh Sharma/OneDrive/Desktop/major projetc/X-EyeNet/dataset/train_1/train_1/1629243_1.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
) as face_mesh:

    results = face_mesh.process(image_rgb)

annotated = image_rgb.copy()

if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        for lm in face_landmarks.landmark:
            h, w, _ = annotated.shape
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (x, y), 1, (0, 255, 0), -1)

plt.imshow(annotated)
plt.title("Face Mesh Visualization")
plt.axis("off")
plt.show()

LEFT_EYE = [362, 263, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 133, 160, 159, 158, 157, 173]

def get_eye_bbox(landmarks, indices, img_shape, pad=10):
    h, w, _ = img_shape
    points = []

    for idx in indices:
        lm = landmarks[idx]
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))

    points = np.array(points)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)

    return max(0, x_min-pad), max(0, y_min-pad), min(w, x_max+pad), min(h, y_max+pad)

visual = image_rgb.copy()
face_landmarks = results.multi_face_landmarks[0].landmark

lx1, ly1, lx2, ly2 = get_eye_bbox(face_landmarks, LEFT_EYE, image.shape)
rx1, ry1, rx2, ry2 = get_eye_bbox(face_landmarks, RIGHT_EYE, image.shape)

cv2.rectangle(visual, (lx1, ly1), (lx2, ly2), (255, 0, 0), 2)
cv2.rectangle(visual, (rx1, ry1), (rx2, ry2), (255, 0, 0), 2)

plt.imshow(visual)
plt.title("Eye Region Localization")
plt.axis("off")
plt.show()

left_eye_crop = image_rgb[ly1:ly2, lx1:lx2]
right_eye_crop = image_rgb[ry1:ry2, rx1:rx2]

plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plt.imshow(left_eye_crop)
plt.title("Left Eye Crop")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(right_eye_crop)
plt.title("Right Eye Crop")
plt.axis("off")

plt.show()

import os

SAVE_DIR = "output_eyes"
os.makedirs(SAVE_DIR, exist_ok=True)

def normalize_eye(eye_img, size=(128,128)):
    eye_resized = cv2.resize(eye_img, size)
    eye_norm = eye_resized.astype(np.float32) / 255.0
    return eye_norm

left_eye_norm = normalize_eye(left_eye_crop)
right_eye_norm = normalize_eye(right_eye_crop)

cv2.imwrite(f"{SAVE_DIR}/left_eye.jpg", cv2.cvtColor(left_eye_crop, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{SAVE_DIR}/right_eye.jpg", cv2.cvtColor(right_eye_crop, cv2.COLOR_RGB2BGR))



