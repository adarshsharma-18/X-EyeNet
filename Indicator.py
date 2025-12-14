from eyecroptool import *



left_eye_norm = normalize_eye(left_eye_crop)
right_eye_norm = normalize_eye(right_eye_crop)

cv2.imwrite(f"{SAVE_DIR}/left_eye.jpg", cv2.cvtColor(left_eye_crop, cv2.COLOR_RGB2BGR))
cv2.imwrite(f"{SAVE_DIR}/right_eye.jpg", cv2.cvtColor(right_eye_crop, cv2.COLOR_RGB2BGR))

def redness_index(eye_img):
    eye = eye_img.astype(np.float32)
    R, G, B = eye[:,:,0], eye[:,:,1], eye[:,:,2]
    redness = R / (R + G + B + 1e-6)
    return redness
def plot_redness_heatmap(eye_img, title="Redness Heatmap"):
    redness = redness_index(eye_img)
    plt.imshow(redness, cmap="jet")
    plt.colorbar()
    plt.title(title)
    plt.axis("off")
    plt.show()

plot_redness_heatmap(left_eye_crop, "Left Eye Redness Heatmap")
plot_redness_heatmap(right_eye_crop, "Right Eye Redness Heatmap")

def yellow_index(eye_img):
    lab = cv2.cvtColor(eye_img, cv2.COLOR_RGB2LAB)
    b_channel = lab[:,:,2]   # b* channel
    return np.mean(b_channel)

print("Left Eye Yellow Index:", yellow_index(left_eye_crop))
print("Right Eye Yellow Index:", yellow_index(right_eye_crop))

def compute_EAR(landmarks, eye_indices, img_shape):
    h, w, _ = img_shape

    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * w, lm.y * h]))

    pts = np.array(pts)

    vertical1 = np.linalg.norm(pts[1] - pts[5])
    vertical2 = np.linalg.norm(pts[2] - pts[4])
    horizontal = np.linalg.norm(pts[0] - pts[3])

    EAR = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
    return EAR

EAR_left = compute_EAR(face_landmarks, LEFT_EYE, image.shape)
EAR_right = compute_EAR(face_landmarks, RIGHT_EYE, image.shape)

print("Left EAR:", EAR_left)
print("Right EAR:", EAR_right)
print("EAR Difference:", abs(EAR_left - EAR_right))

from skimage.filters import frangi
from skimage.color import rgb2gray

def vessel_map(eye_img):
    gray = rgb2gray(eye_img)
    vessels = frangi(gray)
    return vessels

def plot_vessel_map(eye_img, title="Vessel Map"):
    vessels = vessel_map(eye_img)
    plt.imshow(vessels, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

plot_vessel_map(left_eye_crop, "Left Eye Vessel Map")
plot_vessel_map(right_eye_crop, "Right Eye Vessel Map")

features = {
    "EAR_left": EAR_left,
    "EAR_right": EAR_right,
    "EAR_diff": abs(EAR_left - EAR_right),
    "Redness_left": np.mean(redness_index(left_eye_crop)),
    "Redness_right": np.mean(redness_index(right_eye_crop)),
    "Yellow_left": yellow_index(left_eye_crop),
    "Yellow_right": yellow_index(right_eye_crop)
}

print(features)
