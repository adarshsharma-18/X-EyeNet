import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm

# ===============================
# CONFIGURATION (EDIT IF NEEDED)
# ===============================

IMAGE_PATH = r"C:\Users\Adarsh Sharma\OneDrive\Desktop\major projetc\X-EyeNet\file format\image.png"
MODEL_PATH = r"C:\Users\Adarsh Sharma\OneDrive\Desktop\major projetc\X-EyeNet\FINAL_EfficientNetB3_CORAL.pth"
THRESHOLD_PATH = r"C:\Users\Adarsh Sharma\OneDrive\Desktop\major projetc\X-EyeNet\FINAL_CORAL_thresholds.npy"

IMG_SIZE = 384
NUM_CLASSES = 5

# ===============================
# DEVICE
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===============================
# MODEL DEFINITION (CORAL)
# ===============================

class EfficientNetB3_CORAL(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=0
        )
        self.fc = nn.Linear(self.backbone.num_features, num_classes - 1)

    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features)
        return logits

# ===============================
# LOAD MODEL + THRESHOLDS
# ===============================

def load_model_and_thresholds():
    model = EfficientNetB3_CORAL(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    thresholds = np.load(THRESHOLD_PATH)

    print("Model and thresholds loaded successfully")
    print("Thresholds:", thresholds)

    return model, thresholds

# ===============================
# IMAGE PREPROCESSING
# ===============================

def preprocess_image(image_path):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW

    img_tensor = torch.tensor(img).unsqueeze(0).to(device)
    return img_tensor

# ===============================
# CORAL PREDICTION LOGIC
# ===============================

def coral_predict(logits, thresholds):
    probs = torch.sigmoid(logits).cpu().numpy()[0]
    pred_class = sum(probs[i] > thresholds[i] for i in range(len(thresholds)))
    return int(pred_class), probs

# ===============================
# MAIN PREDICTION FUNCTION
# ===============================

def predict_dr_stage(image_path):
    model, thresholds = load_model_and_thresholds()
    img_tensor = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(img_tensor)
        pred_class, probs = coral_predict(logits, thresholds)

    return pred_class, probs

# ===============================
# RUN SCRIPT
# ===============================

if __name__ == "__main__":
    print("\nRunning Diabetic Retinopathy Prediction...\n")

    pred_class, probs = predict_dr_stage(IMAGE_PATH)

    print("====================================")
    print(" Predicted Diabetic Retinopathy Stage")
    print("====================================")
    print(f"Image Path : {IMAGE_PATH}")
    print(f"Predicted Stage : {pred_class}")
    print("Stage Meaning:")
    print("0 → No DR")
    print("1 → Mild")
    print("2 → Moderate")
    print("3 → Severe")
    print("4 → Proliferative DR")
    print("------------------------------------")
    print("CORAL Probabilities:", probs)
    print("====================================")
