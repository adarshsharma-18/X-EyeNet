import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse


# ======================================================
# EfficientNet-B3 Model (same as training)
# ======================================================
import timm

class EfficientNetB3Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b3',
            pretrained=False,      # using trained weights instead
            num_classes=0,
            global_pool='avg'
        )
        self.fc = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


# ======================================================
# FUNDUS PREPROCESSING PIPELINE
# ======================================================
IMG_SIZE = 384

def crop_black_borders(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]

def circle_crop(img):
    h, w = img.shape[:2]
    radius = min(h, w) // 2
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w//2, h//2), radius, 255, -1)
    return cv2.bitwise_and(img, img, mask=mask)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def preprocess_fundus_web(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = crop_black_borders(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = circle_crop(img)
    img = apply_clahe(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), 30), -4, 128)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


# ======================================================
# MODEL LOADING
# ======================================================
def load_model(model_path, device):
    model = EfficientNetB3Regression().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# ======================================================
# PREDICT FUNCTION
# ======================================================
def predict_diabetic_retinopathy(image_path, model, device):
    # preprocess
    img = preprocess_fundus_web(image_path)

    # convert to tensor
    img_np = img.astype("float32") / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))  # CHW
    img_tensor = torch.tensor(img_np).unsqueeze(0).to(device)

    # inference
    with torch.no_grad():
        output = model(img_tensor).squeeze(0)

    raw_value = float(output.item())
    dr_class = int(np.clip(round(raw_value), 0, 4))

    return raw_value, dr_class, img


# ======================================================
# MAIN SCRIPT ENTRY POINT
# ======================================================
def main():
    # parser = argparse.ArgumentParser(description="Diabetic Retinopathy Prediction")
    # parser.add_argument("--image", type=str, required=True, help="Path to input retina image")
    # parser.add_argument("--model", type=str, required=True, help="Path to trained model (.pth)")
    # args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_model(args.model, device)
    print("Model Loaded.")

    print("Processing image...")
    raw_value, dr_class, preprocessed_img = predict_diabetic_retinopathy("image.png","C:\Users\Adarsh Sharma\OneDrive\Desktop\major projetc\X-EyeNet\efficientnetb3_best.pth", device)

    print("\n===========================")
    print("Prediction Results")
    print("===========================\n")
    print(f"Raw Output Score: {raw_value:.4f}")
    print(f"Predicted DR Class: {dr_class} (0–4 scale)\n")

    # show preprocessed image preview
    import matplotlib.pyplot as plt
    plt.imshow(preprocessed_img)
    plt.title(f"Predicted DR Class: {dr_class}")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
