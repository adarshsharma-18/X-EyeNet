# X-EyeNet: AI-Based Diabetic Retinopathy Screening System

This project is a multi-modal AI application designed to predict the presence and severity of Diabetic Retinopathy (DR) using two different types of eye images: Fundus images and External eye images.

The backend consists of two separate Flask microservices, each handling one specific type of image with its own dedicated deep learning model and Python environment.

## Project Structure & APIs

### 1. External Eye API
- **Purpose**: Classifies external eye images as "normal" or "abnormal".
- **File**: `external_eye_flask_api.py`
- **Model**: ResNet-based model built with TensorFlow/Keras (`external_eye_resnet_model.h5`).
- **Endpoint**: `http://localhost:8002/predict_external_eye`
- **Environment**: Global System Python
- **Python Version**: `3.12.1`

#### Required Libraries (External Eye)
| Library | Version | Purpose |
| :--- | :--- | :--- |
| `tensorflow` | 2.16.0rc0 | Loads and runs the ResNet deep learning model (`.h5`). |
| `Flask` | 3.1.2 | Provides the web server framework to expose the REST API. |
| `Flask-Cors` | 4.0.0 | Enables Cross-Origin Resource Sharing for frontend communication. |
| `numpy` | 1.26.3 | Handles numerical operations and image array transformations. |
| `opencv-python` | 4.9.0.80 | Reads, decodes, and resizes the incoming image streams (`cv2`). |

---

### 2. Fundus Image API
- **Purpose**: Predicts the stage of Diabetic Retinopathy (0 to 4) using Fundus images.
- **File**: `fundus_flask_api.py`
- **Model**: EfficientNetB3 with CORAL architecture built with PyTorch and timm (`FINAL_EfficientNetB3_CORAL.pth`).
- **Endpoint**: `http://localhost:8001/predict_fundus`
- **Environment**: Local Virtual Environment (`.venv`)
- **Python Version**: `3.10.9`

#### Required Libraries (Fundus Image)
| Library | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | 2.5.1+cu121 | PyTorch framework to run the deep learning model. |
| `torchvision` | 0.20.1 | PyTorch computer vision utilities. |
| `timm` | 1.0.22 | Provides the pre-built EfficientNetB3 backbone. |
| `Flask` | 3.0.2 | Provides the web server framework for the REST API. |
| `Flask-Cors` | 4.0.1 | Enables Cross-Origin Resource Sharing. |
| `numpy` | 1.23.5 | Matrix operations and thresholds handling. |
| `opencv-python` | 4.7.0.72 | Handles image decoding and basic preprocessing. |
| `Pillow` | 10.2.0 | Image processing library (dependency for torchvision/timm). |

---

## Installation & Setup

Because this project utilizes two separate architectures (TensorFlow for external eye, PyTorch for fundus), it requires two different environments to prevent dependency conflicts (e.g., CUDA version clashes between TF and PyTorch).

### 1. External Eye Environment (System)
Make sure you have Python 3.12.1 installed globally.
```bash
# Install dependencies for the external eye API
pip install tensorflow==2.16.0rc0 Flask==3.1.2 Flask-Cors==4.0.0 numpy==1.26.3 opencv-python==4.9.0.80
```

### 2. Fundus Image Environment (Virtual Environment)
The project comes with a `.venv` folder configured with Python 3.10.9.
Activate the virtual environment and install its dependencies:

```bash
# Windows
.\.venv\Scripts\activate

# Install dependencies from the requirements file
pip install -r requirements.txt
```
*(Note: The `requirements.txt` file is specifically tailored for the Fundus API environment).*

---

## Running the Servers

You need to run both APIs concurrently in separate terminal windows.

**Terminal 1: Start Fundus API**
```bash
# Activate virtual environment
.\.venv\Scripts\activate
# Run the server
python fundus_flask_api.py
```
*Runs on `http://localhost:8001`*

**Terminal 2: Start External Eye API**
```bash
# Ensure you are using the global system Python 3.12.1
python external_eye_flask_api.py
```
*Runs on `http://localhost:8002`*

---

## API Usage Example

### 1. Fundus Prediction
```bash
curl -X POST "http://localhost:8001/predict_fundus" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "fundus_image=@/path/to/fundus_image.jpg"
```

**Response Example:**
```json
{
  "dr_stage": 2,
  "confidence": 0.85,
  "probabilities": [0.99, 0.95, 0.85, 0.12],
  "stage_meaning": "Moderate Non-Proliferative DR",
  "timestamp": "2024-05-04T12:00:00"
}
```

### 2. External Eye Prediction
```bash
curl -X POST "http://localhost:8002/predict_external_eye" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "external_eye_image=@/path/to/external_image.jpg"
```

**Response Example:**
```json
{
  "label": "abnormal",
  "confidence": 0.92,
  "probabilities": [0.08, 0.92],
  "output_type": "softmax",
  "timestamp": "2024-05-04T12:05:00"
}
```

## Troubleshooting
- **Model not found error**: Ensure `external_eye_resnet_model.h5` and `FINAL_EfficientNetB3_CORAL.pth` / `FINAL_CORAL_thresholds.npy` are placed in the correct directories as specified by the respective Python files.
- **Port already in use**: If ports 8001 or 8002 are occupied, edit the `app.run(port=...)` lines at the bottom of the Python scripts.

- # ---------------------------
# CORE VERSIONS (DO NOT CHANGE)
# ---------------------------
python_version == "3.10"
tensorflow==2.10.0
tensorflow-addons==0.20.0
numpy==1.23.5

# ---------------------------
# COMPUTER VISION (SAFE VERSIONS)
# ---------------------------
opencv-python==4.7.0.72
opencv-python-headless==4.7.0.72
scikit-image==0.19.3

# ---------------------------
# AUGMENTATION
# ---------------------------
albumentations==1.3.1
qudida==0.0.4
PyYAML==6.0.1

# ---------------------------
# ML / DATA PROCESSING
# ---------------------------
scikit-learn==1.3.0
pandas==2.0.3
matplotlib==3.7.1

# ---------------------------
# OPTIONAL UTILITIES
# ---------------------------
tqdm
