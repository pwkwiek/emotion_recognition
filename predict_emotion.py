import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = Path("dataset_private")
IN_DIR = DATASET_DIR / "input_images"
OUT_DIR = DATASET_DIR / "output_images"

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "best_model.pt"
LABEL_MAP_PATH = ARTIFACTS / "label_map.json"

IMG_SIZE = 224


# -----------------------------
# Model
# -----------------------------
def build_model(num_classes: int):
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def load_model(device: str):
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    idx_to_label = {int(k): v for k, v in label_map.items()}
    num_classes = len(idx_to_label)

    model = build_model(num_classes).to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, tfm, idx_to_label


# -----------------------------
# Face detection
# -----------------------------
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_faces(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    return faces


def clamp_box(x, y, w, h, W, H):
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


# -----------------------------
# Inference
# -----------------------------
@torch.inference_mode()
def predict_pil(model, tfm, device: str, pil_img: Image.Image):
    x = tfm(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(prob).item())
    conf = float(prob[idx].item())
    return idx, conf


def draw_label(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)

    y1 = max(0, y - th - base - 6)
    x2 = min(img.shape[1], x + tw + 8)
    y2 = min(img.shape[0], y + 6)

    cv2.rectangle(img, (x, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 4, y2 - 4),
                font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm, idx_to_label = load_model(device)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        image_paths.extend(IN_DIR.glob(ext))

    if not image_paths:
        raise RuntimeError(f"No images found in {IN_DIR.resolve()}")

    for p in image_paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Unreadable image: {p}")

        H, W = bgr.shape[:2]
        out = bgr.copy()

        faces = detect_faces(out)

        # STRICT REQUIREMENT (No face detected)
        if len(faces) == 0:
            raise RuntimeError(f"No face detected in image: {p.name}")

        for (x, y, w, h) in faces:
            x, y, w, h = clamp_box(x, y, w, h, W, H)
            face_bgr = out[y:y+h, x:x+w]
            if face_bgr.size == 0:
                continue

            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(face_rgb)

            idx, conf = predict_pil(model, tfm, device, pil)
            label = idx_to_label[idx]

            cv2.rectangle(out, (x, y), (x+w, y+h), (0, 255, 0), 2)
            draw_label(out, f"{label} ({conf:.2f})", x, y)

        out_path = OUT_DIR / p.name
        cv2.imwrite(str(out_path), out)
        print("Saved:", out_path)

    print("All done.")


if __name__ == "__main__":
    main()
