import time
import json
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# -----------------------------
# Config
# -----------------------------
PREDICT_INTERVAL_SEC = 2.0   # wait time between predictions
IMG_SIZE = 224

ARTIFACTS = Path("artifacts")
MODEL_PATH = ARTIFACTS / "best_model.pt"
LABEL_MAP_PATH = ARTIFACTS / "label_map.json"


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

    # Safe load
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


@torch.inference_mode()
def predict_face(model, tfm, device, face_bgr):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(face_rgb)
    x = tfm(pil).unsqueeze(0).to(device)

    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    idx = int(torch.argmax(prob).item())
    conf = float(prob[idx].item())
    return idx, conf


# -----------------------------
# Utils
# -----------------------------
def clamp_box(x, y, w, h, W, H):
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return x, y, w, h


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm, idx_to_label = load_model(device)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Realtime Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_predict_time = 0.0
    last_results = {}  # face_id -> (label, conf)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            H, W = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )

            for i, (x, y, w, h) in enumerate(faces):
                x, y, w, h = clamp_box(x, y, w, h, W, H)
                face_bgr = frame[y:y + h, x:x + w]
                if face_bgr.size == 0:
                    continue

                # predict only every N seconds
                if (now - last_predict_time) >= PREDICT_INTERVAL_SEC:
                    idx, conf = predict_face(model, tfm, device, face_bgr)
                    last_results[i] = (idx, conf)
                    last_predict_time = now

                if i in last_results:
                    idx, conf = last_results[i]
                    label = idx_to_label[idx]
                    text = f"{label} ({conf:.2f})"
                else:
                    text = "..."

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_label(frame, text, x, y)

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
