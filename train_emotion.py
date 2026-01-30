# train_emotion.py
# - Loads dataset
# - Stratified split train/val/test
# - Strong augmentation (crop/affine/perspective/blur/erasing)
# - Class-weighted loss (handles imbalance)
# - Two-phase training: (1) train head with frozen backbone, (2) fine-tune backbone with low LR
# - ReduceLROnPlateau scheduler + EarlyStopping
# - Evaluate on test: accuracy, confusion matrix, classification report
# - Save best model + final model + label_map + plots

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt


@dataclass
class CFG:
    data_dir: str = "dataset"
    out_dir: str = "artifacts"
    seed: int = 42

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 2

    # training schedule
    head_epochs: int = 5          # phase 1: freeze backbone, train classifier
    finetune_epochs: int = 45     # phase 2: unfreeze, fine-tune
    head_lr: float = 1e-3
    finetune_lr: float = 3e-5
    weight_decay: float = 1e-4

    # early stopping on val_acc
    early_stop_patience: int = 6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int):
    # EfficientNet-B0 pretrained
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def compute_class_weights_from_indices(
    targets: np.ndarray, train_indices: np.ndarray, num_classes: int, device: str
):
    train_labels = targets[train_indices]
    counts = Counter(train_labels.tolist())

    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        weights[c] = 1.0 / max(1, counts.get(c, 0))

    # normalize so average weight ~ 1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device), counts


@torch.inference_mode()
def run_eval(model, loader, device: str, criterion=None):
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        if criterion is not None:
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

        pred = torch.argmax(logits, dim=1)
        all_pred.extend(pred.detach().cpu().numpy().tolist())
        all_true.extend(y.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    avg_loss = (total_loss / len(loader.dataset)) if criterion is not None else None
    return avg_loss, acc, all_true, all_pred


def plot_curves(history: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(out_dir / "loss_curve.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(out_dir / "acc_curve.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_plot(cm: np.ndarray, class_names: list[str], out_path: Path):
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=90)
    plt.yticks(ticks, class_names)
    plt.tight_layout()
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    cfg = CFG()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Transforms
    # -----------------------------
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=5
            )
        ], p=0.7),
        transforms.RandomApply([
            transforms.RandomPerspective(distortion_scale=0.2)
        ], p=0.2),
        transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -----------------------------
    # Dataset + split
    # -----------------------------
    base_ds = datasets.ImageFolder(cfg.data_dir)
    class_names = base_ds.classes
    num_classes = len(class_names)

    label_map = {i: name for i, name in enumerate(class_names)}
    (out_dir / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    targets = np.array([y for _, y in base_ds.samples])
    indices = np.arange(len(base_ds))

    train_idx, test_idx = train_test_split(
        indices, test_size=0.15, random_state=cfg.seed, stratify=targets
    )
    train_targets = targets[train_idx]
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.15, random_state=cfg.seed, stratify=train_targets
    )

    train_ds = Subset(datasets.ImageFolder(cfg.data_dir, transform=train_tfms), train_idx)
    val_ds   = Subset(datasets.ImageFolder(cfg.data_dir, transform=eval_tfms), val_idx)
    test_ds  = Subset(datasets.ImageFolder(cfg.data_dir, transform=eval_tfms), test_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    # -----------------------------
    # Model + loss
    # -----------------------------
    model = build_model(num_classes).to(device)

    class_weights, train_counts = compute_class_weights_from_indices(
        targets=targets, train_indices=train_idx, num_classes=num_classes, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    counts_txt = "\n".join([f"{class_names[k]}: {train_counts.get(k, 0)}" for k in range(num_classes)])
    (out_dir / "train_class_counts.txt").write_text(counts_txt, encoding="utf-8")

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    best_val_acc = -1.0
    best_path = out_dir / "best_model.pt"

    total_epochs = cfg.head_epochs + cfg.finetune_epochs

    def train_one_epoch(epoch_idx: int, optimizer):
        model.train()
        total_loss = 0.0
        all_pred, all_true = [], []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch_idx}/{total_epochs} [train]"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = torch.argmax(logits, dim=1)
            all_pred.extend(pred.detach().cpu().numpy().tolist())
            all_true.extend(y.detach().cpu().numpy().tolist())

        avg_loss = total_loss / len(train_ds)
        acc = accuracy_score(all_true, all_pred)
        return avg_loss, acc

    # -----------------------------
    # Phase 1: freeze backbone, train head
    # -----------------------------
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.head_lr,
        weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    patience_left = cfg.early_stop_patience

    epoch_counter = 0
    for _ in range(cfg.head_epochs):
        epoch_counter += 1

        tr_loss, tr_acc = train_one_epoch(epoch_counter, optimizer)
        va_loss, va_acc, _, _ = run_eval(model, val_loader, device, criterion=criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch_counter}/{total_epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered (phase 1).")
                break

    # -----------------------------
    # Phase 2: unfreeze backbone, fine-tune
    # -----------------------------
    patience_left = cfg.early_stop_patience

    for p in model.features.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    remaining = total_epochs - epoch_counter
    for _ in range(remaining):
        epoch_counter += 1

        tr_loss, tr_acc = train_one_epoch(epoch_counter, optimizer)
        va_loss, va_acc, _, _ = run_eval(model, val_loader, device, criterion=criterion)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch_counter}/{total_epochs} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping triggered (phase 2).")
                break

    plot_curves(history, out_dir)
    (out_dir / "lr_history.txt").write_text("\n".join([f"{lr:.10f}" for lr in history["lr"]]), encoding="utf-8")

    # -----------------------------
    # Test evaluation for best model
    # -----------------------------
    state_dict = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    _, test_acc, y_true, y_pred = run_eval(model, test_loader, device, criterion=None)

    print("\nTEST ACC:", test_acc)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", report)

    (out_dir / "test_accuracy.txt").write_text(f"{test_acc:.6f}\n", encoding="utf-8")
    np.save(out_dir / "confusion_matrix.npy", cm)
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix_plot(cm, class_names, out_dir / "confusion_matrix.png")

    torch.save(model.state_dict(), out_dir / "model_final.pt")

    print(f"\nSaved best model: {best_path}")
    print(f"Saved final model: {out_dir / 'model_final.pt'}")
    print(f"Saved plots/metrics in: {out_dir.resolve()}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
