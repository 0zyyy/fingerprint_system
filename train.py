#!/usr/bin/env python3
"""
Training script: HOG feature extraction → PyTorch ANN → ONNX export

Dataset layout expected:
  <DATASET_ROOT>/
    NRML/   MIDFinger_1.bmp ... MIDFinger_50.bmp   (subjects 0-9, 5 imgs each)
    OILY/   ...
    DST/    ...
    WT/     ...

Subject ID derived from filename index: (idx - 1) // IMGS_PER_SUBJECT
"""

import os
import re
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix

from config import HOG_CONFIG, MODEL_CONFIG
from core.preprocessing import ImagePreprocessor
from core.feature_extraction import HOGExtractor
from models.ann_model import FingerprintANN, TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

DATASET_ROOT = Path("SHIDIQ JARI TENGAH KNN/SHIDIQ JARI TENGAH KNN")
CONDITIONS = ["NRML", "OILY", "DST", "WT"]
IMGS_PER_SUBJECT = 5   # 50 images / 10 subjects


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def parse_subject_id(filename: str, imgs_per_subject: int = IMGS_PER_SUBJECT) -> int:
    """MIDFinger_N.bmp → subject_id = (N-1) // imgs_per_subject"""
    match = re.search(r"_(\d+)\.", filename)
    if not match:
        raise ValueError(f"Cannot parse subject ID from: {filename}")
    idx = int(match.group(1))
    return (idx - 1) // imgs_per_subject


def load_dataset(dataset_root: Path = DATASET_ROOT) -> tuple[np.ndarray, np.ndarray]:
    preprocessor = ImagePreprocessor()
    extractor = HOGExtractor()

    features, labels = [], []
    skipped = 0

    for condition in CONDITIONS:
        condition_dir = dataset_root / condition
        if not condition_dir.exists():
            logger.warning(f"Condition dir not found, skipping: {condition_dir}")
            continue

        bmp_files = sorted(condition_dir.glob("*.bmp"))
        logger.info(f"{condition}: {len(bmp_files)} files")

        for bmp_path in bmp_files:
            try:
                subject_id = parse_subject_id(bmp_path.name)
            except ValueError as e:
                logger.warning(str(e))
                skipped += 1
                continue

            img = cv2.imread(str(bmp_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Could not read: {bmp_path}")
                skipped += 1
                continue

            processed = preprocessor.process(img)
            hog_vec = extractor.extract(processed)

            features.append(hog_vec.astype(np.float32))
            labels.append(subject_id)

    if skipped:
        logger.warning(f"Skipped {skipped} files")

    X = np.stack(features)
    y = np.array(labels, dtype=np.int64)
    logger.info(f"Dataset: {X.shape[0]} samples, {len(np.unique(y))} subjects, feature dim={X.shape[1]}")
    return X, y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: TrainingConfig, dataset_root: Path = DATASET_ROOT, device: str = "cpu"):
    X, y = load_dataset(dataset_root)

    n_subjects = len(np.unique(y))
    assert n_subjects == MODEL_CONFIG["OUTPUT_DIM"], (
        f"Found {n_subjects} subjects but OUTPUT_DIM={MODEL_CONFIG['OUTPUT_DIM']}. "
        "Update config.py or check IMGS_PER_SUBJECT."
    )

    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model = FingerprintANN(
        input_dim=MODEL_CONFIG["INPUT_DIM"],
        hidden_layers=MODEL_CONFIG["HIDDEN_LAYERS"],
        output_dim=MODEL_CONFIG["OUTPUT_DIM"],
        dropout=config.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma
    )

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    logger.info(f"Training on {train_size} samples, validating on {val_size}")
    logger.info(f"Device: {device} | Epochs: {config.epochs} | LR: {config.learning_rate}")

    for epoch in range(1, config.epochs + 1):
        # --- train ---
        model.train()
        train_loss, train_correct = 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()

        # --- validate ---
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()

        train_acc = train_correct / train_size
        val_acc = val_correct / val_size
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Loss {train_loss/train_size:.4f} | Acc {train_acc:.4f} | "
            f"Val Loss {val_loss/val_size:.4f} | Val Acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch} (best val_acc={best_val_acc:.4f})")
                break

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")

    # Restore best weights
    model.load_state_dict(best_state)
    return model, X, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model: FingerprintANN, X: np.ndarray, y: np.ndarray, device: str = "cpu"):
    model.eval()
    X_t = torch.from_numpy(X).to(device)
    with torch.no_grad():
        logits = model(X_t)
        preds = logits.argmax(1).cpu().numpy()

    print("\n=== Classification Report ===")
    print(classification_report(y, preds))
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y, preds))


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: FingerprintANN, output_path: str = "models/fingerprint_ann.onnx"):
    model.eval()
    dummy = torch.zeros(1, MODEL_CONFIG["INPUT_DIM"])

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["hog_features"],
        output_names=["logits"],
        dynamic_axes={"hog_features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info(f"ONNX model exported to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HOG+ANN fingerprint classifier")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dataset", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--output", type=str, default="models/fingerprint_ann.onnx")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    args = parser.parse_args()

    cfg = TrainingConfig(
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dropout=args.dropout,
        early_stopping_patience=args.patience,
    )

    model, X, y = train(cfg, dataset_root=Path(args.dataset), device=args.device)
    evaluate(model, X, y, device=args.device)

    if not args.no_export:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        export_onnx(model, args.output)
