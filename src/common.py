"""Đường dẫn, đọc dữ liệu và tính metric dùng chung cho các script."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
METRICS_FILE = MODELS_DIR / "metrics.json"

LABEL_NAMES = {0: "REAL", 1: "FAKE"}  # nhãn 1 = tin giả


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise SystemExit(f"Không thấy {path}\nHãy chạy trước:  python src/prepare_data.py")
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("").astype(str)
    df["label"] = df["label"].astype(int)
    return df


def compute_metrics(y_true, y_pred) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
    }


def confusion(y_true, y_pred) -> list:
    """[[REAL đoán đúng, REAL đoán nhầm FAKE], [FAKE đoán nhầm REAL, FAKE đoán đúng]]"""
    return confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()


def print_metrics(title: str, m: dict) -> None:
    print(f"  {title:<6} acc={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
          f"recall={m['recall']:.4f}  f1={m['f1']:.4f}")


def save_metrics(model_key: str, results: dict) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if METRICS_FILE.exists():
        try:
            data = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[model_key] = results
    METRICS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
