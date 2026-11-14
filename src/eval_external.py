"""
Đánh giá các baseline đã huấn luyện trên bộ WELFake bên ngoài.

Mặc định:
- ISOT của dự án: 0 = REAL, 1 = FAKE
- WELFake: 0 = REAL, 1 = FAKE
- Dự án:  0 = REAL, 1 = FAKE
  Hai bộ dữ liệu dùng cùng quy ước nên không đảo nhãn.

Chạy từ thư mục gốc dự án:
    python src/eval_external.py
    python src/eval_external.py --models logistic svc
    python src/eval_external.py --sample 10000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from common import MODELS_DIR, PROCESSED_DIR, ROOT, compute_metrics
from text_utils import build_input


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Đánh giá baseline trên WELFake")
    ap.add_argument(
        "--data",
        default=str(ROOT / "data" / "external" / "WELFake_Dataset.csv"),
        help="Đường dẫn file WELFake CSV",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "svc"],
        default=["logistic", "svc"],
        help="Các baseline cần đánh giá",
    )
    ap.add_argument("--min-words", type=int, default=30,
                    help="Bỏ bài có phần nội dung ngắn hơn N từ")
    ap.add_argument("--sample", type=int, default=None,
                    help="Chỉ đánh giá ngẫu nhiên N dòng để chạy thử")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=5000,
                    help="Số văn bản biến đổi TF-IDF mỗi batch")
    ap.add_argument(
        "--keep-isot-overlap",
        action="store_true",
        help="Không loại bài trùng với các split ISOT",
    )
    ap.add_argument(
        "--output-dir",
        default=str(ROOT / "data" / "external" / "eval_results"),
        help="Thư mục lưu metric và các bài dự đoán sai",
    )
    return ap.parse_args()


def load_welfake(path: Path, min_words: int) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise SystemExit(f"Không thấy {path}")

    print(f"Đang đọc {path} ...")
    raw = pd.read_csv(path, usecols=lambda c: c in {"title", "text", "label"})
    required = {"title", "text", "label"}
    missing = required.difference(raw.columns)
    if missing:
        raise SystemExit(f"WELFake thiếu cột: {sorted(missing)}")

    stats: dict[str, int] = {"raw_rows": int(len(raw))}
    raw["title"] = raw["title"].fillna("").astype(str)
    raw["text"] = raw["text"].fillna("").astype(str)
    raw["label"] = pd.to_numeric(raw["label"], errors="coerce")

    bad_label = ~raw["label"].isin([0, 1])
    stats["removed_invalid_label"] = int(bad_label.sum())
    raw = raw.loc[~bad_label].copy()
    raw["welfake_label"] = raw["label"].astype(int)

    # WELFake và dự án đều dùng: 0=REAL, 1=FAKE.
    raw["label"] = raw["welfake_label"]

    print("Đang làm sạch và ghép title + text ...")
    raw["content"] = [build_input(t, b) for t, b in zip(raw["title"], raw["text"])]
    raw["body_words"] = raw["text"].str.split().str.len().fillna(0).astype(int)

    too_short = raw["body_words"] < min_words
    stats["removed_short"] = int(too_short.sum())
    raw = raw.loc[~too_short].copy()

    empty = raw["content"].eq("")
    stats["removed_empty_after_cleaning"] = int(empty.sum())
    raw = raw.loc[~empty].copy()

    # Đồng nhất với prepare_data.py: nhận diện trùng bằng 300 ký tự đầu đã làm sạch.
    raw["key"] = raw["content"].str[:300]
    label_counts = raw.groupby("key")["label"].nunique()
    conflict_keys = label_counts[label_counts > 1].index
    conflict = raw["key"].isin(conflict_keys)
    stats["removed_internal_label_conflicts"] = int(conflict.sum())
    raw = raw.loc[~conflict].copy()

    before = len(raw)
    raw = raw.drop_duplicates("key").copy()
    stats["removed_internal_duplicates"] = int(before - len(raw))
    stats["usable_before_isot_dedup"] = int(len(raw))
    return raw, stats


def load_isot_keys() -> set[str]:
    keys: set[str] = set()
    for name in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{name}.csv"
        if not path.exists():
            raise SystemExit(
                f"Không thấy {path}. Hãy chạy python src/prepare_data.py trước, "
                "hoặc dùng --keep-isot-overlap."
            )
        part = pd.read_csv(path, usecols=["text"])
        text = part["text"].fillna("").astype(str)
        keys.update(text.str[:300].tolist())
    return keys


def predict_in_batches(vec, model, texts: pd.Series, batch_size: int) -> np.ndarray:
    predictions = []
    total = len(texts)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        x = vec.transform(texts.iloc[start:stop])
        predictions.append(model.predict(x))
        print(f"    {stop:,}/{total:,}", end="\r", flush=True)
    print(" " * 40, end="\r")
    return np.concatenate(predictions).astype(int)


def main() -> None:
    args = parse_args()
    if args.min_words < 0:
        raise SystemExit("--min-words phải >= 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size phải > 0")
    if args.sample is not None and args.sample <= 0:
        raise SystemExit("--sample phải > 0")

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, stats = load_welfake(data_path, args.min_words)

    if not args.keep_isot_overlap:
        print("Đang loại các bài trùng với ISOT ...")
        isot_keys = load_isot_keys()
        overlap = df["key"].isin(isot_keys)
        stats["removed_isot_overlap"] = int(overlap.sum())
        df = df.loc[~overlap].copy()
    else:
        stats["removed_isot_overlap"] = 0

    if args.sample is not None and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.seed).copy()
        stats["sampled_rows"] = int(args.sample)

    stats["evaluated_rows"] = int(len(df))
    stats["evaluated_real"] = int((df["label"] == 0).sum())
    stats["evaluated_fake"] = int((df["label"] == 1).sum())

    if df.empty:
        raise SystemExit("Không còn dữ liệu để đánh giá sau khi lọc.")
    if df["label"].nunique() < 2:
        raise SystemExit("Dữ liệu đánh giá chỉ còn một lớp nhãn.")

    print(
        f"Dữ liệu đánh giá: {len(df):,} bài "
        f"(REAL {stats['evaluated_real']:,} / FAKE {stats['evaluated_fake']:,})"
    )

    tfidf_path = MODELS_DIR / "baseline" / "tfidf.joblib"
    if not tfidf_path.exists():
        raise SystemExit(f"Không thấy {tfidf_path}. Hãy train baseline trước.")
    vec = joblib.load(tfidf_path)

    all_results: dict = {"dataset": "WELFake", "filter_stats": stats, "models": {}}
    y_true = df["label"].to_numpy(dtype=int)

    for key in args.models:
        model_path = MODELS_DIR / "baseline" / f"{key}.joblib"
        if not model_path.exists():
            print(f"Bỏ qua {key}: không thấy {model_path}")
            continue

        print(f"\n== {key} ==")
        model = joblib.load(model_path)
        started = time.perf_counter()
        y_pred = predict_in_batches(vec, model, df["content"], args.batch_size)
        elapsed = time.perf_counter() - started

        metrics = compute_metrics(y_true, y_pred)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
        report = classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            output_dict=True,
            zero_division=0,
        )
        all_results["models"][key] = {
            "metrics": metrics,
            "confusion_matrix": matrix,
            "classification_report": report,
            "evaluation_seconds": round(elapsed, 3),
        }

        print(
            f"  accuracy={metrics['accuracy']:.4f}  "
            f"precision={metrics['precision']:.4f}  "
            f"recall={metrics['recall']:.4f}  f1={metrics['f1']:.4f}"
        )
        print("  Confusion matrix [[REAL->REAL, REAL->FAKE], [FAKE->REAL, FAKE->FAKE]]")
        print(f"  {matrix}")

        errors = df.loc[y_pred != y_true, ["title", "text", "label", "welfake_label"]].copy()
        errors.insert(3, "prediction", y_pred[y_pred != y_true])
        errors["true_name"] = errors["label"].map({0: "REAL", 1: "FAKE"})
        errors["predicted_name"] = errors["prediction"].map({0: "REAL", 1: "FAKE"})
        error_path = output_dir / f"{key}_errors.csv"
        errors.to_csv(error_path, index=False, encoding="utf-8-sig")
        print(f"  Sai {len(errors):,} bài; đã lưu {error_path}")

    if not all_results["models"]:
        raise SystemExit("Không có model nào được đánh giá.")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nĐã lưu tổng hợp: {metrics_path}")


if __name__ == "__main__":
    main()
