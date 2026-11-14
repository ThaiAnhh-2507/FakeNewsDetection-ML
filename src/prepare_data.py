"""
Bước 1: đọc ISOT (data/raw/Fake.csv + True.csv) -> làm sạch -> lọc -> bỏ trùng
-> chia train/val/test -> lưu vào data/processed/.

Chạy:  python src/prepare_data.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import PROCESSED_DIR, RAW_DIR
from text_utils import clean_text


def main() -> None:
    ap = argparse.ArgumentParser(description="Chuẩn bị dữ liệu ISOT")
    ap.add_argument("--raw-dir", default=str(RAW_DIR), help="Thư mục chứa Fake.csv và True.csv")
    ap.add_argument("--min-words", type=int, default=30,
                    help="Bỏ bài có nội dung ít hơn N từ (mặc định 30)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    fake_path, true_path = raw / "Fake.csv", raw / "True.csv"
    for p in (fake_path, true_path):
        if not p.exists():
            raise SystemExit(f"Không thấy {p}\nHãy copy Fake.csv và True.csv vào {raw}")

    keep = lambda c: c in {"title", "text"}  # bỏ subject/date để tránh rò rỉ nhãn
    fake = pd.read_csv(fake_path, usecols=keep)
    true = pd.read_csv(true_path, usecols=keep)
    fake["label"], true["label"] = 1, 0
    df = pd.concat([fake, true], ignore_index=True)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    stats = {"raw_rows": len(df), "raw_fake": int((df.label == 1).sum()),
             "raw_real": int((df.label == 0).sum())}
    print(f"Đọc xong: {stats['raw_rows']:,} bài "
          f"(fake {stats['raw_fake']:,} / real {stats['raw_real']:,})")

    print("Đang làm sạch văn bản (khoảng vài chục giây)...")
    title_clean = df["title"].map(clean_text)
    body_clean = df["text"].map(clean_text)
    df["content"] = (title_clean + " " + body_clean).str.strip()
    df["words"] = body_clean.str.split().str.len().fillna(0).astype(int)

    # 1) Bỏ bài quá ngắn / rỗng (trong ISOT các bài rỗng gần như toàn là tin giả)
    short = df["words"] < args.min_words
    stats["removed_short"] = int(short.sum())
    stats["removed_short_fake"] = int((short & (df.label == 1)).sum())
    df = df[~short].copy()

    # 2) Bỏ bài trùng / gần trùng (so 300 ký tự đầu) TRƯỚC khi chia tập,
    #    nếu không cùng một bài có thể rơi vào cả train lẫn test.
    df["key"] = df["content"].str[:300]
    n_labels = df.groupby("key")["label"].nunique()
    conflict = df["key"].isin(n_labels[n_labels > 1].index)
    stats["removed_conflicting_label"] = int(conflict.sum())
    df = df[~conflict]
    before = len(df)
    df = df.drop_duplicates("key")
    stats["removed_duplicates"] = before - len(df)

    # 3) Chia 70 / 15 / 15, giữ tỉ lệ nhãn
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=args.seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=args.seed)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        out = part[["content", "label"]].rename(columns={"content": "text"})
        out.to_csv(PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8")
        stats[f"{name}_rows"] = len(out)
        stats[f"{name}_fake"] = int((out.label == 1).sum())
    (PROCESSED_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\nKết quả:")
    for k, v in stats.items():
        print(f"  {k:<28} {v:,}")
    print(f"\nĐã lưu vào {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
