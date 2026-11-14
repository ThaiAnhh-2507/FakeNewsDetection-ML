"""
Bước 2: huấn luyện 2 model nền: TF-IDF + Logistic Regression / Linear SVC.
Chạy trên CPU, mất khoảng 1-3 phút.

Chạy:  python src/train_baseline.py
"""
from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from common import MODELS_DIR, compute_metrics, confusion, load_split, print_metrics, save_metrics

OUT_DIR = MODELS_DIR / "baseline"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-features", type=int, default=50000)
    args = ap.parse_args()

    train, val, test = (load_split(n) for n in ("train", "val", "test"))
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    # Vectorizer chỉ học (fit) trên tập train
    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2),
                          min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform(train["text"])
    x_val, x_test = vec.transform(val["text"]), vec.transform(test["text"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, OUT_DIR / "tfidf.joblib")

    models = {
        "logistic": LogisticRegression(C=1.0, max_iter=1000),
        "svc": LinearSVC(C=1.0, dual="auto"),
    }
    for key, clf in models.items():
        print(f"\n== {key} ==")
        t0 = time.time()
        clf.fit(x_train, train["label"])
        secs = round(time.time() - t0, 1)
        val_pred, test_pred = clf.predict(x_val), clf.predict(x_test)
        results = {
            "train_seconds": secs,
            "val": compute_metrics(val["label"], val_pred),
            "test": compute_metrics(test["label"], test_pred),
            "confusion_test": confusion(test["label"], test_pred),
        }
        print(f"  thời gian train: {secs}s")
        print_metrics("val", results["val"])
        print_metrics("test", results["test"])
        joblib.dump(clf, OUT_DIR / f"{key}.joblib")
        save_metrics(key, results)

    # In các từ model coi là quan trọng nhất: dùng để KIỂM TRA RÒ RỈ.
    # Nếu thấy 'reuters', 'video', 'getty'... ở đây thì dữ liệu vẫn còn lối tắt.
    names = vec.get_feature_names_out()
    coef = models["logistic"].coef_[0]
    order = np.argsort(coef)
    print("\nTop 15 từ/cụm báo hiệu FAKE :", ", ".join(names[order[-15:][::-1]]))
    print("Top 15 từ/cụm báo hiệu REAL :", ", ".join(names[order[:15]]))
    print(f"\nĐã lưu model vào {OUT_DIR}")


if __name__ == "__main__":
    main()
