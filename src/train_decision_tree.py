"""Train and evaluate TF-IDF + Decision Tree on the existing ISOT splits."""

from time import perf_counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

from common import MODELS_DIR, compute_metrics, confusion, load_split, print_metrics, save_metrics


def main() -> None:
    train, val = (load_split(name) for name in ("train", "val"))
    print(f"Train {len(train):,} | Val {len(val):,}", flush=True)

    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    print("Fitting TF-IDF on the training split...", flush=True)
    x_train = vectorizer.fit_transform(train["text"]).tocsc()
    x_val = vectorizer.transform(val["text"])

    candidates = []
    for depth in (10, 20, 30):
        print(f"Fitting Decision Tree with max_depth={depth}...", flush=True)
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, random_state=42)
        start = perf_counter()
        tree.fit(x_train, train["label"])
        seconds = perf_counter() - start
        val_metrics = compute_metrics(val["label"], tree.predict(x_val))
        print_metrics(f"val/{depth}", val_metrics)
        candidates.append((depth, tree, seconds, val_metrics))

    # Prefer the shallower tree if validation F1 ties.
    depth, tree, seconds, val_metrics = max(
        candidates, key=lambda candidate: (candidate[3]["f1"], -candidate[0])
    )
    pipeline = make_pipeline(vectorizer, tree)

    test = load_split("test")
    test_pred = pipeline.predict(test["text"])
    results = {
        "train_seconds": round(seconds, 1),
        "val": val_metrics,
        "test": compute_metrics(test["label"], test_pred),
        "confusion_test": confusion(test["label"], test_pred),
        "config": {
            "max_depth": depth,
            "min_samples_leaf": tree.min_samples_leaf,
            "seed": tree.random_state,
        },
        "validation_f1_by_max_depth": {
            str(candidate_depth): metrics["f1"]
            for candidate_depth, _, _, metrics in candidates
        },
    }

    output = MODELS_DIR / "baseline" / "decision_tree.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    save_metrics("decision_tree", results)

    print(f"Selected max_depth={depth} | Test {len(test):,}")
    print_metrics("test", results["test"])
    print(f"Confusion matrix (REAL, FAKE): {results['confusion_test']}")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()
