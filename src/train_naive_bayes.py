"""Train TF-IDF + Multinomial Naive Bayes on the existing ISOT splits."""

from time import perf_counter

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from common import (
    MODELS_DIR,
    compute_metrics,
    confusion,
    load_split,
    print_metrics,
    save_metrics,
)


def main() -> None:
    train, val, test = (load_split(name) for name in ("train", "val", "test"))
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    vectorizer = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )
    x_train = vectorizer.fit_transform(train["text"])

    model = MultinomialNB(alpha=1.0)
    start = perf_counter()
    model.fit(x_train, train["label"])
    train_seconds = round(perf_counter() - start, 1)

    pipeline = make_pipeline(vectorizer, model)
    val_pred = pipeline.predict(val["text"])
    test_pred = pipeline.predict(test["text"])
    results = {
        "train_seconds": train_seconds,
        "val": compute_metrics(val["label"], val_pred),
        "test": compute_metrics(test["label"], test_pred),
        "confusion_test": confusion(test["label"], test_pred),
    }

    output = MODELS_DIR / "baseline" / "multinomial_nb.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output)
    save_metrics("multinomial_nb", results)

    print_metrics("val", results["val"])
    print_metrics("test", results["test"])
    print(f"Confusion matrix (REAL, FAKE): {results['confusion_test']}")
    print(f"Saved model: {output}")


if __name__ == "__main__":
    main()