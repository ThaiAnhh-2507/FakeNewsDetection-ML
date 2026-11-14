"""Nạp các model đã train và dự đoán. Chỉ model nào đã train xong mới được liệt kê."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from common import MODELS_DIR
from text_utils import build_input

MODEL_INFO = {
    "logistic": "Logistic Regression (TF-IDF)",
    "svc": "Linear SVC (TF-IDF)",
    "lstm": "BiLSTM",
    "bert": "DistilBERT",
}


class Predictor:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.dir = Path(models_dir)
        self._cache: dict = {}

    def _required_files(self, key: str) -> list:
        d = self.dir
        return {
            "logistic": [d / "baseline" / "tfidf.joblib", d / "baseline" / "logistic.joblib"],
            "svc": [d / "baseline" / "tfidf.joblib", d / "baseline" / "svc.joblib"],
            "lstm": [d / "lstm" / "lstm.pt", d / "lstm" / "vocab.json"],
            "bert": [d / "bert" / "config.json", d / "bert" / "app_config.json"],
        }[key]

    def available(self) -> list:
        return [k for k in MODEL_INFO if all(p.exists() for p in self._required_files(k))]

    def _load(self, key: str):
        if key in self._cache:
            return self._cache[key]
        if key in ("logistic", "svc"):
            import joblib
            if "_tfidf" not in self._cache:
                self._cache["_tfidf"] = joblib.load(self.dir / "baseline" / "tfidf.joblib")
            obj = (self._cache["_tfidf"], joblib.load(self.dir / "baseline" / f"{key}.joblib"))
        elif key == "lstm":
            import torch
            from lstm_model import LSTMClassifier
            ckpt = torch.load(self.dir / "lstm" / "lstm.pt", map_location="cpu")
            vocab = json.loads((self.dir / "lstm" / "vocab.json").read_text(encoding="utf-8"))
            model = LSTMClassifier(**ckpt["model_args"])
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            obj = (model, vocab, int(ckpt["max_len"]))
        elif key == "bert":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            path = self.dir / "bert"
            tok = AutoTokenizer.from_pretrained(str(path))
            model = AutoModelForSequenceClassification.from_pretrained(str(path))
            model.eval()
            cfg = json.loads((path / "app_config.json").read_text(encoding="utf-8"))
            obj = (tok, model, int(cfg.get("max_len", 256)))
        else:
            raise KeyError(key)
        self._cache[key] = obj
        return obj

    def predict(self, key: str, title: str = "", body: str = "") -> dict:
        text = build_input(title, body)
        if not text:
            raise ValueError("Văn bản trống sau khi làm sạch.")
        obj = self._load(key)  # thời gian nạp model không tính vào ms
        t0 = time.perf_counter()

        if key == "logistic":
            vec, clf = obj
            p_fake = float(clf.predict_proba(vec.transform([text]))[0][1])
        elif key == "svc":
            vec, clf = obj
            score = float(clf.decision_function(vec.transform([text]))[0])
            p_fake = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, score))))  # ước lượng, không hiệu chuẩn
        elif key == "lstm":
            import torch
            from lstm_model import encode
            model, vocab, max_len = obj
            x = torch.tensor([encode(text, vocab, max_len)], dtype=torch.long)
            with torch.no_grad():
                p_fake = float(torch.softmax(model(x), dim=-1)[0, 1])
        else:  # bert
            import torch
            tok, model, max_len = obj
            inputs = tok(text, truncation=True, max_length=max_len, return_tensors="pt")
            with torch.no_grad():
                p_fake = float(torch.softmax(model(**inputs).logits, dim=-1)[0, 1])

        ms = (time.perf_counter() - t0) * 1000
        p_fake = min(max(p_fake, 0.0), 1.0)
        return {
            "model": key,
            "name": MODEL_INFO[key],
            "label": "FAKE" if p_fake >= 0.5 else "REAL",
            "fake_prob": p_fake,
            "real_prob": 1.0 - p_fake,
            "confidence": max(p_fake, 1.0 - p_fake),
            "ms": ms,
            "words": len(text.split()),
            "clean_text": text,
        }
