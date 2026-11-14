"""Huấn luyện BiLSTM cho bài toán phát hiện tin giả.

Chạy từ thư mục gốc dự án:
    python src/train_lstm.py
    python src/train_lstm.py --epochs 8 --batch-size 64

Quy ước nhãn: 0=REAL, 1=FAKE.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from common import MODELS_DIR, compute_metrics, confusion, load_split, print_metrics, save_metrics
from lstm_model import LSTMClassifier, build_vocab, encode

OUT_DIR = MODELS_DIR / "lstm"


class NewsDataset(Dataset):
    def __init__(self, texts, labels, word2idx: dict, max_len: int):
        self.texts = texts.tolist()
        self.labels = labels.astype(int).tolist()
        self.word2idx = word2idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        x = torch.tensor(encode(self.texts[idx], self.word2idx, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())
    return total_loss / max(len(loader.dataset), 1), np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện BiLSTM")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.max_len <= 0:
        raise SystemExit("epochs, batch-size và max-len phải > 0")

    set_seed(args.seed)
    train, val, test = (load_split(n) for n in ("train", "val", "test"))
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")

    print("Đang tạo vocabulary...")
    vocab = build_vocab(train["text"], max_size=args.vocab_size, min_freq=2)
    print(f"Vocabulary: {len(vocab):,} token")

    train_ds = NewsDataset(train["text"], train["label"], vocab, args.max_len)
    val_ds = NewsDataset(val["text"], val["label"], vocab, args.max_len)
    test_ds = NewsDataset(test["text"], test["label"], vocab, args.max_len)

    pin = torch.cuda.is_available()
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")
    model_args = {
        "vocab_size": len(vocab),
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "dropout": args.dropout,
    }
    model = LSTMClassifier(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = OUT_DIR / "lstm.pt"
    best_f1 = -1.0
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * y.size(0)

        train_loss = running / len(train_ds)
        val_loss, val_y, val_pred = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_y, val_pred)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        print_metrics("val", val_metrics)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            stale = 0
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "model_args": model_args,
                "max_len": args.max_len,
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, best_path)
            print("  Đã lưu checkpoint tốt nhất")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Dừng sớm sau {epoch} epoch")
                break

    (OUT_DIR / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )

    ckpt = torch.load(best_path, map_location=device)
    model = LSTMClassifier(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    _, val_y, val_pred = evaluate(model, val_loader, device)
    _, test_y, test_pred = evaluate(model, test_loader, device)
    results = {
        "train_seconds": round(time.time() - started, 1),
        "best_epoch": int(ckpt["epoch"]),
        "device": str(device),
        "val": compute_metrics(val_y, val_pred),
        "test": compute_metrics(test_y, test_pred),
        "confusion_test": confusion(test_y, test_pred),
        "config": vars(args),
    }
    print("\nKết quả checkpoint tốt nhất:")
    print_metrics("val", results["val"])
    print_metrics("test", results["test"])
    print(f"Confusion test: {results['confusion_test']}")
    save_metrics("lstm", results)
    print(f"Đã lưu model vào {OUT_DIR}")


if __name__ == "__main__":
    main()
