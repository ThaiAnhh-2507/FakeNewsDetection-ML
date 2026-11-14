"""Fine-tune DistilBERT cho bài toán phát hiện tin giả.

Chạy từ thư mục gốc dự án:
    python src/train_bert.py
    python src/train_bert.py --epochs 3 --batch-size 8

Cần: torch, transformers. Quy ước nhãn: 0=REAL, 1=FAKE.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import MODELS_DIR, compute_metrics, confusion, load_split, print_metrics, save_metrics

OUT_DIR = MODELS_DIR / "bert"


class BertNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts.tolist()
        self.labels = labels.astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs, labels=labels)
            total_loss += outputs.loss.item() * labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.logits.argmax(dim=1).cpu().numpy())
    return total_loss / max(len(loader.dataset), 1), np.asarray(y_true), np.asarray(y_pred)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT")
    ap.add_argument("--model-name", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0 or args.max_len <= 0:
        raise SystemExit("epochs, batch-size và max-len phải > 0")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise SystemExit("Thiếu transformers. Cài bằng: pip install transformers") from exc

    set_seed(args.seed)
    train, val, test = (load_split(n) for n in ("train", "val", "test"))
    print(f"Train {len(train):,} | Val {len(val):,} | Test {len(test):,}")
    print(f"Đang nạp {args.model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "REAL", 1: "FAKE"},
        label2id={"REAL": 0, "FAKE": 1},
    )

    train_ds = BertNewsDataset(train["text"], train["label"], tokenizer, args.max_len)
    val_ds = BertNewsDataset(val["text"], val["label"], tokenizer, args.max_len)
    test_ds = BertNewsDataset(test["text"], test["label"], tokenizer, args.max_len)
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
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_dir = OUT_DIR / "_best"
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            labels = batch.pop("labels").to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running += outputs.loss.item() * labels.size(0)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}")

        train_loss = running / len(train_ds)
        val_loss, val_y, val_pred = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_y, val_pred)
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        print_metrics("val", val_metrics)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            stale = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print("  Đã lưu checkpoint tốt nhất")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Dừng sớm sau {epoch} epoch")
                break

    model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(best_dir)
    # Dataset giữ tokenizer cũ cùng loại; model/tokenizer tốt nhất được lưu ra thư mục chính.
    _, val_y, val_pred = evaluate(model, val_loader, device)
    _, test_y, test_pred = evaluate(model, test_loader, device)
    results = {
        "train_seconds": round(time.time() - started, 1),
        "best_epoch": best_epoch,
        "device": str(device),
        "base_model": args.model_name,
        "val": compute_metrics(val_y, val_pred),
        "test": compute_metrics(test_y, test_pred),
        "confusion_test": confusion(test_y, test_pred),
        "config": vars(args),
    }

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    (OUT_DIR / "app_config.json").write_text(
        json.dumps({"max_len": args.max_len, "labels": {"0": "REAL", "1": "FAKE"}}, indent=2),
        encoding="utf-8",
    )
    print("\nKết quả checkpoint tốt nhất:")
    print_metrics("val", results["val"])
    print_metrics("test", results["test"])
    print(f"Confusion test: {results['confusion_test']}")
    save_metrics("bert", results)
    print(f"Đã lưu model vào {OUT_DIR}")


if __name__ == "__main__":
    main()
