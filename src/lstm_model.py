"""Kiến trúc BiLSTM + hàm tạo từ điển / mã hóa văn bản (dùng chung train & dự đoán)."""
from __future__ import annotations

from collections import Counter

import torch
from torch import nn

PAD, UNK = 0, 1


def build_vocab(texts, max_size: int = 20000, min_freq: int = 2) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    word2idx = {"<PAD>": PAD, "<UNK>": UNK}
    for word, freq in counter.most_common(max_size - 2):
        if freq < min_freq:
            break
        word2idx[word] = len(word2idx)
    return word2idx


def encode(text: str, word2idx: dict, max_len: int) -> list:
    ids = [word2idx.get(w, UNK) for w in text.split()[:max_len]]
    return ids + [PAD] * (max_len - len(ids))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, 2)

    def forward(self, x):
        mask = (x != PAD).unsqueeze(-1).float()            # (B, T, 1)
        out, _ = self.lstm(self.drop(self.emb(x)))         # (B, T, 2H)
        # Trung bình có mask: không để các token PAD làm nhiễu kết quả
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.fc(self.drop(pooled))
