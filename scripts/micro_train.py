"""Тренировка micro-классификатора на embeddings train-набора.

Pipeline:
  1. Загрузить data/hard/train.jsonl (160 примеров с lable)
  2. Embed каждый user-текст через text-embedding-3-small (batch up to 256)
  3. Сохранить эмбеддинги в data/hard/embeddings_train.npy + metadata.json
  4. Fit LogisticRegression на (X=embeddings, y=labels)
  5. Сохранить модель через joblib

Usage:
  python scripts/micro_train.py
  python scripts/micro_train.py --force-reembed   # игнорировать кэш
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBED_MODEL = "text-embedding-3-small"
RANDOM_STATE = 42

EMBED_PRICE_PER_1M = 0.020  # USD


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("error: pip install openai python-dotenv", file=sys.stderr)
        sys.exit(2)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENAI_API_KEY"):
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(2)
    return OpenAI()


def load_train(path: Path) -> tuple[list[str], list[str]]:
    texts, labels = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            texts.append(obj["messages"][1]["content"])
            labels.append(obj["messages"][2]["content"])
    return texts, labels


def embed_batch(client, texts: list[str], batch_size: int = 256) -> tuple[np.ndarray, int]:
    """Возвращает (embeddings array shape [N, dim], total_tokens)."""
    all_embs = []
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for d in resp.data:
            all_embs.append(d.embedding)
        total_tokens += resp.usage.total_tokens
    return np.array(all_embs, dtype=np.float32), total_tokens


def save_embeddings(path: Path, X: np.ndarray, texts: list[str], labels: list[str]) -> None:
    np.save(path.with_suffix(".npy"), X)
    meta = {"texts": texts, "labels": labels, "shape": list(X.shape), "dtype": str(X.dtype)}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str], list[str]] | None:
    npy_path = path.with_suffix(".npy")
    meta_path = path.with_suffix(".meta.json")
    if not (npy_path.exists() and meta_path.exists()):
        return None
    X = np.load(npy_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return X, meta["texts"], meta["labels"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/hard")
    parser.add_argument("--force-reembed", action="store_true")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    cache_base = data_dir / "embeddings_train"
    train_jsonl = data_dir / "train.jsonl"

    texts, labels = load_train(train_jsonl)
    print(f"Loaded {len(texts)} train examples")
    label_set = sorted(set(labels))
    print(f"Classes: {label_set}")
    counts = {c: labels.count(c) for c in label_set}
    print(f"Per-class counts: {counts}")

    cached = None if args.force_reembed else load_embeddings(cache_base)
    X = None
    if cached is not None:
        X_cached, cached_texts, cached_labels = cached
        if cached_texts == texts and cached_labels == labels:
            X = X_cached
            print(f"Loaded cached embeddings from {cache_base}.npy ({X.shape})")
        else:
            print("Cache mismatch — re-embedding")

    if X is None:
        client = get_client()
        print(f"Embedding {len(texts)} texts via {EMBED_MODEL}...")
        t0 = time.perf_counter()
        X, tokens_used = embed_batch(client, texts)
        elapsed = time.perf_counter() - t0
        print(f"  done in {elapsed:.1f}s, tokens={tokens_used}, "
              f"cost ~${tokens_used / 1_000_000 * EMBED_PRICE_PER_1M:.6f}")
        save_embeddings(cache_base, X, texts, labels)
        print(f"  saved to {cache_base}.npy + {cache_base}.meta.json")

    print(f"Embeddings shape: {X.shape}")

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
        C=10.0,  # less regularization → more confident predict_proba
    )
    print("\nFitting LogisticRegression...")
    t0 = time.perf_counter()
    clf.fit(X, labels)
    print(f"  done in {(time.perf_counter() - t0) * 1000:.0f}ms")

    train_acc = clf.score(X, labels)
    print(f"  train accuracy: {train_acc:.0%}")

    out_path = ROOT / "results" / "micro_classifier.joblib"
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump({
        "classifier": clf,
        "embed_model": EMBED_MODEL,
        "classes": clf.classes_.tolist(),
        "n_train": len(texts),
        "train_accuracy": train_acc,
    }, out_path)
    print(f"\nSaved classifier to {out_path}")


if __name__ == "__main__":
    main()
