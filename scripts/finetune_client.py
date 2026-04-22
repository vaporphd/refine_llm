"""Клиент запуска fine-tuning через OpenAI API.

Шаги:
  1. Upload train.jsonl и eval.jsonl (purpose=fine-tune)
  2. Create fine-tuning job
  3. Poll status, печатать events по ходу
  4. Сохранить метаданные job в results/finetune_<id>.json

Режимы:
  --dry-run (по умолчанию): НИЧЕГО не загружает, только печатает что сделал бы
  --go: реальный запуск (требует OPENAI_API_KEY)

Usage:
  python scripts/finetune_client.py                  # dry-run
  python scripts/finetune_client.py --go             # реальный запуск
  python scripts/finetune_client.py --go --epochs 3  # с параметрами
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
BASE_MODEL = "gpt-4o-mini-2024-07-18"
POLL_INTERVAL_SEC = 30
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


def ensure_files_exist() -> tuple[Path, Path]:
    train = DATA / "train.jsonl"
    holdout = DATA / "eval.jsonl"
    for p in (train, holdout):
        if not p.exists():
            print(f"error: {p} not found — сначала запусти split.py", file=sys.stderr)
            sys.exit(2)
    return train, holdout


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
        print("error: OPENAI_API_KEY not set (see .env.example)", file=sys.stderr)
        sys.exit(2)
    return OpenAI()


def upload_file(client, path: Path):
    print(f"[upload] {path.name} ...", flush=True)
    with path.open("rb") as f:
        resp = client.files.create(file=f, purpose="fine-tune")
    print(f"[upload] {path.name} -> id={resp.id} bytes={resp.bytes}", flush=True)
    return resp


def create_job(client, train_id: str, holdout_id: str, epochs: int | str):
    hp = {"n_epochs": epochs} if epochs != "auto" else {}
    print(f"[job] create on base={BASE_MODEL} epochs={epochs}", flush=True)
    job = client.fine_tuning.jobs.create(
        training_file=train_id,
        validation_file=holdout_id,
        model=BASE_MODEL,
        hyperparameters=hp,
    )
    print(f"[job] id={job.id} status={job.status}", flush=True)
    return job


def poll(client, job_id: str, out_path: Path) -> dict:
    seen_event_ids: set[str] = set()
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        events = client.fine_tuning.jobs.list_events(job_id, limit=20)
        new_events = []
        for e in reversed(list(events.data)):
            if e.id not in seen_event_ids:
                seen_event_ids.add(e.id)
                new_events.append(e)
        for e in new_events:
            print(f"  [event] {e.created_at} {e.level:<5} {e.message}", flush=True)

        print(f"[poll] status={job.status} trained_tokens={job.trained_tokens}", flush=True)

        if job.status in TERMINAL_STATES:
            result = {
                "job_id": job.id,
                "status": job.status,
                "model": job.model,
                "fine_tuned_model": job.fine_tuned_model,
                "trained_tokens": job.trained_tokens,
                "training_file": job.training_file,
                "validation_file": job.validation_file,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "error": job.error.model_dump() if job.error else None,
            }
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n[done] saved job metadata to {out_path}", flush=True)
            return result
        time.sleep(POLL_INTERVAL_SEC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--go", action="store_true", help="реальный запуск (default: dry-run)")
    parser.add_argument("--epochs", default="auto", help="n_epochs или 'auto'")
    args = parser.parse_args()

    train, holdout = ensure_files_exist()
    train_size = train.stat().st_size
    holdout_size = holdout.stat().st_size
    train_lines = sum(1 for _ in train.open(encoding="utf-8"))
    holdout_lines = sum(1 for _ in holdout.open(encoding="utf-8"))

    print("=" * 60)
    print(f"Mode:            {'GO (real)' if args.go else 'DRY RUN'}")
    print(f"Base model:      {BASE_MODEL}")
    print(f"Train file:      {train} ({train_lines} lines, {train_size} bytes)")
    print(f"Validation file: {holdout} ({holdout_lines} lines, {holdout_size} bytes)")
    print(f"Epochs:          {args.epochs}")
    print("=" * 60)

    if not args.go:
        print("\n[dry-run] Ничего не загружено, задание не создано.")
        print("[dry-run] Запусти с --go для реального старта.")
        return

    try:
        epochs = int(args.epochs) if args.epochs != "auto" else "auto"
    except ValueError:
        print(f"error: invalid --epochs value {args.epochs!r}", file=sys.stderr)
        sys.exit(2)

    client = get_client()
    train_file = upload_file(client, train)
    holdout_file = upload_file(client, holdout)
    job = create_job(client, train_file.id, holdout_file.id, epochs)

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / f"finetune_{job.id}.json"
    poll(client, job.id, out_path)


if __name__ == "__main__":
    main()
