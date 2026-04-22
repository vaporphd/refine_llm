"""Валидация JSONL-датасета для OpenAI fine-tuning.

Проверки:
  1. Каждая строка — валидный JSON.
  2. Есть ключ "messages" — список из 3 элементов.
  3. Роли строго: system, user, assistant — в этом порядке.
  4. Все content — непустые строки (после strip).
  5. assistant.content ∈ ALLOWED_LABELS.
  6. Длина user.content: 5..500 символов.
  7. Нет дублей по user.content (hash).

Выход:
  - Код 0 и "OK: N valid" если всё хорошо.
  - Код 1 и список ошибок построчно если нет.

Usage:
  python scripts/validate.py data/raw.jsonl
  python scripts/validate.py data/train.jsonl data/eval.jsonl
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ALLOWED_LABELS = {"search", "understand", "describe", "modify"}
REQUIRED_ROLES = ("system", "user", "assistant")
MIN_USER_LEN = 5
MAX_USER_LEN = 500


def validate_line(raw: str, line_no: int) -> list[str]:
    errors: list[str] = []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return [f"line {line_no}: invalid JSON: {e}"]

    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return [f"line {line_no}: missing or non-list 'messages'"]
    if len(msgs) != 3:
        errors.append(f"line {line_no}: expected 3 messages, got {len(msgs)}")
        return errors

    for i, (msg, expected_role) in enumerate(zip(msgs, REQUIRED_ROLES)):
        if not isinstance(msg, dict):
            errors.append(f"line {line_no}: message[{i}] is not an object")
            continue
        role = msg.get("role")
        if role != expected_role:
            errors.append(
                f"line {line_no}: message[{i}] role={role!r}, expected {expected_role!r}"
            )
        content = msg.get("content")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"line {line_no}: message[{i}] ({expected_role}) has empty content")

    if errors:
        return errors

    user_content = msgs[1]["content"]
    assistant_content = msgs[2]["content"]

    if assistant_content not in ALLOWED_LABELS:
        errors.append(
            f"line {line_no}: assistant label {assistant_content!r} not in {sorted(ALLOWED_LABELS)}"
        )

    if len(user_content) < MIN_USER_LEN:
        errors.append(
            f"line {line_no}: user content too short ({len(user_content)} < {MIN_USER_LEN})"
        )
    if len(user_content) > MAX_USER_LEN:
        errors.append(
            f"line {line_no}: user content too long ({len(user_content)} > {MAX_USER_LEN})"
        )

    return errors


def validate_file(path: Path) -> tuple[int, list[str]]:
    if not path.exists():
        return 0, [f"file not found: {path}"]

    errors: list[str] = []
    seen_hashes: dict[str, int] = {}
    total = 0

    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            total += 1
            line_errors = validate_line(raw, line_no)
            errors.extend(line_errors)
            if line_errors:
                continue
            obj = json.loads(raw)
            user = obj["messages"][1]["content"].strip().lower()
            h = hashlib.sha1(user.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                errors.append(
                    f"line {line_no}: duplicate user content (first seen at line {seen_hashes[h]})"
                )
            else:
                seen_hashes[h] = line_no

    return total, errors


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/validate.py <file.jsonl> [more.jsonl ...]", file=sys.stderr)
        return 2

    exit_code = 0
    for arg in sys.argv[1:]:
        path = Path(arg)
        total, errors = validate_file(path)
        if errors:
            exit_code = 1
            print(f"\n{path} — FAIL ({len(errors)} error{'s' if len(errors) != 1 else ''}):")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"{path} — OK ({total} examples)")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
