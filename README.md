# refine_llm

**Fine-tuning pipeline for a 4-class classifier of developer queries to AI coding assistants.**

A reproducible workflow for labeling, fine-tuning, and evaluating a small classifier that routes developer queries to a code-aware LLM into one of four intent classes:

| Class | What the developer is asking for |
|---|---|
| `search` | Find / locate code (file, function, usage) |
| `understand` | Explain a specific piece of code or function |
| `describe` | High-level overview of a module, architecture, or project |
| `modify` | Change code — refactor, generate, fix |

The goal is a public, labeled taxonomy plus reusable artifacts (datasets, validators, baseline metrics, OpenAI fine-tuning client) that other coding-agent projects can plug into for intent routing, eval, and ablation studies.

## Repo layout

```
data/            JSONL datasets (raw, train, eval)
scripts/         validate.py, baseline.py, finetune_client.py
results/         baseline measurements, evaluation criteria
tasks/           todo.md, lessons.md
docs/            design notes
```

## Quick start

```bash
cp .env.example .env       # add OPENAI_API_KEY
pip install openai python-dotenv
python scripts/validate.py data/raw.jsonl
python scripts/baseline.py
python scripts/finetune_client.py --dry-run
```

## Data sources

- **Real (~20%)**: prompts collected from local `~/.claude/projects/` and `thoughts/` folders.
- **Synthetic (~80%)**: generated via Claude / GPT across multiple phrasing styles to cover the distribution.

## Models

- Baseline: `gpt-4o-mini` (zero-shot, no fine-tuning).
- Target: `ft:gpt-4o-mini:*` (after a fine-tuning run).

## License

MIT — see [LICENSE](LICENSE).

---

# refine_llm — Fine-tuning классификатора запросов разработчика (RU)

## Задача
Классификация запросов разработчика к AI-ассистенту по кодовой базе по 4 классам:

| Класс | Описание |
|---|---|
| `search` | Найти/локализовать код (файл, функцию, использование) |
| `understand` | Объяснить конкретный код или функцию |
| `describe` | Высокоуровневое описание модуля, архитектуры, проекта |
| `modify` | Изменить код: рефакторинг, генерация, фикс |

## Структура
```
data/            # JSONL-датасеты (raw, train, eval)
scripts/         # validate.py, baseline.py, finetune_client.py
results/         # baseline-замеры, критерии оценки
tasks/           # todo.md, lessons.md
```

## Запуск
```bash
cp .env.example .env   # вставить OPENAI_API_KEY
pip install openai python-dotenv
python scripts/validate.py data/raw.jsonl
python scripts/baseline.py
python scripts/finetune_client.py --dry-run
```

## Источник данных
- **Реальные (20%+)**: промпты из `~/.claude/projects/` и `thoughts/` папок
- **Синтетика (80%)**: генерация через Claude/GPT с разными стилями

## Модель
- Baseline: `gpt-4o-mini` (без ФТ)
- Target: `ft:gpt-4o-mini:*` (после запуска)
