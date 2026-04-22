# refine_llm — Fine-tuning классификатора запросов разработчика

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
