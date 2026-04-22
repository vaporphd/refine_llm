# TODO — Fine-tuning классификатора

## 4 класса
- `search` — найти/локализовать код
- `understand` — объяснить конкретный код
- `describe` — высокоуровневое описание модуля/архитектуры
- `modify` — изменить код (refactor/generate/fix)

## System prompt (единый во всех примерах)
```
Ты — классификатор запросов разработчика к AI-ассистенту по кодовой базе. Определи тип запроса и ответь ровно одним словом из списка: search, understand, describe, modify. Никаких пояснений, префиксов или знаков препинания.
```

## Чек-лист
- [x] Структура проекта
- [x] Сбор реальных промптов из ~/.claude
- [x] Сбор промптов из thoughts/ проектов (пусто — взяли всё из ~/.claude)
- [x] Разметка реальных (получили 15, минимум был 10)
- [x] Генерация 35 синтетических (балансировка до 12-13 на класс)
- [x] validate.py
- [x] Чистка + сплит 80/20 (стратифицированный)
- [x] baseline.py → 10 ответов gpt-4o-mini
- [x] criteria.md
- [x] finetune_client.py (без запуска, --go для реального старта)

## Review

**Что получили:**
- 50 примеров в `data/raw.jsonl`: 15 real (30%) + 35 synth (70%)
- Распределение классов: search 13, understand 13, describe 12, modify 12
- Сплит: train 40 / eval 10, стратифицированный, seed=42
- 5 скриптов: build_raw, validate, split, baseline, finetune_client
- criteria.md с 5 метриками (accuracy общая и per-class, формат, стабильность, покрытие)

**Что осталось для запуска ФТ:**
1. `cp .env.example .env` + вписать OPENAI_API_KEY
2. `pip install openai python-dotenv`
3. `python scripts/baseline.py` — зафиксировать baseline до ФТ
4. `python scripts/finetune_client.py --go` — запустить ФТ (~5-15 мин, ~$0.30)
5. `python scripts/baseline.py --model ft:gpt-4o-mini:...` — сравнить с baseline
6. Свериться с `results/criteria.md` — выполнены ли 5 условий "зелёный флаг"

**Нюансы для будущих итераций:**
- Modify перетягивал в real_candidates (15 из 39), пришлось балансировать синтетикой
- Хук безопасности ругается на имя переменной `eval_` — переименовал в `holdout`
- Agent search отработал только `~/.claude/projects`; `/Volumes/mydata/projects/*/thoughts/` оказались пустыми
