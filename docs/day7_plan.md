# День 7. План выполнения

## Контекст
Используем существующий пайплайн классификатора (4 класса: search/understand/describe/modify) и датасет `data/hard/` (40 eval, baseline 85%, 6 ошибок). Это идеальная задача для оценки уверенности — есть знаемая truth-разметка, есть известные слабые места baseline (describe↔understand), на которых можно проверить, поймает ли механизм уверенности именно эти ошибки.

## Стратегия: 3 подхода вместо 2 (для сравнения)

### Подход A: Constraint-based (самый дешёвый — фильтр, не отдельный вызов)
- Уже частично есть: regex проверка формата ответа в `baseline.py`
- Расширить: добавить проверку что ответ ∈ {search, understand, describe, modify}, normalized (lowercase, no punctuation)
- Стоимость: 0 (post-processing после inference)
- Действует как первая линия — отсеивает мусорные ответы

### Подход B: Redundancy / Self-consistency
- Прогнать тот же промпт **3 раза** с `temperature=0.7` (нужна вариативность)
- Majority vote: если 3/3 одинаковые → OK, 2/3 → UNSURE, all different → FAIL
- Стоимость: ×3 cost, ×3 latency (можно параллельно — latency почти такой же)
- Это самый "честный" способ оценки уверенности — не доверяет модели саморефлексии

### Подход C: Scoring (модель сама возвращает confidence)
- Промпт просит ответ в формате `{"label": "search", "confidence": 0.0-1.0}`
- Проверяем калибровку: реально ли confidence коррелирует с accuracy
- Стоимость: ×1.1 (чуть больше токенов, тот же вызов)
- Это самый дешёвый способ, но известно, что LLM плохо калиброваны

### (Self-check намеренно НЕ делаем)
Self-check (модель проверяет свой ответ) дорогой и часто даёт ложные подтверждения — модель склонна соглашаться сама с собой. Если будет время в конце — добавим как 4-й подход для полноты.

## Конвейер контроля качества (gating)

Финальный prediction проходит цепочку:
```
input → [Constraint check] → [Confidence ≥ threshold?] → [Self-consistency 3×] → output
                                       ↓ no                       ↓ disagree
                                   UNSURE                       FAIL
```

3 статуса вывода:
- **OK**: формат ✓, confidence ≥ 0.85, 3/3 vote одинаковые → принимаем ответ
- **UNSURE**: формат ✓, но confidence < 0.85 ИЛИ 2/3 vote → нужен human review
- **FAIL**: формат ✗ ИЛИ 0-1 голос за топ-класс → отклонить, не отдавать пользователю

## Что строим (артефакты)

```
scripts/
  confidence_check.py     # основной скрипт: A+B+C на 40 eval
  confidence_compare.py   # сравнить с ground truth, посчитать метрики
results/
  confidence_hard.json    # сырые ответы всех 3 проходов + аггрегаты
  confidence_report.md    # выводы: rejection rate, latency, cost, OK/UNSURE/FAIL по truth
docs/
  day7_findings.md        # финальный отчёт
```

## Тест-сеты (3 типа входов)

1. **Корректные** — 40 примеров из `data/hard/eval.jsonl` (баланс 10/10/10/10)
2. **Пограничные** — 6 известных baseline-ошибок (3× describe↔understand, 2× search↔understand, 1× search→modify) — это уже в eval, выделяем подмножество для отдельного анализа
3. **Шумные / adversarial** — 10 новых, генерим вручную:
   - очень короткие однословные ("?", "что", "fix")
   - почти-дубли с противоречием ("найди и удали Х")
   - на третьих языках (испанский, немецкий) — модель не должна паниковать
   - длинные с двумя интенциями подряд

## Метрики

### Качество отбора
- **Pickup rate** на корректных: % accepted (status=OK) — должен быть высокий, ~85%+
- **Rejection rate** на adversarial: % rejected (FAIL/UNSURE) — должен быть высокий, ~70%+
- **Catch rate** на пограничных: из 6 известных ошибок, сколько помечено как UNSURE/FAIL? Цель: ≥4 из 6 (механизм должен ловить свои ошибки)

### Стоимость
- **Latency**: p50 / p95 на одном запросе через каждый подход (A, B, C, A+B+C combined)
- **Cost**: токены × цена. Сравнение baseline (1 вызов) vs gated (3-4 вызова при self-consistency)

### Калибровка scoring
- Корреляция Spearman между `confidence` (от модели) и `correct` (truth match) на 40 eval
- Если корреляция < 0.3 → scoring approach не работает на gpt-4o-mini

## Этапы и сроки (в часах)

| Шаг | Время | Что |
|---|---|---|
| 1 | 0.3 | Создать структуру, обновить TODO, добавить docs |
| 2 | 0.5 | Сгенерировать 10 adversarial тест-кейсов |
| 3 | 0.7 | Написать `confidence_check.py` (3 подхода + gating) |
| 4 | 0.5 | Прогнать на 40 eval + 10 adversarial = 50 примеров (~150 API calls с self-consistency) |
| 5 | 0.5 | Написать `confidence_compare.py` — анализ |
| 6 | 0.4 | `day7_findings.md` с выводами |
| 7 | 0.2 | Commit + push |

Итого ~3 часа чистой работы. API cost ~$0.05–0.10 (50 примеров × 4 вызова × дешёвая модель).

## Решения (зафиксировано)

1. **Подходы**: Constraint + Redundancy + Scoring (3 из 4)
2. **Датасет**: `data/hard/eval.jsonl` (40) + 10 новых adversarial = 50 примеров
3. **Threshold**: confidence ≥ 0.85 → OK; < 0.85 → UNSURE (строгий, для критических задач)

## Финальный gating-конвейер

```
input
  │
  ▼
[A. Constraint check]──→ format invalid → FAIL
  │ format ok
  ▼
[B. Self-consistency 3× @ T=0.7] (parallel)
  │
  ├─ 3/3 agree    → check (C)
  ├─ 2/3 agree    → UNSURE (return majority + flag)
  └─ all different → FAIL
  │
  ▼
[C. Scoring: model returns {label, confidence}] (1 call @ T=0)
  │
  ├─ confidence ≥ 0.85  AND label matches majority from B → OK
  ├─ confidence < 0.85  OR  label differs from B          → UNSURE
  └─ confidence < 0.5                                      → FAIL
```

**Почему такой порядок**: cheap-first. Constraint бесплатный, Redundancy тяжёлая (3 вызова), Scoring можно совместить с одним из voting-вызовов чтобы не делать лишнего. Финальная оптимизация: один из 3 redundancy-вызовов делаем через JSON-схему `{label, confidence}` — экономим один полный вызов.

## API-вызовов на 1 пример: 3 (вместо наивных 4)

- 3 вызова T=0.7 для self-consistency, один из них с JSON-схемой возвращает confidence
- Итого 50 примеров × 3 = **150 API вызовов**
- При среднем ~50 input + 20 output токенов на вызов → ~10500 токенов
- Cost gpt-4o-mini: ~$0.005-0.01 на весь эксперимент
