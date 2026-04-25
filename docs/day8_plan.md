# День 8. План — Routing между моделями

## Контекст и связь с предыдущими днями
День 7 уже даёт нам готовую gating-логику: каждый запрос получает статус **OK / UNSURE / FAIL** через confidence + redundancy + constraint. Routing для Дня 8 — естественное продолжение: то что попало в UNSURE/FAIL у дешёвой модели, эскалируется наверх.

Используем тот же датасет: `data/hard/eval.jsonl` (40) + `data/hard/adversarial.jsonl` (10) = 50 примеров.

## Стратегия: 2-tier router

### Tier 1 (Cheap, быстрый):
- 1 вызов с JSON-схемой `{label, confidence}` (вместо 3 параллельных как в Дне 7)
- Если `confidence ≥ 0.85` И label валидный → **принимаем**, не эскалируем
- Если `confidence < 0.85` ИЛИ format invalid → **эскалация**

### Tier 2 (Strong, дорогой):
- Один вызов с тем же JSON-форматом
- Возвращаем его ответ как финальный (без вторичной эскалации — нет Tier 3)

### Эвристика принятия решения: **confidence threshold**
- Это самая прямая эвристика из 3 вариантов в задании
- Уже валидирована Днём 7 (Spearman +0.5)
- Простая граница: 0.85

### Альтернативные эвристики (опционально):
- **Длина ответа**: для классификации нерелевантна (ответ всегда 1 слово). Применима в open-ended задачах.
- **Self-consistency**: если бюджет позволяет — 2 параллельных cheap-вызова, рассогласие → эскалация. Это дороже но точнее.

## Что измеряем

### Распределение нагрузки
- % запросов обслужено на Tier 1 (cheap)
- % эскалировано на Tier 2 (strong)
- На каких классах/типах входов чаще эскалация (search/understand/describe/modify, holdout vs adversarial)

### Качество
- Accuracy финальных ответов router'а vs:
  - чистый baseline (только cheap, как на Дне 6)
  - чистый strong (только дорогая модель)
  - gating без эскалации (Дня 7 — UNSURE отправляются человеку)
- Чтобы это сравнение было честным — прогоним все 3 варианта на тех же 50 примерах

### Cost & latency
- Tokens cheap vs Tokens strong, итоговая стоимость
- Сравнение с двумя baselines: «всегда cheap» и «всегда strong»
- Latency p50/p95 для эскалированных vs не-эскалированных

## Что строим

```
scripts/
  router.py              # 2-tier с эскалацией по confidence
  router_compare.py      # сводка cheap-only vs strong-only vs router
results/
  router_results.json    # маршруты + финальные ответы + cost/latency
docs/
  day8_findings.md       # отчёт
```

## Какие модели брать (Cheap → Strong)?

OpenAI на данный момент даёт несколько вариантов пар:

| Cheap | Strong | Cost ratio (input) | Когда выбирать |
|---|---|---|---|
| gpt-4o-mini → gpt-4o | ~17× | Классическая пара, уже используем mini |
| gpt-4o-mini → gpt-4.1 | ~10-15× | Если 4.1 доступен и стабильнее |
| gpt-4.1-nano → gpt-4.1 | ~5× | Меньше gap, дешевле эксперимент |
| gpt-4o-mini → o1-mini | varies | Reasoning-модель для эскалации (медленнее) |

Дефолтная рекомендация: **gpt-4o-mini → gpt-4o** — пара уже из стека, известно поведение mini, gap ощутимый.

## Этапы (~2 часа)

| Шаг | Время | Что |
|---|---|---|
| 1 | 0.2 | Решения + AskUserQuestion (модели, эвристика, threshold) |
| 2 | 0.5 | `router.py` — Tier 1 + escalation + Tier 2 |
| 3 | 0.3 | Прогон на 50 примерах (~100 вызовов) |
| 4 | 0.4 | `router_compare.py` — cheap-only vs strong-only vs router |
| 5 | 0.4 | `day8_findings.md` |
| 6 | 0.2 | Commit + push |

API cost ≈ $0.05–0.10 (включая контрольные cheap-only / strong-only прогоны для сравнения).

## Решения (зафиксировано)

1. **Пара моделей**: `gpt-4o-mini` (cheap) → `gpt-4o` (strong)
2. **Эвристика**: чистый confidence < 0.85 от cheap-модели
3. **Контрольные прогоны**: и cheap-only, и strong-only (для сравнения accuracy/cost)

## Конкретный конвейер

```
input
  │
  ▼
[Tier 1: gpt-4o-mini, JSON {label, confidence}]    ← 1 вызов
  │
  ├─ confidence ≥ 0.85 AND label valid    → STAY     (return Tier 1 answer)
  └─ confidence < 0.85 OR format invalid  → ESCALATE
                                                │
                                                ▼
                                  [Tier 2: gpt-4o, JSON {label, confidence}]  ← 1 вызов
                                                │
                                                └─ return Tier 2 answer

Total calls per example: 1 (если уверен) или 2 (если эскалация)
```

## Прогоны для сравнения (на тех же 50 примерах)

| Прогон | Модель | Calls | Назначение |
|---|---|---|---|
| **A. Cheap-only** | только gpt-4o-mini | 50 | Уже есть в `baseline_hard.json` (85%) — можем переиспользовать или прогнать с JSON-форматом для честности |
| **B. Strong-only** | только gpt-4o | 50 | Верхний потолок accuracy |
| **C. Router** | mini → 4o при low conf | 50–100 | Главный артефакт |

## Финальные метрики

- **Распределение нагрузки**: % примеров обработано на cheap, % эскалировано
- **Accuracy финал**: A vs B vs C
- **Cost**: tokens × price для трёх прогонов
- **Latency**: p50/p95 для STAY и ESCALATE отдельно
- **Saving ratio**: cost(C) / cost(B) — насколько дешевле полностью strong при сопоставимом качестве
