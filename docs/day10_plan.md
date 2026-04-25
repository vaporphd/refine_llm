# День 10. План — Micro-model first

## Контекст и подключение к предыдущим дням

День 8 показал routing: `gpt-4o-mini → gpt-4o`. Cost ratio ×17, что не настолько драматично. День 10 идёт **на порядок дешевле**: первый tier — не LLM вообще, а embedding-based классификатор.

Cost gap получается **×100+**:
- text-embedding-3-small: $0.020 / 1M tokens (~$0.00001 на запрос)
- gpt-4o-mini: $0.150 / 1M input + $0.600 / 1M output (~$0.001 на запрос)

При том что у нас уже есть **160 размеченных train-примеров** из дня 6 — идеальная база для обучения классификатора.

## Задача

Та же что в днях 6-8: классификация запроса разработчика на 4 класса (search/understand/describe/modify). 50 примеров (40 holdout + 10 adversarial).

Это упрощённая версия Дня 9 — берём только `primary_intent`, без других полей. Так классификатор будет на эту единственную ось.

## Архитектура

```
input
  │
  ▼
[Micro: text-embedding-3-small + LogReg]    ← ~50ms, ~$0.00001
  │ predict_proba
  │
  ├─ max_proba ≥ threshold  → STAY  (return micro answer)
  └─ max_proba < threshold  → ESCALATE
                                  │
                                  ▼
                         [LLM: gpt-4o-mini, JSON]    ← ~1000ms, ~$0.001
                                  │
                                  └─ return LLM answer
```

## Что строим

```
scripts/
  micro_train.py        # эмбеддинги train + обучение LogReg, сохранить через joblib
  micro_router.py       # инференс: embed → predict → если low conf → gpt-4o-mini
  micro_compare.py      # сравнить с cheap-only / strong-only / micro-router
data/hard/
  embeddings_train.npz  # сохранённые эмбеддинги (для reproducibility)
results/
  micro_classifier.joblib
  micro_router.json
docs/
  day10_findings.md
```

## Уровень 1: Embedding + Classifier

### Embeddings
- Модель: `text-embedding-3-small` (1536 dim, дешевле и часто лучше для коротких текстов чем `large` 3072)
- Один вызов на текст, ~$0.00001 на короткий запрос (50 токенов)

### Classifier
- **Logistic Regression** через scikit-learn (рекомендуется)
  - Быстрый train, calibrated probabilities из коробки
  - На 160 примерах + 1536-dim фичах работает мгновенно
- Альтернативы: KNN (k=5), Cosine centroid, MLP

### Confidence
- `predict_proba()` возвращает массив [p1, p2, p3, p4]
- Confidence = `max(predict_proba)`
- Если confidence ≥ THRESHOLD → принять
- Иначе → escalate

### Threshold
Стартовый порог: 0.6 (LogReg выдаёт более калиброванные вероятности чем LLM-self-confidence). Подберём на калибровочной выборке если будет время.

## Уровень 2: LLM fallback

`gpt-4o-mini` с JSON-схемой `{label, confidence}` (как в Дне 7-8).

## Прогоны для сравнения

| Прогон | Описание | Calls |
|---|---|---|
| **A. cheap-only** | Только gpt-4o-mini (есть из Дня 8 — `router_cheap.json`) | 50 LLM |
| **B. strong-only** | Только gpt-4o (есть из Дня 8 — `router_strong.json`) | 50 LLM |
| **C. router (8)** | gpt-4o-mini → gpt-4o (есть из Дня 8) | 50–100 LLM |
| **D. micro-router (NEW)** | LogReg → gpt-4o-mini fallback | 50 embeds + N LLM |
| **E. micro+llm-router** | LogReg → gpt-4o-mini → gpt-4o (опционально) | 50 embeds + N1 + N2 LLM |

Главный сравнительный прогон — D vs A. Cost saving может быть 10-50× при сопоставимой accuracy.

## Метрики

### Распределение нагрузки
- % запросов обслужено micro-моделью
- % эскалировано
- В каких классах чаще эскалация

### Качество
- Accuracy финал
- Accuracy на тех что micro обработал самостоятельно
- Сколько правильных предсказаний micro "сдал" в LLM зря (false escalations)
- Сколько неправильных micro "потерял" из-за слишком высокого confidence

### Cost & latency
- Total cost (embeddings + LLM calls)
- Saving ratio vs cheap-only, vs strong-only
- p50/p95 latency для STAY vs ESCALATE отдельно

## Этапы (~2 часа)

| Шаг | Время | Что |
|---|---|---|
| 1 | 0.2 | AskUserQuestion (тип micro, classifier, threshold) |
| 2 | 0.4 | `micro_train.py` — embed train + fit LogReg |
| 3 | 0.5 | `micro_router.py` — inference pipeline |
| 4 | 0.3 | Прогон micro-router на 50 (~50 embeds + ~10-20 LLM calls) |
| 5 | 0.3 | `micro_compare.py` — сравнение с прошлыми прогонами |
| 6 | 0.4 | `day10_findings.md` |
| 7 | 0.2 | Commit + push |

API cost: ~$0.01 (160 train embeds + 50 holdout embeds + ~10-20 fallback LLM calls). Дёшево.

## Решения (зафиксировано)

1. **Тип micro**: `text-embedding-3-small` + `sklearn.LogisticRegression`
2. **Classifier**: LogisticRegression (calibrated probabilities)
3. **Threshold**: 0.6 (max_proba >= 0.6 → принять; иначе → escalate to gpt-4o-mini)

## Финальный конвейер

```
input
  │
  ▼
[Embed: text-embedding-3-small]              ← ~50ms, ~$0.00001
  │
  ▼
[LogReg: predict + predict_proba]            ← ~1ms (CPU)
  │
  ├─ max_proba ≥ 0.6 AND label valid  → STAY     (return micro answer)
  └─ max_proba < 0.6                  → ESCALATE
                                            │
                                            ▼
                                  [gpt-4o-mini, JSON {label, confidence}]    ← ~1000ms, ~$0.001
                                            │
                                            └─ return LLM answer
```

## Reproducibility

- Seed для LogReg: 42
- Все эмбеддинги train сохраняются в `data/hard/embeddings_train.npz` для воспроизводимости
- Модель сохраняется через `joblib.dump` в `results/micro_classifier.joblib`
- При повторном запуске micro_router.py — загрузка из артефактов без повторных embed-вызовов на train
