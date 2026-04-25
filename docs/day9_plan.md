# День 9. План — Multi-stage inference

## Контекст
Простая классификация на 4 класса (Дни 6-8) **слишком хорошо решается одним запросом** (cheap-only 90%). Чтобы получить осмысленный сравнительный эксперимент monolithic vs multi-stage, расширяем задачу до **multi-field structured extraction**.

## Новая задача: rich intent extraction

Из запроса разработчика к AI-ассистенту извлекаем структуру:

```json
{
  "primary_intent":   "search | understand | describe | modify",
  "secondary_intent": "search | understand | describe | modify | none",
  "target_type":      "file | function | module | project | feature | bug | config | test | other",
  "target_name":      "<extracted identifier or null>",
  "urgency":          "low | normal | high",
  "negation":         true | false
}
```

Пример:
- Вход: `"не трогай parser, только fix logger срочно"`
- Эталон:
  ```json
  {"primary_intent": "modify", "secondary_intent": "none", "target_type": "file",
   "target_name": "logger", "urgency": "high", "negation": true}
  ```

**Почему эта задача сложна для 1 запроса**:
- 6 полей, каждое со своим набором допустимых значений
- Negation требует учёта контекста ("не трогай Х, только Y" — negation true, но действие позитивное)
- Urgency не явная, выводится из тональности ("срочно", "fix that shit", "ASAP")
- target_name требует NER-как умения вытащить идентификатор и не выдумать его
- secondary_intent — гибрид кейс ("explain why and then fix") должен правильно расставить primary/secondary

## Два варианта решения

### A. Monolithic (один большой запрос)
- 1 вызов модели с system prompt описывающим все 6 полей и правила
- Возвращает полный JSON
- 1 call, 1 раз tokens
- Уязвим к: пропускам полей, галлюцинациям target_name, путанице primary/secondary

### B. Multi-stage (4 короткие стадии)

**Stage 1 — Normalize**: вытащить ключевые признаки текста
```
Input:  "не трогай parser, только fix logger срочно"
Output: {"verbs": ["fix"], "negated_targets": ["parser"], "positive_targets": ["logger"], "tone": "urgent"}
```

**Stage 2 — Classify intents**: primary + secondary
```
Input:  normalized + original
Output: {"primary": "modify", "secondary": "none"}
```

**Stage 3 — Extract target**: type + name
```
Input:  normalized + original
Output: {"target_type": "file", "target_name": "logger"}
```

**Stage 4 — Sentiment & negation**:
```
Input:  normalized + original
Output: {"urgency": "high", "negation": true}
```

**Композиция**: финальный JSON собирается локальным кодом из 4 stage outputs.

**Каждая стадия**:
- Короткий system prompt (50-100 токенов вместо 400+ в monolithic)
- JSON-схема с 1-3 полями
- gpt-4o-mini (можем варьировать на gpt-4o-nano если нужно)

## Формат: compact JSON везде

Не TOON и не enum — JSON с `response_format={"type":"json_object"}`. Причины:
- TOON оптимален для иерархии — здесь все стадии плоские
- Enum (одно слово) применим к 1 полю, у нас 6
- JSON универсален и совместим с существующей инфраструктурой

В monolithic варианте JSON-Schema режим (через `json_schema` если поддерживается, иначе free-form JSON).

## Тест-сет

40 holdout из `data/hard/eval.jsonl` + 10 adversarial = 50 примеров. **НО** — для нового задания нужна расширенная разметка (6 полей вместо 1).

**План разметки**:
1. Использовать существующий `primary_intent` из `expected` (уже есть)
2. Я размечаю остальные 5 полей вручную для всех 50 примеров (10-15 минут работы)
3. Результат сохраняется в `data/hard/rich_eval.jsonl`

## Метрики

### Field-level accuracy
Для каждого из 6 полей: какой % правильно угадан в monolithic vs в multi-stage?

### Composite accuracy
Все 6 полей правильны одновременно (exact match).

### Cost & latency
- Monolithic: 1 call, X токенов
- Multi-stage: 4 calls, ~X/2 токенов суммарно (короткие промпты)
- Multi-stage можно запустить **параллельно** stages 2/3/4 (они не зависят друг от друга), stage 1 — sequential первым

### Failure modes
- Monolithic: какие поля чаще всего ошибочны?
- Multi-stage: какие стадии чаще всего ошибаются и приводят к каскаду ошибок?
- Halluciation rate: target_name указано когда в тексте нет? (только в monolithic в теории, multi-stage по идее устойчивее)

## Артефакты

```
scripts/
  multistage.py            # оба режима (--mode monolithic | multistage)
  multistage_label.py      # помощник для разметки rich_eval (опционально)
  multistage_compare.py    # сравнение accuracy / cost / latency
data/hard/
  rich_eval.jsonl          # 50 примеров с 6-полевой разметкой
results/
  multistage_mono.json
  multistage_multi.json
docs/
  day9_findings.md
```

## Этапы (~3 часа)

| Шаг | Время | Что |
|---|---|---|
| 1 | 0.2 | AskUserQuestion (модель, формат, задача) |
| 2 | 0.6 | Разметка 50 примеров по 6 полям → rich_eval.jsonl |
| 3 | 0.7 | `multistage.py` — оба режима + параллелизация stages 2/3/4 |
| 4 | 0.3 | Прогон обоих режимов (50 + 200 calls = 250 API calls) |
| 5 | 0.5 | `multistage_compare.py` — анализ |
| 6 | 0.4 | `day9_findings.md` |
| 7 | 0.2 | Commit + push |

API cost: ~$0.05–0.10.

## Решения (зафиксировано)

1. **Задача**: Rich extraction — 6 полей (`primary_intent`, `secondary_intent`, `target_type`, `target_name`, `urgency`, `negation`)
2. **Стадий в multi-stage**: 4 (Normalize → Classify intents → Extract target → Sentiment+negation)
3. **Модель**: одна — `gpt-4o-mini` на всех стадиях. Это даёт честное сравнение mono vs multi — разница только в архитектуре.

## Финальный конвейер

### Monolithic (Variant A)
```
input
  │
  ▼
[gpt-4o-mini, JSON-схема всех 6 полей]    ← 1 call
  │
  ▼
{primary_intent, secondary_intent, target_type, target_name, urgency, negation}
```

### Multi-stage (Variant B)
```
input
  │
  ▼
[Stage 1: Normalize]                       ← 1 call (sequential)
  │
  ▼
{verbs, targets, tone, language}
  │
  ├──────────┬──────────┐
  ▼          ▼          ▼
[Stage 2:    [Stage 3:    [Stage 4:        ← 3 calls (parallel)
 Classify]    Extract     Sentiment +
              target]      negation]
  │          │          │
  ▼          ▼          ▼
{primary,    {target_   {urgency,
 secondary}   type,      negation}
              target_name}
  │          │          │
  └──────────┴──────────┘
              │
              ▼
[Local merge → final JSON]
```

**API calls на пример**:
- Mono: 1
- Multi: 4 (1 sequential + 3 parallel)

**Wall-clock latency**:
- Mono: 1× single call
- Multi: 2× single call (stage 1 + max(2,3,4))

**Total tokens** (грубо):
- Mono: 1 large prompt (~400 token system) + 50 token completion ≈ 450 input + 50 output
- Multi: 4 small prompts (~100-150 token system each) + 10-20 token completion each ≈ 500 input + 60 output total

Multi немного дороже по токенам, но возможно выигрывает по точности на каждом отдельном поле.
