# День 9. Выводы — Multi-stage vs Monolithic inference

## TL;DR

На задаче извлечения 6 полей (`primary_intent`, `secondary_intent`, `target_type`, `target_name`, `urgency`, `negation`) из запроса разработчика **monolithic подход с одним вызовом обогнал 4-stage декомпозицию по всем метрикам** — accuracy, cost, latency-per-call (но НЕ wall-clock — там multi выигрывает за счёт параллелизма).

**Composite accuracy**: mono 46% vs multi 22% — провал в 2 раза.
**Cost**: multi в 1.71× дороже.
**Wall-clock**: multi 22s vs mono 28s (✅ multi быстрее на 22% — параллелизм дал эффект).

**Главный урок**: декомпозиция не бесплатна. Для коротких enum-полей mono более точен. Multi-stage оправдан только когда стадии имеют сложную семантику и контекст одной мешает другой.

## Конфигурация

| Параметр | Значение |
|---|---|
| Модель (обе режима) | `gpt-4o-mini` |
| Temperature | 0.0 |
| Тест-сет | 50 примеров (40 holdout + 10 adversarial) |
| Поля разметки | 6 (primary_intent, secondary_intent, target_type, target_name, urgency, negation) |
| Format | JSON через `response_format={"type":"json_object"}` |

## Архитектура

### Mono (1 call)
Один большой system prompt со всеми 6 полями, 1 вызов на пример.

### Multi (4 calls, parallel-aware)
```
Stage 1 (sequential): Normalize → {verbs, nouns, tone, language}
                  ┌────────┬─────────┐
Stage 2 ────┐    ▼        ▼         ▼   (parallel)
Stage 3 ────┼─→ Classify  Extract   Sentiment
Stage 4 ────┘    intents   target   urgency
                  ↓        ↓        ↓
            [Local merge → final 6-field JSON]
```

## Headline numbers

| Метрика | Mono | Multi | Δ |
|---|---|---|---|
| Examples | 50 | 50 | — |
| API calls | 50 | 200 | ×4 |
| **Composite accuracy** | **46%** | **22%** | **−24 pp** |
| Cost | $0.0051 | $0.0088 | +71% |
| Wall clock (parallel) | 27.7s | 21.9s | **−21%** ✅ |
| Latency p50 (per example) | 1977ms | 2745ms | +39% |
| Latency p95 | 4169ms | 5930ms | +42% |
| Tokens total | 26344 | 44807 | +70% |

## Field-level accuracy

| Field | Mono | Multi | Δ |
|---|---|---|---|
| `primary_intent` | 90% | 90% | 0pp |
| `secondary_intent` | 92% | 94% | +2pp ✅ |
| `target_type` | 66% | 48% | **−18pp** ❌ |
| `target_name` | 72% | 70% | −2pp |
| `urgency` | 94% | 66% | **−28pp** ❌❌ |
| `negation` | 100% | 100% | 0pp |

**Где mono впереди:**
- `target_type` (−18pp у multi): отдельная стадия классификации type без полного контекста чаще выбирает `other`
- `urgency` (−28pp у multi): Stage 4 видит только нормализованные признаки + текст. Без доступа к "общей картине" (через Stage 1 nouns/verbs) теряет нюансы тональности — типа "поставь главной модель X" mono читает как `normal`, multi — как `low` (видя короткий список verbs)

**Где multi немного лучше:**
- `secondary_intent` (+2pp): отдельная фокусированная стадия классификации интенций ловит гибриды чуть лучше

**Где паритет:**
- `primary_intent` (90% оба): обе архитектуры классифицируют одинаково
- `negation` (100% оба): тривиальное поле, обе справляются

## Pairwise сравнение (50 примеров)

| Случай | Кол-во |
|---|---|
| Both correct (composite) | 10 |
| Mono only correct | **13** ⬅ доминирует |
| Multi only correct | **1** |
| Both wrong | 26 |

Multi "спас" mono всего в 1 случае: `"почему disasm падает на этом байте 0xFF 0xC0?"`.

Mono "спас" multi в 13 случаях. Типичные потери multi:
- `"найди где мы считаем retry count"` — multi сломал target_type и target_name
- `"where does the packet tunnel config live?"` — multi проиграл urgency
- `"поставь главной модель minimax/minimax-m2.5:free"` — target_type ушёл в `other` вместо `config`
- `"describe cron scheduler и его взаимодействие с job runner"` — target_type+urgency

## Per-field error attribution в multi-stage

| Field | Errors |
|---|---|
| target_type | 26 (52% от всех) |
| urgency | 17 |
| target_name | 15 |
| primary_intent | 5 |
| secondary_intent | 3 |
| negation | 0 |

**`target_type` и `urgency` — главные источники регрессии**. Это поля где **полный контекст важен**, а multi даёт стадии только: `(оригинальный текст + нормализованные признаки)`. Этого недостаточно когда классификация требует тонкого ощущения текста как целого.

## Token usage

| | Mono | Multi | Ratio |
|---|---|---|---|
| Input tokens | 23732 | 40254 | ×1.70 |
| Output tokens | 2612 | 4553 | ×1.74 |
| Total | 26344 | 44807 | ×1.70 |

Multi-stage платит 70% наценкой по токенам **в основном за повторные system prompts**. Каждая из 4 стадий несёт свою инструкцию (~100 input tokens), и в Stages 2-4 ещё передаётся `enriched_user` который включает оригинал + features.

## Wall-clock vs per-example latency

Это разные метрики с разным значением:

- **Wall-clock** (multi 22s vs mono 28s): время на весь dataset при `max_workers=8`. Multi выигрывает потому что параллелит 3 stages внутри одного примера, а у mono каждый пример — атомарная единица. Параллелизм лучше утилизирует API rate limit.

- **Latency-per-example p50** (multi 2745ms vs mono 1977ms): время одного запроса end-to-end. Multi медленнее потому что: stage1 sequential + max(stage2,3,4). Это критично для real-time UX.

**Вывод**: если ты обрабатываешь batch (фоновая разметка) — multi-stage выгодна. Если real-time UI — mono.

## Главные выводы

### 1. Декомпозиция стоит дорого
Multi-stage принёс +71% cost, +39% per-call latency, и **−24pp composite accuracy** на этой задаче. Декомпозиция должна оплачиваться сильным выигрышем, который тут не появился.

### 2. Контекст-зависимые поля страдают первыми
`target_type` и `urgency` требуют ощущения текста как целого. Изолировать их в отдельных стадиях с обрезанным контекстом — плохая идея. **Если поле требует full context — оставь его в monolithic части**.

### 3. Параллелизм multi-stage спасает только wall-clock
22s vs 28s wall-clock — единственная победа multi. Но это batch-метрика, не latency для пользователя. На real-time это не помогает (2.7s vs 2.0s p50).

### 4. Mono prompt легче калибровать
В mono ты пишешь ОДИН большой prompt с 6 полями и калибруешь его правила. В multi нужно калибровать 4 prompts отдельно, и баги в Stage 1 (Normalize) каскадно разрушают всё ниже. Эту хрупкость хорошо видно: ошибка в normalize → enriched_user не содержит нужных признаков → 3 стадии ниже работают вслепую.

### 5. Где multi-stage всё-таки помог
- `secondary_intent` +2pp — гибридные интенции лучше различаются с фокусированной стадией
- 1 редкий случай (`disasm 0xFF 0xC0?`) — изолированная сентимент-стадия дала правильный urgency

### 6. Когда multi-stage был бы оправдан
Декомпозиция работает когда:
- Каждая стадия требует **разной модели или инструмента** (например, NER-модель + классификатор + reasoner)
- Стадии имеют **сложную внутреннюю логику** (multi-step reasoning, поиск в базе знаний)
- Данные из разных стадий нужно **сохранять отдельно** для последующего использования
- Output одной стадии **существенно меняет** prompt последующей (а не просто добавляется к нему)

В нашем случае все 4 стадии — простые enum-классификации с одним контекстом. Это идеальный mono-кейс.

## Что улучшить (если бы был день 10)

1. **Гибрид**: mono для простых полей (urgency, negation, primary_intent), multi для сложных (target_type+target_name требуют extraction + classification)
2. **Stage 1 → MAJOR refactor**: nounns/verbs нормализация теряет полезный контекст. Лучше вообще убрать Stage 1 и делать 3 параллельных вызова на оригинальном тексте
3. **Schema-strict response**: использовать `json_schema` mode (поддерживается на gpt-4o-mini) чтобы заставить выходной формат валидным даже без JSON-объяснений в prompt
4. **Voting между mono и multi**: ансамблирование двух режимов — если соглашаются → принять, если нет → human review

## Артефакты

- `data/hard/rich_eval.jsonl` — 50 примеров с 6-полевой разметкой
- `scripts/multistage.py` — оба режима с параллелизмом stages 2-4
- `scripts/multistage_compare.py` — сравнение field-level + composite
- `results/multistage_mono.json`, `results/multistage_multi.json` — сырые результаты обоих прогонов
