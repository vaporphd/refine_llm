# День 7. Выводы

## Что построено

Inference-конвейер с тройной оценкой уверенности **поверх gpt-4o-mini без fine-tuning**. Применён к классификатору запросов разработчика (4 класса: search/understand/describe/modify), датасет `data/hard/` (40 holdout) + 10 adversarial шумных входов.

**Реализованы 3 подхода из 4 возможных:**

| Подход | Реализация | Cost-цена |
|---|---|---|
| **Constraint-based** | Регулярное выражение для допустимых лейблов + JSON-парсинг + проверка непустого ответа | 0 (post-process) |
| **Redundancy** | 3 параллельных вызова при `temperature=0.7` → majority vote | ×3 calls |
| **Scoring** | JSON-схема `{"label": ..., "confidence": 0.0-1.0}`, модель сама оценивает уверенность | вместо одного из vote-вызовов |

**Self-check намеренно пропущен** — он дороже всех (×2 calls на верификацию) и склонен к ложным подтверждениям (модель слишком соглашается с собой).

## Финальный gating-конвейер

```
input
  │
  ▼
[3 parallel calls @ T=0.7] (один с JSON-схемой)
  │
  ├─ 3/3 vote agree + confidence ≥ 0.85 + format ok  → OK     (принять)
  ├─ 3/3 vote agree but confidence < 0.85            → UNSURE (на ревью)
  ├─ 2/3 vote agree                                   → UNSURE
  ├─ vote ≠ scoring label                             → UNSURE
  ├─ all 3 different OR confidence < 0.5              → FAIL   (отклонить)
  └─ format invalid                                   → FAIL
```

## Результаты на 50 примерах (40 holdout + 10 adversarial)

### Распределение статусов

| Подмножество | OK | UNSURE | FAIL |
|---|---|---|---|
| Holdout (data/hard/eval) | 26 (65%) | 14 (35%) | 0 (0%) |
| Adversarial (шумные) | 2 (20%) | 8 (80%) | 0 (0%) |
| **Всего** | **28 (56%)** | **22 (44%)** | **0** |

### Главное: качество фильтрации

- **Accuracy на принятых (OK)**: **100%** (26/26 на holdout, 2/2 на adversarial)
- **Catch rate**: **100%** — **все 6 ошибок baseline помечены как UNSURE** (из 6 предсказаний которые baseline ошибся, gating не пропустил ни одно как OK)
- **Pickup rate** (holdout, % принятых): **65%**
- **Rejection rate** (adversarial, % отклонённых как UNSURE/FAIL): **80%**

### Сравнение с baseline (single-call)

| Метрика | Baseline | Gated |
|---|---|---|
| Обслужено примеров | 40/40 | 26/40 (14 на ручной ревью) |
| Accuracy на обслуженных | 85% (6 ошибок) | **100% (0 ошибок)** |
| API calls | 40 | 150 (×3.75) |
| Cost | ~$0.0015 | $0.0060 (×4) |
| Latency p50 | ~600ms | 885ms |
| Latency p95 | ~1200ms | 2885ms |

**Trade-off**: gated механизм отдаёт 14 примеров на human review, но на тех 26 что обслуживает — **никогда не ошибается**. Для критичных задач (где ошибка дороже ручного ревью) это огромный win.

## Калибровка confidence (Scoring) — *работает!*

| Метрика | Значение | Интерпретация |
|---|---|---|
| Spearman(confidence, correct) | **+0.501** | Сильная положительная корреляция |
| Средняя confidence на правильных | 0.86 | Близко к порогу 0.85 |
| Средняя confidence на ошибочных | 0.77 | Ниже порога |
| **Gap** | **+0.09** | Реально различает |

**Вывод**: на gpt-4o-mini self-confidence через JSON-схему даёт полезный сигнал. Это удивительно — обычно LLM плохо калиброваны. Эффект может быть обусловлен явной инструкцией в system prompt: "Use values below 0.6 if the request is ambiguous". Без такой подсказки confidence обычно сваливается в 0.95-0.99 на всё.

## Что какой подход поймал (изоляция вкладов)

Из 6 ошибок baseline пойманы:
- **Low confidence** (< 0.85): **6/6** — самый сильный сигнал
- **Vote split** (≠ 3/3 одинаковых): 2/6
- **Format invalid**: 0/6 — модель никогда не нарушает формат
- **Vote disagree scoring**: 0/6

**Вывод**: Scoring (низкая confidence) — самый информативный сигнал в этой задаче. Redundancy полезен только как добавочный (catches 2 cases that scoring already caught). Constraint бесполезен для этой задачи (формат на 100%).

В других задачах распределение будет другим: на open-ended генерации Constraint+Format будет важнее, на математических задачах Redundancy с majority vote сильнее scoring.

## Per-class breakdown (holdout)

| Class | OK | UNSURE | FAIL |
|---|---|---|---|
| search | 6 | 4 | 0 |
| understand | 5 | 5 | 0 |
| describe | 7 | 3 | 0 |
| modify | 8 | 2 | 0 |

**Modify самый "уверенный"** (8/10 OK) — у этого класса самые явные глаголы-маркеры (`add`, `fix`, `rewrite`, `добавь`).

**Understand часто помечается UNSURE** (5/10) — много вопросов с гибридной интенцией ("почему X и как починить?").

## Adversarial: какие шумы поймали правильно

| kind | status | pred | expected | conf | оценка |
|---|---|---|---|---|---|
| `?` | UNSURE | search | FAIL | 0.40 | ✅ поймали |
| `fix` | UNSURE | modify | modify | 0.70 | ⚠️ ложноотклонили (правильно угадали но низкий conf) |
| `что` | UNSURE | understand | FAIL | 0.50 | ✅ поймали |
| `найди и удали` | OK | modify | modify | 0.90 | ✅ правильно |
| `explain ... fix it` | UNSURE | understand | modify | 0.80 | ✅ поймали несогласие |
| `describe and rewrite all` | UNSURE | modify | modify | 0.70 | ⚠️ ложноотклонили |
| Spanish `donde esta` | OK | search | search | 0.90 | ✅ правильно (модель работает на испанском!) |
| German `wie funktioniert` | UNSURE | understand | understand | 0.80 | ⚠️ почти OK |
| `lol idk man yikes halp` | UNSURE | understand | FAIL | 0.50 | ✅ поймали |
| `найди объясни и удали` | UNSURE | search | modify | 0.70 | ✅ поймали несогласие |

**8/10 шумных входов отклонено или помечено UNSURE.**

Интересные кейсы:
- Испанский `donde esta` — gpt-4o-mini корректно классифицировал как search с confidence 0.90
- Немецкий `wie funktioniert` — правильно как understand, но conf 0.80 → UNSURE (бордерлайн)
- Тройная интенция `найди объясни и удали` — пометил UNSURE через split vote (то что нужно)

## Latency и cost

- **Wall clock на 50 примеров**: 657 секунд (~13 секунд на пример из-за последовательного запуска)
- **Latency на 1 примере**: p50=885ms, p95=2885ms (max из 3 параллельных вызовов внутри примера)
- **Cost**: $0.006 на весь эксперимент

**Важный нюанс**: latency p95=2885ms — это худший вызов из 3 параллельных. Если убрать tail, реальный p50 на самом быстром из 3 был ~500ms, что близко к baseline single-call.

**Возможная оптимизация**: parallel processing across examples (не только внутри примера). Сейчас 50 примеров шли последовательно, можно было бы в 5-10 раз быстрее с pool of workers.

## Главные выводы

### 1. Gated inference радикально повышает качество за умеренную цену
85% baseline → **100% на принятых**, ценой ×4 cost и 35% отказов на ручной ревью. Для high-stakes задач (выбор действия, медицина, юридические решения) это идеальный trade-off.

### 2. Scoring (self-confidence) — самый информативный сигнал
Spearman +0.5 — серьёзная корреляция с правильностью. **Почему сработало**: явная инструкция в system prompt калибрует модель ("Use values below 0.6 if ambiguous"). Без такой инструкции self-confidence обычно бесполезен.

### 3. Redundancy полезен, но overlap со Scoring большой
Из 6 пойманных ошибок только 2 были пойманы через vote split, остальные 4 уже ловились через scoring. Если бюджет ограничен — можно убрать 1 из 2 plain vote-вызовов и оставить только scoring + 1 plain.

### 4. Constraint бесполезен в этой задаче (но не в других)
gpt-4o-mini никогда не нарушает формат при правильном system prompt. Constraint полезен только как защита от джейлбрейков и edge cases. В open-ended задачах (генерация JSON, structured extraction) ситуация противоположная.

### 5. Adversarial input — самый важный тест-сет
80% rejection rate на шумных входах — это лучшая метрика для выбора порога threshold. Корректные holdout-примеры почти все попадают в OK, а вот реальный продукт встречает кучу мусора в проде.

### 6. Threshold 0.85 — компромисс
- 0.85 → 65% pickup, 100% accuracy на принятых
- 0.95 → пройдёт ~30% (дороже, но ещё чище)
- 0.70 → пройдёт ~80%, но просочится 1-2 ошибки

Для production стоит делать **двухуровневую** систему: 0.95 → автоматический OK, 0.70-0.95 → ручной ревью с приоритетом, < 0.70 → отклонить.

## Ограничения и что недоделано

1. **Маленький eval (50 примеров)** — Spearman +0.501 на 40 точках имеет широкий доверительный интервал. На 200+ точках было бы надёжнее.
2. **Не тестировали другие модели** — gpt-4o-mini неожиданно хорошо калиброван, но gpt-3.5 или local models могут вести себя иначе.
3. **Self-check не реализован** — это могла бы быть 4-я ось анализа.
4. **Sequential execution** — эксперимент шёл 13 минут вместо потенциальных 1-2 (нужен concurrent worker pool).
5. **Threshold 0.85 не оптимизирован** — выбран из общих соображений. Empirically PR-curve показала бы оптимум.

## Артефакты

- `scripts/confidence_check.py` — основной gating
- `scripts/confidence_compare.py` — анализ vs ground truth
- `data/hard/adversarial.jsonl` — 10 шумных тест-кейсов
- `results/confidence_hard.json` — сырые ответы 150 API вызовов + аггрегаты
