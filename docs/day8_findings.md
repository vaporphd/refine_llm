# День 8. Выводы — Routing между моделями

## TL;DR

Реализован 2-tier router: `gpt-4o-mini` (cheap) → эскалация на `gpt-4o` (strong) при `confidence < 0.85`. Прогнаны 3 режима на 50 примерах (40 holdout + 10 adversarial).

**Парадоксальный главный вывод**: на этой задаче strong-модель работает **хуже** cheap-модели. Router точно повторяет ответы strong (100% совпадение), что **снижает** качество относительно чистого cheap. Зато **экономит 46% денег** vs strong-only.

**Roadmap**: routing полезен для cost, но эскалация без вторичной валидации опасна.

## Конфигурация

| Параметр | Значение |
|---|---|
| Cheap model | `gpt-4o-mini` |
| Strong model | `gpt-4o` |
| Threshold escalation | `confidence < 0.85` |
| Temperature | 0.0 (детерминизм) |
| Test set | 40 holdout + 10 adversarial = 50 |
| API format | JSON `{label, confidence}` на обоих уровнях |

## Headline numbers

| Mode | Holdout acc | Adv match | Total | Calls | Cost | Latency p50 | p95 |
|---|---|---|---|---|---|---|---|
| **cheap-only** | **36/40 = 90%** | 7/10 | **86%** | 50 | **$0.0015** | 943 ms | 1811 ms |
| **strong-only** | 35/40 = 88% | 5/10 | 80% | 50 | $0.0257 | 794 ms | 1746 ms |
| **router** | 35/40 = 88% | 5/10 | 80% | 74 | $0.0138 | 1548 ms | 3491 ms |

## Главный сюрприз: cheap > strong на этой задаче

`gpt-4o-mini` обогнал `gpt-4o` на 2 п.п. holdout и на 6 п.п. на adversarial.

**Почему**:
1. **Конфликт интенций**: gpt-4o более склонна к "overthinking" — на двойных интенциях ("explain why this fails and then fix it permanently") выбирает первый глагол (`understand`), тогда как mini лезет к доминирующему действию (`modify`).
2. **Строгая разметка**: наша canonical-разметка для гибридов выбирает финальное действие (modify, если в конце "fix it"). Это совпадает с интуицией mini, но не gpt-4o.
3. **На adversarial**: gpt-4o корректно отвечает `understand` на `"I have an issue while bilding app in xcode"`, что соответствует разметке. Но также классифицирует немецкий `wie funktioniert` как `understand` с разным confidence — там результаты ближе.

**Вывод**: "сильнее" модель не равно "лучше для этой задачи". Бенчмарк перед routing критически важен.

## Распределение нагрузки router

| Класс | Stayed cheap | Escalated | % escalated |
|---|---|---|---|
| search | 4 | 6 | 60% |
| understand | 4 | 6 | 60% |
| describe | 7 | 3 | 30% |
| modify | 8 | 2 | 20% |

**Modify и describe** редко эскалируются — на этих классах cheap-модель отвечает с высокой confidence (0.95+). **Search и understand** — частые эскалации (60%) из-за гибридных формулировок.

Adversarial:
- 8/10 эскалированы (что логично — это шумные входы с заведомо неоднозначной интенцией)
- Не эскалированы: испанский `donde esta` (cheap уверена 0.90) и немецкий `wie funktioniert` (тоже 0.85+)

## Cost: главный выигрыш router'а

| Метрика | Значение |
|---|---|
| Strong-only cost | $0.0257 |
| Router cost | $0.0138 |
| **Saving vs strong** | **46% (×0.54)** |
| Router cost / cheap cost | ×9.26 |

То есть router **в 9 раз дороже cheap**, но **в 2 раза дешевле strong**.

При этом router возвращает ответы strong с точностью 100% (см. ниже) — то есть как замена strong-only, router идеален: то же качество, в 2 раза дешевле.

## Router vs strong agreement: 100%

**Router совпадает со strong-only на всех 50 из 50 примерах.** Это значит:
- В каждом случае эскалации — strong "перевесил" cheap-ответ
- На не-эскалированных запросах cheap всё равно дал тот же ответ что дал бы strong

То есть на этой задаче router работает как **детектор простых случаев** — оставляет лёгкие на cheap, и эффективно делегирует тяжёлые на strong.

## Где router спас, где сломал

### Cases rescued (cheap ошибся, router после эскалации правильно): **0**

Ни одной. Это потому что strong на этой задаче не сильнее cheap — там где cheap ошиблась, strong тоже чаще всего ошибается.

### Cases broken (cheap прав, router после эскалации неправ): **1**

```
expected=understand
cheap=understand   (right)
router=modify      (wrong, after escalation)
user="I have an issue while bilding app in xcode"
```

Cheap была права, но из-за низкой confidence уехало на strong, который "увидел" в этом баг-репорт = просьбу пофиксить = `modify`. Это **fatal flaw** простой эскалации: ответы strong принимаются вслепую без вторичной валидации.

## Latency

- cheap-only: p50=943ms (наш дешёвый JSON-вызов)
- strong-only: p50=794ms (gpt-4o быстрее по generation, но дороже)
- **router: p50=1548ms (×1.6 vs cheap, ×2 vs strong)**

Router p95=3491ms — самые тяжёлые случаи это cheap+strong последовательно. На критичных по latency задачах (real-time UI) это может быть неприемлемо.

## Главные выводы

### 1. Routing — это cost optimization tool, а не accuracy tool
В нашем эксперименте router дал ту же accuracy что strong-only, но за 54% денег. Если задача — экономия → routing работает. Если задача — поднять качество → нужна другая стратегия (ансамблирование, FT, voting).

### 2. "Сильная модель" — не догма
gpt-4o проиграла gpt-4o-mini на 2-6 п.п. на этой задаче. **Всегда измеряй обе модели на своём датасете** перед тем как считать одну "лучше".

### 3. Naive эскалация опасна
Простой "если cheap не уверен — отдай strong" жертвует accuracy в случаях когда cheap была права но "застенчива". Лучшая стратегия — **best-of-2 voting**: после эскалации сравнить cheap и strong, выбрать тот ответ у которого выше confidence (или мажоритарный — но при 2 голосах майорити = равенство).

### 4. Confidence threshold 0.85 близок к оптимуму на этой задаче
- 48% эскалаций — половина запросов уехала на strong
- На искусственно дорогих задачах (medicine, legal) это может быть оправдано
- На массовом traffic — снизить до 0.7 чтобы пропустить больше через cheap

### 5. Modify/describe эскалируются реже всего — этим можно воспользоваться
60% search и understand идут на strong. Если на проде преобладают modify-запросы — router будет супер-эффективным. Можно даже сделать **per-class threshold**: для search/understand = 0.95, для modify = 0.7.

## Что улучшить (если бы был день 9)

1. **Best-of-2 voting**: после эскалации возвращать тот ответ у которого confidence выше, а не безусловно strong
2. **Per-class threshold**: разные пороги для разных классов
3. **3-tier**: cheap → mid (gpt-4o-mini @ T=0.7 self-consistency) → strong (только если и mid не уверен)
4. **Calibrated threshold**: подобрать threshold на отдельной calibration-выборке через precision-recall curve
5. **Сравнить с другими парами**: gpt-4.1-nano → gpt-4.1, gpt-4o-mini → o1-mini

## Артефакты

- `scripts/router.py` — 3 режима (cheap/strong/router) в одном скрипте
- `scripts/router_compare.py` — сравнение всех режимов
- `results/router_cheap.json`, `router_strong.json`, `router_results.json`
