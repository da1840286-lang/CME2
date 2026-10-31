# Передачи между этапами CME (FIND → ALL-отбор → ANALYZE → BRAIN → FRAME)

Что один этап пишет, что следующий читает, какие инварианты держит код и какой тест это проверяет.
Здесь нет новой архитектуры: это описание уже существующих файлов и функций.

| Передача | Файл / функция | Обязательные поля | Инвариант | Тест |
|---|---|---|---|---|
| Меню → FIND | `live-smoke.yml` (блоки `find` и `all`), `run_find(params)` | country, period, max_requests, max_ads, keywords, offer, selection_mode, video_duration_range, performance_order_bys, include_retention | оба блока строят одинаковые `params`; значения = то, что введено | `test_workflow_menu` |
| Меню → проверка входов | bash-шаг «Validate credentials, FIND parameters…» | те же | всё, что шаг принимает, принимает и FIND; недопустимые комбинации отклоняются до платных вызовов | `test_workflow_menu` |
| FIND → кандидаты | `candidates.json` | `id`, `video_url`, `source_selection` (role, promotion_evidence, target_promotion_evidence, mode), `analysis_eligibility`, `stage`, `source_gate_eligible`, `query_product_match` (`null` = не вычислялся), `performance_evidence`, `request` | все статусы допуска пересчитываются одной функцией `sync_pattern_source_status` после финального role/promotion gate; `performance_plan_effective` называет то, что реально выполнено | `test_stage_handoffs`, `test_patch5` |
| Кандидаты → отбор `all` | `select.rank_pattern_candidates` / `pattern_source_reason` | как выше | в `product` старые записи без `query_product_match` допускаются только при independently verified `TARGET_PRODUCT` + target promotion; в `mechanism` требуется `matches=true` и непустой `query_anchor`, иначе fail-closed; | `test_stage_handoffs` |
| Отбор → Analyze | `selected_candidate.json`, `Store.ingest/get` | `id`, `video_url`, `source_selection`, `analysis_eligibility`, `performance_evidence`, `commercial_evidence`, `query_product_match`, `duration_seconds`, `find_request` | поля FIND не теряются в SQLite и в `selected_candidate.json` | `test_stage_handoffs` |
| Analyze → Brain | `source_contract.json` → `build_source_transfer_context` | 7 секций анализа, `evidence`, `limits`, **`section_kinds` (OBSERVED/INFERRED)** | URL и пути не пересекают границу; метка доказательности сохраняется; связка `result`, построенная из INFERRED-секции, помечена `source_kind: INFERRED`, и модель знает, что это гипотеза | `test_stage_handoffs`, `test_source_transfer_bridge` |
| Brain → gate → атомы | `target_fit_gate.json`, `atoms.json`, `atoms_manifest.json` | `contract_id`, `atoms_sha256`, `source_contract_sha256`, `mechanism_links`, `transfer_decision` | `abstract_mechanism` атомов = утверждённому гейтом; хеш контракта в манифесте = хешу файла | `test_stage_handoffs`, `test_target_fit_gate` |
| Атомы → Frame | `atoms.json`, `state.json`, `receipts/drec_*.json` | `contract_id`, `rows` по MUST-атомам, `atom_provenance` | один `contract_id` в манифесте, состоянии и квитанции; каждый MUST image-атом проверен в квитанции | `test_stage_handoffs`, `test_firstrun` |

## Известные границы (не ошибки)
- Стандартный Product-путь (`find_performance_order=default`) использует целевой Sponsored Search: точное название оффера, затем ключевые слова оператора, затем тип/категория товара. Ads Top и unfiltered retention не вызываются.
- На один Product FIND приходится не более трёх запросов. Ключевые слова, которые не поместились, перечисляются в `candidates.json → request.dropped_operator_keywords`.
- Numeric `play_2s_rate`/`play_6s_rate` и target-specific retention этим продуктовым поиском не предоставляются. Если пользователь запросил эти показатели, `performance_evidence.requested_metric_status` явно сообщает `NOT_OBTAINABLE_FOR_PRODUCT_QUERY`; они не подменяются нулём и не становятся gate. Старый Ads Top путь доступен только при явном `performance_plan="legacy_ads_top"` вне стандартного workflow.
- `atoms_manifest.json` использует `schema_version=2`; `branding_policy` и нормализованный `approved_brand_marks` привязаны к fingerprint. Workflow menu — единственный источник политики. QA-only атом не считается визуальным переносом: механизм должен быть в MUST image atom.
