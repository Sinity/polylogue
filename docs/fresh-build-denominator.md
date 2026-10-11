# Private fresh-build denominator

The denominator tool inventories declared source roots without parsing transcripts or opening an archive. It projects production-normalized workload counts from supplied component receipts. Inventory and projection files contain private source coordinates and must remain outside the checkout. The tool refuses to overwrite a result or write into a declared source root.

`devtools.fresh_build_bench.denominator.main(argv)` exposes two actions. The fresh-build command surface owns registration of this entry point.

```text
inventory --plan PLAN --output NEW_STAGE --members N
          [--previous PRIOR_STAGE] [--structural-bytes N]
project --inventory STAGE --calibration CALIBRATION --output NEW_RESULT
```

## Inventory contract

A private plan has format `polylogue.denominator-plan.v1`, a `frontier` object with nonempty `id` and `basis`, and a nonempty `roots` list. Each root declares `id`, absolute `path`, native watch-source `origin`, `stratum`, `shape` and `population_kind` (`normal` or `exceptional-hooks`). A stratum has one declared shape and population kind. The frontier basis describes the owner's actual input boundary, such as a sealed immutable tree or a timed live-tree observation. Naming a frontier does not certify its currentness.

The collector calls the existing native ordered source walk with all declared roots together. Original layout admission, source ownership, link containment and walk faults remain authoritative. Unit grouping is the benchmark sampler's existing grouping: Claude parent and children share a unit, Gemini retains its project grouping, and Codex loose files remain individual units. These are source units, not certified independent sessions or revision chains. Static assembly coordinates use the production Claude and Codex resolvers. Present files, optional absence and refused or nonregular coordinates are recorded separately. Repeated shared metadata coordinates contribute bytes once, including when a coordinate is also an ordinary walked member.

An immutable JSONL stage contains a header, streamed member and dependency observations, and a final trailer. The header binds the exact plan, predecessor stage, collector body, checkout revision, runtime and pre-walk imported source authority. Every originally bound authority entry must remain present and unchanged in later stages; later imports may be added. This is not a full post-walk import census. The trailer reports its continuation, admitted members, inspected directory entries, dispositions and changed shared observations. `--members` pauses after that many admitted files. It does not cap the population or silently skip the remainder. Parent listings can recur on resume, so summed inspected-entry and fault counts describe observation work rather than distinct population entries. A refused root ends that root's attempt and remains a coverage gap.

The default is stat-only. Optional `--structural-bytes` authorizes at most that many bytes of physical JSONL/NDJSON reading during the stage. It records nonblank complete LF-delimited physical lines, byte coverage and a prefix hash. Only a fully read file whose member observation and read-start/read-end observations agree receives a complete SHA-256. An unfinished final line is not counted as a complete record. These counts do not validate JSON or count normalized messages. Partial reads remain partial when traversal resumes at the next file; there is no automatic second content scan.

Stages can be continued until traversal ends. Projection checks the immutable predecessor chain, each terminal trailer and every required source binding. `traversal_complete` describes the end of enumeration; `population_complete` additionally requires no observed walk fault, member refusal, changed observation or refused dependency. Neither certifies that an unfrozen live tree remained unchanged behind the cursor. The projection uses a disposable private SQLite accounting file for unit/path deduplication and byte quantiles, then closes and removes it. It never opens a product archive database.

## Calibration contract

Calibration has format `polylogue.denominator-calibration.v1`, `sampling_design` and `samples`. The design names `kind` (`probability` or `targeted-model`) and the actual selection `mechanism`. Each row declares `frontier_id`, observed `unit_id`, `stratum`, `shape`, `inclusion_weight`, optional `inclusion_probability`, `held_out`, and immutable absolute-path/SHA-256 references `selection` and `receipts` (a list).

Probability samples require an explicit positive inclusion probability at most one and a matching inverse-inclusion weight. Targeted samples use explicitly declared positive scenario weights. The collector does not attest that a caller's sampling mechanism was executed. A declared stratum shape is a model supplied by the caller; artifact-kind distributions make mixed populations visible.

The selection file binds the row's frontier, unit, stratum and shape, sets `complete_unit`, references its immutable `closure`, and lists ordered `normalization_inputs` with distinct `member_id`, `input_sha256` and `recipe_sha256`. All corresponding distinct immutable production receipts must be supplied. Equal bytes can represent separate artifact preparation work; raw digest equality does not collapse separate inputs. This supports a unit with several normalization inputs, such as a parent and children. The selection/closure files attest membership and dependency coverage; filesystem stats alone do not prove content equality or assembly completeness.

The supported normalization owner is `polylogue.component-receipt.v1`, component `retained-preparation`. A receipt must bind a clean candidate, exact product tree, runtime, imported product modules, input and recipe. It must retain all original validation verdict fields and `output.status=passed`. `output.preparation` must declare `state=complete`, `complete_success=true`, matching input SHA-256, both accepted and parsed prefixes `null`, and a fully consumed session count matching normalized sessions and the session content-hash list. Prefix preparation cannot calibrate a complete source unit. Missing or failed validation is refused, rather than treated as success.

`output.normalized.count_basis` must be `production-preparation-before-lineage`. Its nonnegative integer sessions, messages, blocks and events count complete production preparation work before archive lineage resolution, deduplication and materialization. Parent and child inherited prefixes can therefore contribute work more than once. Projections describe preparation-work population, not final Index row counts. Schema-validation `sample_count` is refused as a substitute. Missing counts in any input keep that unit's metric unknown. Different candidate/runtime calibrations cannot be pooled.

Revisions and jobs remain unknown because preparation does not publish or certify their population. Embeddings are excluded. Hooks retain their separate exceptional-population estimate and do not inflate the ordinary denominator. Acquisition, CPU service, serialized publication, remote completion and final settlement are separate unestimated costs; this tool does not infer their elapsed times from bytes or normalized counts.

## Projection and uncertainty

Within each stratum, the estimator multiplies observed source-unit population by the inclusion-weighted mean normalized count per calibration unit. This is a ratio-weighted model estimate, with its sampling design recorded. Partial inventories project only observed units. Unsupported strata and metrics produce explicit unknowns.

The result reports effective sample size, weighted dispersion, observed per-unit range and its population-scaled scenario range. It supplies no confidence interval. Targeted morphology samples cannot establish statistical coverage, and even a probability sample needs design-aware uncertainty beyond these descriptive summaries. Unobserved long tails remain unknown. Member bytes, source units, normalized sessions, messages, physical lines, revisions and jobs retain separate fields.

Held-out units are excluded from fitting. Their normalized actual counts, predictions and relative errors are recorded per metric. `held_out_checked_metrics` is populated only when traversal and observed coverage are complete. It is a prediction check, not performance acceptance, proof of representativeness or source currentness. Without bound held-out normalization, estimates remain unvalidated scenarios. The result does not claim live archive debt or forecast a daemon completion time.

All examples and tests use neutral synthetic files. A real denominator estimate requires an explicitly authorized population inventory, actual selection/closure evidence, production-normalized calibration and separate held-out qualification.
