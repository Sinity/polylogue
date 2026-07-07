# Demo Radar

Recurring demo-generation checkpoint for the Polylogue conductor loop. Keep it
current enough that the next agent can see what artifact should exist next,
what was selected, and what proof/caveat makes the artifact honest.

Current default: `.agent/demos` is the curated demo shelf. Older entries below
may mention `/realm/inbox/demos_polylogue`; treat those as historical paths, not
current defaults.

## 2026-06-30T04:17:00+02:00

Trigger: live context-image demo slice
Candidate demos: refresh readable demos bundle; create query-to-context-image live packet; capture context-pack omission as substrate gap
Selected/improved demo: `/realm/inbox/demos_polylogue/03-live-context-image`
Artifact action: generated compact query output, context-pack JSON/Markdown, bounded read excerpts, README, manifest, and regenerated `CONCATENATED_READABLE.md`
Proof/caveat: query/read composition works on the canonical archive, but context-pack currently omits long sessions whole instead of slicing matched windows
Next demo question: can intra-session context slicing make the same live packet include useful matched context without a recovery/export/demo silo?

## 2026-06-30T04:31:21+02:00

Trigger: context slicing proof passed
Candidate demos: before/after live context-image packet; focused unit proof; later browser/API demo
Selected/improved demo: /realm/inbox/demos_polylogue/03-live-context-image after-slicing artifacts
Artifact action: added context-image.after-slicing.json/md and regenerated CONCATENATED_READABLE.md
Proof/caveat: focused context-pack tests pass; live artifact now has one bounded segment, zero omissions, explicit caveats
Next demo question: Should next loop expose message-window controls in CLI/MCP/API, or keep defaults internal and improve query-anchor centering first?

## 2026-06-30T04:45:48+02:00

Trigger: context-pack silo audit corrected the slice
Candidate demos: query-preserving context-image artifact; future projection/layout DSL demo; public context-pack token retirement
Selected/improved demo: /realm/inbox/demos_polylogue/03-live-context-image query-after-slicing artifacts
Artifact action: added context-image.query-after-slicing.json/md, archive-status.after-slicing.json, updated README/MANIFEST, regenerated CONCATENATED_READABLE.md
Proof/caveat: focused tests pass; live artifact proves ContextSpec.seed_query preserves match anchors and bounded ContextImage projection emits caveats rather than pretending completeness
Next demo question: Should the next slice replace read --view context-pack with an explicit projection/layout expression, or first add the missing filter fields to ContextSpec so context image absorbs the remaining selector helper?

## 2026-06-30T04:52:38+02:00

Trigger: context-image selector substrate cleanup
Candidate demos: public context-pack token retirement; filter-capable ContextSpec demo; API/web context-image route demo
Selected/improved demo: /realm/inbox/demos_polylogue/03-live-context-image remains the live artifact shelf
Artifact action: recorded selector cleanup as architecture proof; no new artifact file needed because behavior artifact already proves ContextImage output
Proof/caveat: focused tests pass and rg found no old MCP selector imports; caveat: public context-pack read view and MCP tool labels still remain
Next demo question: Next demo should show a query/filter expression compiling directly to ContextImage without context-pack-named API or view semantics

## 2026-06-30T05:00:30+02:00

Trigger: filter-capable ContextSpec proof on live archive
Candidate demos: public context-pack token retirement; explicit projection/layout DSL; browser/API context-image route demo
Selected/improved demo: /realm/inbox/demos_polylogue/03-live-context-image/context-image.filtered-spec.json
Artifact action: added filtered-spec live artifact and regenerated README/MANIFEST/CONCATENATED_READABLE
Proof/caveat: focused tests pass; live artifact proves filtered selection is in ContextSpec/compile_context; caveat: CLI/MCP labels still say context-pack
Next demo question: Next demo should exercise a context-image-named read/projection route instead of the context-pack lens

## 2026-06-30T05:21:02+02:00

Trigger: public context-image surface migration
Candidate demos: successor-context DTO cleanup; generated material-origin naming audit; projection/layout DSL demo
Selected/improved demo: /realm/inbox/demos_polylogue/03-live-context-image/context-image.public-view.json
Artifact action: added public context-image live artifact, updated README/MANIFEST, regenerated CONCATENATED_READABLE
Proof/caveat: focused tests and live command prove context-image is public; caveat: successor-context internals still contain ContextPackOmission names
Next demo question: Next demo should show successor context as ContextImage/projection output or justify keeping a distinct successor-context DTO

## 2026-06-30T05:27:10+02:00

Trigger: successor-context terminology cleanup
Candidate demos: generated_context_pack material-origin audit; context-image projection/layout DSL demo; live temporal-analysis demo refresh
Selected/improved demo: /realm/inbox/demos_polylogue/03-live-context-image/context-image.public-view.json
Artifact action: no new demo file; updated internal terminology so public demo/read surface and successor-context internals no longer say context-pack
Proof/caveat: insight and benchmark tests pass; grep shows only generated_context_pack material-origin vocabulary remains
Next demo question: Next demo should either justify/rename generated_context_pack with schema evidence or demonstrate another general projection/query capability on the live archive

## 2026-06-30T05:48:41+02:00

Trigger: situation brief 5.4: finish one externalizable artifact
Candidate demos: agent recovery two-arm demo; claim-vs-evidence report; methodology post evidence pack
Selected/improved demo: /realm/inbox/demos_polylogue/04-claim-vs-evidence
Artifact action: generated report.md, README.md, charts, MANIFEST.readable.json, and CONCATENATED_READABLE.md; extended scripts/agent_forensics.py instead of adding a standalone silo
Proof/caveat: proof: report ran over active v18 archive with 13,208 sessions and 3,833,656 messages; caveat: silent_proceed is lexical next-turn acknowledgment heuristic over structured failures, not final semantic judgment
Next demo question: stop-rule decision: either publish/hand off this first instance as sufficient, or add only one minimal non-silo refinement such as JSON sample export

## 2026-06-30T10:49:27+02:00

Trigger: temporal demos needed reusable composition instead of bespoke report glue
Candidate demos: temporal evidence-window proof; phase-latency devloop report; query/read temporal projection wiring
Selected/improved demo: `/realm/inbox/demos_polylogue/13-temporal-evidence-window`
Artifact action: generated `summary.json`, `temporal-window.json`, `events.csv`, `phase-spans.csv`, README, and regenerated the demo shelf manifest/concatenated readable bundle
Proof/caveat: proof: `polylogue.surfaces.temporal_evidence.build_temporal_evidence_window` ran over real conductor log and git events (134 events, 12 buckets, 125 phase spans); caveat: operating-log parsing is still artifact-generator glue, while the committed primitive starts at typed occurrence events
Next demo question: wire the primitive into an operator-facing read/query projection so temporal windows are produced by normal Polylogue surfaces rather than a demo generator

## 2026-06-30T11:06:10+02:00

Trigger: temporal primitive needed a public query/read projection.
Candidate demos: `read --view temporal` live archive output; multi-session projection algebra; temporal phase-latency devloop report.
Selected/improved demo: `/realm/inbox/demos_polylogue/13-temporal-evidence-window`
Artifact action: added `read-view-temporal.json` and `read-view-temporal.md`, updated the README/summary, and regenerated `MANIFEST.readable.json` plus `CONCATENATED_READABLE.md`.
Proof/caveat: proof: `polylogue --plain find 'repo:polylogue' --limit 8 then read --first --view temporal --format json` produced a `TemporalEvidenceWindow` with 8 archive-session events and 3 buckets against the canonical v18 archive; caveat: root read cardinality currently requires `--first` even though the view itself projects the selected query set.
Next demo question: make multi-session projection/cardinality semantics cleaner in the DSL/read algebra, or first isolate generated-surface dirty state so future proof commits can include rendered docs cleanly.

## 2026-06-30T11:13:43+02:00

Trigger: temporal read-view proof exposed that query-set projections were still treated as singleton read actions.
Candidate demos: query-set temporal read output; summary/transcript query-set projections; projection/layout DSL expression examples.
Selected/improved demo: `/realm/inbox/demos_polylogue/13-temporal-evidence-window`
Artifact action: added `read-view-temporal-query-set.json`, updated README/summary, and regenerated `MANIFEST.readable.json` plus `CONCATENATED_READABLE.md`.
Proof/caveat: proof: `polylogue --plain find 'repo:polylogue' --limit 8 then read --view temporal --format json` now works without `--first`, producing 8 archive-session events and 3 buckets from the canonical v18 archive; caveat: richer temporal event families still require upstream query/unit projection work, because this view currently uses session-summary timestamps.
Next demo question: either extend temporal projection beyond session summaries through general query units, or isolate/commit generated-surface dirty state so rendered docs stop obscuring proof diffs.

## 2026-06-30T14:45:00+02:00

Trigger: temporal dogfood refresh exposed stale blank `EVENTS.jsonl` bodies for
recent devloop entries.
Candidate demos: refreshed devloop temporal window; sidecar freshness proof;
agent-history uplift demo.
Selected/improved demo: `/realm/inbox/demos_polylogue/14-devloop-temporal-dogfood`
Artifact action: rebuilt `EVENTS.jsonl` from the filled operating log during
sync, regenerated `devloop-events.temporal-window.json`, refreshed summary,
README, and CSV companion files.
Proof/caveat: proof: current temporal window has 177 events
(`devloop-log=142`, `git=35`) from `structured_jsonl`; `devloop-review` now
checks that the latest structured event matches the latest Markdown log entry.
Caveat: this is still local conductor-source adaptation, not yet a general
archive-captured event-family query.
Next demo question: build one agent-history uplift demo that uses these
temporal/query primitives to improve a concrete continuation or audit task.

## 2026-06-30T14:55:00+02:00

Trigger: need a concrete agent-history uplift artifact rather than more
substrate cleanup.
Candidate demos: continuation handoff packet; current-session temporal read;
bounded current-session chronicle.
Selected/improved demo:
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff`
Artifact action: generated live query, temporal, chronicle, README, and summary
files; added the demo to the top-level shelf README; regenerated manifest and
concatenated readable bundle.
Proof/caveat: proof: current Codex session is the top live query result;
temporal read gives 13 bounded events; chronicle gives 16 edge messages from a
3,819-message session with 3,803 explicit omissions. Caveat: the packet is still
manual composition, not a first-class projection/render spec.
Next demo question: turn this handoff composition into a reusable query +
projection + renderer pattern without naming it recovery.

## 2026-06-30T15:05:00+02:00

Trigger: the agent-history handoff demo was still manual composition even
though the read/context DSL accepted `temporal,chronicle`.
Candidate demos: composed handoff context JSON; renderer/layout handoff packet;
read option-sprawl cleanup proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff`
Artifact action: added `current-session.handoff-context.json` and
`current-session.handoff-context.md`, updated README and summary, and regenerated
`MANIFEST.readable.json` plus
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: the live command `find "session:$session" --limit 1 then
read --view temporal,chronicle --format json` now emits two context-image
segments (`temporal`, `chronicle`) with zero omissions, and the Markdown form is
a 9.6 KB readable packet instead of unsupported view omissions. Caveat: the
layout is still generic temporal+chronicle rendering; more opinionated handoff
layout remains the next renderer/product question.
Next demo question: polish this same query + projection output into a readable
handoff renderer, or clean remaining read flags/options that mix selection,
projection, rendering, and destination.

## 2026-06-30T15:15:00+02:00

Trigger: flat `read --help` still obscured which options belong to which
projection/read view.
Candidate demos: read-view option ownership inventory; projection/render help
cleanup; handoff layout polish.
Selected/improved demo:
`/realm/inbox/demos_polylogue/10-legacy-read-export-flag-audit`
Artifact action: added `read-views.options.txt` and
`read-views.options.json`, updated README and summary, and regenerated
`MANIFEST.readable.json` plus `CONCATENATED_READABLE.md`.
Proof/caveat: proof: live `read --views` now reports per-view scope and owned
CLI options, with JSON carrying `cli_options`, `session_policy`, and
`accepts_query_set`. Caveat: the top-level `read --help` still lists options
flatly; this slice makes ownership discoverable before any more aggressive
presentation split.
Next demo question: decide whether to keep tightening read help presentation or
switch back to a live archive value artifact.

## 2026-06-30T17:20:53+02:00

Trigger: `read --help` still presented projection, delivery, cardinality, and
view-owned controls as one flat option list.
Candidate demos: grouped read help proof; projection/render spec contract;
live archive value artifact after option cleanup.
Selected/improved demo:
`/realm/inbox/demos_polylogue/10-legacy-read-export-flag-audit`
Artifact action: refreshed `polylogue-read-help.txt`, `read-views.options.*`,
README, summary JSON, shelf manifest, and concatenated readable bundle after
grouping top-level read help by option ownership.
Proof/caveat: proof: focused CLI tests pass and live help now has Projection,
Delivery and format, Cardinality and pagination, Context-image projection,
Context and neighbor views, Correlation view, and Other options sections.
Caveat: this clarifies the current surface; it does not yet replace view-
specific option sprawl with a typed projection/render expression.
Next demo question: either implement the typed projection/render spec contract
for one concrete view family, or switch back to a live archive value artifact
that benefits from the clearer read surface.

## 2026-06-30T18:28:00+02:00

Trigger: `has_paste` construct-validity cleanup needed a real archive proof
that showed both the useful predicate and the exact non-claim.
Candidate demos: breaking API/schema naming cleanup proof; paste-evidence
query/render artifact; another temporal/query algebra artifact.
Selected/improved demo:
`/realm/inbox/demos_polylogue/18-paste-evidence-query-render`
Artifact action: generated README, `summary.json`, boundary-state counts, top
sessions, sampled message rows, direct `polylogue find 'has:paste'` output,
and refreshed the shelf README, manifest, and concatenated readable bundle.
Proof/caveat: proof: the active v18 archive has 4 paste-evidence messages
across 3 sessions, all with `paste_boundary_state=projected`; the direct CLI
query returns matching real archive results. Caveat: the stored/internal field
name is still `has_paste`; the artifact is honest because it labels the
operator-facing predicate as paste evidence and explicitly rejects the stronger
"literal pasted payload" claim.
Next demo question: choose a deliberate `has_paste` API/schema naming cleanup
slice, or use the clearer predicate to build a more valuable query/render
artifact.

## 2026-06-30T18:40:00+02:00

Trigger: demo 18 proved the honest predicate but still exposed `has_paste` as a
public reader/API/schema name.
Candidate demos: public contract rename proof; storage-schema rename/reingest
plan; typed projection/render spec.
Selected/improved demo:
`/realm/inbox/demos_polylogue/18-paste-evidence-query-render`
Artifact action: added `messages-after-contract-rename.json`, updated demo 18
README and summary, and refreshed the shelf README, manifest, and concatenated
readable bundle after commit `eb187736a`.
Proof/caveat: proof: live message-view output for `has:paste` now emits 50
`has_paste_evidence` keys and 0 `has_paste` keys; generated CLI schemas and
OpenAPI also use `has_paste_evidence`. Caveat: the SQLite storage column and
internal archive kwargs remain `has_paste`; changing that requires a deliberate
schema-bump/reingest slice, not a public compatibility alias.
Next demo question: either plan/execute the storage physical rename if worth
the schema bump, or move to typed projection/render spec work now that this
predicate is honest at public boundaries.

## 2026-06-30T18:47:00+02:00

Trigger: recent handoff/context demos still had to explain multi-view
composition manually even though the desired model is query + projection +
render.
Candidate demos: projection/render spec introspection; storage physical rename;
another live archive value artifact.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: added `read --spec`, generated
`read-spec-temporal-chronicle.json`, wrote README/summary, updated the shelf
README, and refreshed `MANIFEST.readable.json` plus
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: live `find 'repo:polylogue' then read --view
temporal,chronicle --spec` emits a `QueryProjectionSpec` with selection,
projection families, body policy, and render destination. Caveat: existing read
handlers are not yet driven by the spec; this slice exposes the contract before
deeper routing.
Next demo question: pass the spec into one handler path, migrate one
view-specific option cluster into projection/render fields, or dogfood
`read --spec` in an agent-history handoff artifact.

## 2026-06-30T18:55:00+02:00

Trigger: `read --spec` was useful introspection but handlers did not yet carry
the same algebraic contract.
Candidate demos: handler-threading proof; one renderer consuming the spec;
agent-history handoff packet that includes the spec beside rendered output.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: updated README/summary to record
`ReadViewInvocation.projection_spec`, refreshed the shelf manifest, and
regenerated `CONCATENATED_READABLE.md`.
Proof/caveat: proof: focused CLI test captures a standard `run_read_view`
invocation and verifies selection query/origin/limit, projection families/body
policy, and render destination; static checks and `render all --check` pass.
Caveat: renderers do not all consume the spec yet, and context-image multi-view
execution still has its own path.
Next demo question: make one renderer consume `projection_spec`, migrate one
view-owned option cluster into the typed spec, or dogfood the spec in the
agent-history handoff artifact.

## 2026-06-30T19:06:00+02:00

Trigger: the multi-view context-image path still had no rendered
projection-spec evidence after the standard handler path started carrying it.
Candidate demos: context-image JSON/Markdown with top-level projection spec;
agent-history handoff packet with embedded spec; typed option-cluster migration
proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: added `context-image-with-projection-spec.json`,
`context-image-with-projection-spec.md`, and
`context-image-with-projection-summary.json`; updated README/summary and
refreshed the shelf manifest plus `CONCATENATED_READABLE.md`.
Proof/caveat: proof: live `temporal,chronicle` context-image JSON now includes
top-level `projection_spec` with families
`temporal/sessions/chronicle/messages`; Markdown renders projection families,
body policy, and render destination. Caveat: registered single-view handlers
mostly still carry the spec as metadata rather than consuming it.
Next demo question: dogfood this enriched context-image output in the
agent-history handoff artifact, or migrate one view-owned option cluster into
the typed spec.

## 2026-06-30T19:17:00+02:00

Trigger: enriched context-image output needed to prove value in a real
continuation artifact, not only in demo 19.
Candidate demos: refreshed agent-history handoff packet; option-cluster spec
migration proof; layout/defaults improvement proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff`
Artifact action: rebuilt live query, temporal, chronicle, composed context JSON,
composed Markdown, summary, and README; refreshed the shelf manifest and
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: the target current-devloop session remains the top live
Polylogue repo query result; composed handoff context has two segments, zero
omissions, top-level `projection_spec`, and Markdown projection contract lines.
Caveat: the handoff is still bounded and intentionally omits 4,637 middle
chronicle messages; selector/layout quality is the next product pressure.
Next demo question: improve selector/layout defaults for this normal
context-image projection, or migrate one view-owned option cluster into
`QueryProjectionSpec`.

## 2026-06-30T19:26:00+02:00

Trigger: the refreshed handoff and projection demos showed context-image
`projection_spec.selection.refs` was empty after query resolution, and `--limit
2` still selected five sessions.
Candidate demos: corrected resolved-ref projection spec; selector/layout
defaults; option-cluster spec migration.
Selected/improved demo:
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff` and
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: refreshed composed context JSON/Markdown and summaries after
the code fix; regenerated the shelf manifest and concatenated readable bundle.
Proof/caveat: proof: demo 19 now has `selection.limit=2` and exactly two
resolved `session:` refs; demo 17 has the target current-session ref in the
composed handoff `projection_spec`. Caveat: layout/default quality is still
basic and the bounded chronicle still omits the middle transcript by design.
Next demo question: improve handoff layout/defaults while staying on
context-image, or migrate a view-owned option cluster into `QueryProjectionSpec`.

## 2026-06-30T19:34:00+02:00

Trigger: context-image JSON carried the full selection/projection contract, but
Markdown still omitted the selection query, limit, and resolved refs.
Candidate demos: selection-contract Markdown header; typed option-cluster
migration; current-agent live-value artifact.
Selected/improved demo:
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff` and
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: refreshed context-image JSON/Markdown, summaries, READMEs,
shelf manifest, and `CONCATENATED_READABLE.md` after rendering selection
contract lines.
Proof/caveat: proof: demo 17 Markdown now shows selection query and resolved
target ref; demo 19 Markdown shows selection query `repo:polylogue`, selection
limit `2`, and two resolved session refs. Caveat: this improves header
auditability, not the deeper bounded-chronicle layout or selector ranking.
Next demo question: migrate one view-owned option cluster into
`QueryProjectionSpec`, or produce another live-value artifact over current agent
history using the improved context-image header.

## 2026-06-30T19:32:00+02:00

Trigger: standalone chronicle `read --limit` was an edge-count policy, but
`read --spec` previously described it as `selection.limit`, which made the
typed contract misleading.
Candidate demos: chronicle edge-limit spec proof; next option-cluster
migration; renderer-default proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: added `read-spec-chronicle-edge-limit.json`, updated README
and summary proof facts, refreshed the shelf README, manifest, and
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: live standalone chronicle spec now has selection query
`repo:polylogue`, no selection limit, projection families
`chronicle/sessions/messages`, and `projection.edge_limit=3`. Caveat:
context-image/multi-view still intentionally treats `--limit` as selected
session cardinality, so this is a semantic split rather than a global rename.
Next demo question: migrate the next view-owned option cluster into the typed
projection/render contract, or make one renderer/default consume more of the
spec directly.

## 2026-06-30T19:38:00+02:00

Trigger: standalone `messages` and `raw` `--limit/--offset` describe a body
window, but before this slice `read --spec` had no way to represent that
projection policy separately from session selection.
Candidate demos: message body-window spec proof; neighbor option migration;
context-image selector-field proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: added `read-spec-messages-body-window.json`, updated README
and summary proof facts, refreshed the shelf README, manifest, and
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: live standalone messages spec now has session query
selection, no selection limit, projection families `messages/blocks`,
`projection.body_limit=7`, and `projection.body_offset=2`. Caveat:
context-image/multi-view still intentionally treats `--limit` as selected
session cardinality; body-window policy applies to standalone message/raw
projection.
Next demo question: migrate neighbor window/limit, lift context-image selector
fields into the selection spec, or make renderer layout/defaults consume more
of the typed spec.

## 2026-06-30T19:46:00+02:00

Trigger: context-image project/date/origin/query/max-session selectors were
still carried as context-image-only options rather than visible selection
contract fields.
Candidate demos: context-image selector spec proof; neighbor option migration;
renderer default/layout consumption proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: added `read-spec-context-image-selectors.json`, updated README
and summary proof facts, refreshed the shelf README, manifest, and
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: live context-image spec now records selection query
`projection spec`, origin `codex-session`, since/until `2026-06-30`, project
path `/realm/project/polylogue`, project repo `Sinity/polylogue`, and
selection limit `3`. Caveat: this makes selector contracts visible and typed;
it does not yet redesign ranking/layout defaults.
Next demo question: migrate neighbor window/limit into typed policy, or make
renderer layout/defaults consume more of the spec directly.

## 2026-06-30T19:54:00+02:00

Trigger: standalone neighbor `--limit` and `--window-hours` were still
view-local options even though they describe neighbor projection policy.
Candidate demos: neighbor policy spec proof; renderer default/layout
consumption proof; updated agent-history handoff using richer spec fields.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: added `read-spec-neighbor-policy.json`, updated README and
summary proof facts, refreshed the shelf README, manifest, and
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: live neighbors spec now records session query selection,
no selection limit, projection families `neighbors/sessions`,
`projection.neighbor_limit=4`, and `projection.neighbor_window_hours=12`.
Caveat: this completes another option-policy migration; it does not yet make
renderer defaults or layouts derive from the spec.
Next demo question: make renderer layout/defaults consume more of the typed
spec, preferably in a way visible in demo 17 or demo 19 rather than adding
another field.

## 2026-06-30T19:59:00+02:00

Trigger: context-image Markdown rendered selection/families/body/render but
still omitted the richer projection policy fields now present in
`ProjectionSpec`.
Candidate demos: refreshed demo 19 context-image Markdown; refreshed demo 17
handoff Markdown; renderer/layout default proof.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec` and
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff`
Artifact action: regenerated context-image Markdown artifacts, updated
summaries/READMEs, and refreshed the shelf manifest plus
`CONCATENATED_READABLE.md`.
Proof/caveat: proof: demo 19 Markdown now renders `Projection max tokens:
1000` and `Projection redact paths: true`; demo 17 handoff Markdown renders
`Projection redact paths: true`. Caveat: this is header contract rendering,
not a deeper layout redesign.
Next demo question: dogfood the richer handoff Markdown for current continuation
quality, or move another renderer/layout default under `QueryProjectionSpec`.

## 2026-06-30T20:08:00+02:00

Trigger: context-image and multi-view read specs now carry explicit render
layout instead of inheriting the inert `standard` default.
Candidate demos: demo 19 projection/render spec; demo 17 refreshed handoff
packet.
Selected/improved demo:
`/realm/inbox/demos_polylogue/19-projection-render-spec`
Artifact action: regenerated `read-spec-temporal-chronicle.json`,
`read-spec-context-image-selectors.json`,
`context-image-with-projection-spec.json/md`,
`context-image-with-projection-summary.json`, `summary.json`, root shelf
README, `MANIFEST.readable.json`, and `CONCATENATED_READABLE.md`.
Proof/caveat: proof: live multi-view spec and context-image selector spec now
contain `render.layout=context-image`, and Markdown renders
`- Render layout: context-image`. Caveat: ordinary single-view handlers
intentionally remain `layout=standard` and still do not all consume every
projection-spec field.
Next demo question: dogfood refreshed demo 17/19 Markdown for continuation
quality, or audit the next hidden renderer default that should become
render/projection policy.

## 2026-06-30T20:18:00+02:00

Trigger: refreshed projection/render layout needed to be proven on the actual
agent-history handoff packet, not only demo 19's spec artifact.
Candidate demos: demo 17 current-session handoff; demo 19 projection/render
spec; reusable summary generator.
Selected/improved demo:
`/realm/inbox/demos_polylogue/17-agent-history-uplift-handoff`
Artifact action: regenerated live query, temporal, chronicle, composed
context-image JSON/Markdown, summary JSON, README, root shelf README,
`MANIFEST.readable.json`, and `CONCATENATED_READABLE.md`.
Proof/caveat: proof: current handoff Markdown renders selection refs,
projection policy, caveats, and `Render layout: context-image`; summary now
records `selection_contract_lines=true`, `render_layout_line=true`, two
segments, zero omissions, and a 773-token estimate. Caveat: the summary builder
is still local artifact glue, so reusable artifact-summary generation remains a
candidate substrate slice.
Next demo question: should summary/manifest generation become a reusable
devtools/demo primitive instead of repeated local Python snippets?

## 2026-06-30T20:24:00+02:00

Trigger: demo 17/19 refreshes repeatedly regenerated the shelf manifest and
concatenated readable bundle through local snippets.
Candidate demos: reusable shelf refresh for `/realm/inbox/demos_polylogue`;
same command against `/realm/inbox/demos_sinex`; per-demo summary generation.
Selected/improved demo:
`/realm/inbox/demos_polylogue`
Artifact action: added `devtools workspace demo-shelf`, refreshed
`MANIFEST.readable.json` and `CONCATENATED_READABLE.md` through that command,
and made generated aggregate files explicitly excluded from the manifest.
Proof/caveat: proof: `devtools workspace demo-shelf --check --json` reports
the real shelf current with 155 files and 137 readable artifacts. Caveat: this
solves shelf aggregation only; individual demo `summary.json` generation is
still local glue and remains a candidate next slice.
Next demo question: make per-demo summary generation similarly reusable, or use
the now-stable shelf command to compare Polylogue/Sinex readable demo packets.

## 2026-06-30T20:31:00+02:00

Trigger: summary generation reuse looked tempting, but the existing summaries
are heterogeneous and should not be coerced into one false schema.
Candidate demos: summary coverage index; demo 17 summary repair; Sinex summary
claim/non-claim pass.
Selected/improved demo:
`/realm/inbox/demos_polylogue/SUMMARY_INDEX.json` and
`/realm/inbox/demos_sinex/SUMMARY_INDEX.json`
Artifact action: extended `devtools workspace demo-shelf` to emit
`SUMMARY_INDEX.json` with per-summary common fields and coverage lists for
claim, non-claim, proof fields, and caveat fields.
Proof/caveat: proof: Polylogue shelf indexes 14 summaries and Sinex indexes 10;
the coverage lists immediately identify older/foreign summaries that lack
explicit claim/non-claim/proof/caveat fields. Caveat: this is a quality index,
not a generator that certifies the underlying summaries are true.
Next demo question: repair the highest-value missing summary fields, or use the
index as a dashboard while returning to read/projection cleanup.

## 2026-06-30T21:04:14+02:00

Trigger: timestamp policy moved into RenderSpec
Candidate demos: demo 19 projection/render spec; current-session handoff; no new shelf needed
Selected/improved demo: /realm/inbox/demos_polylogue/19-projection-render-spec
Artifact action: patched read-spec temporal/context-image JSON, context-image projection JSON/Markdown, nested summary, root summary, and README to record render.timestamps=include-available
Proof/caveat: proof: live read --spec for temporal,chronicle and context-image reports render.timestamps=include-available; context-image Markdown renders '- Render timestamps: include-available'; caveat: this is a render policy to include available source timestamps, not a guarantee every selected row is timestamped
Next demo question: next slice should either route another export-quality knob through QueryProjectionSpec or dogfood temporal analysis on the current devloop

## 2026-06-30T21:08:49+02:00

Trigger: current devloop temporal dogfood refresh
Candidate demos: demo 14 temporal dogfood; no new report silo; possible follow-up temporal-query algebra
Selected/improved demo: /realm/inbox/demos_polylogue/14-devloop-temporal-dogfood
Artifact action: refreshed devloop-events.temporal-window.json, added temporal-read-profile.current.json, and updated README/summary counts and process feedback
Proof/caveat: proof: devtools workspace temporal-devloop now reports 270 events from structured EVENTS.jsonl plus git (195 devloop-log, 75 git); temporal-read-profile current reports 39 archive events in 2.33s with project_actions still slowest at 1.37s; caveat: windows remain capped/open and this is evidence for prioritization, not proof of complete temporal coverage
Next demo question: next slice should use these temporal facts to target either action projection cost, richer devloop phase analysis, or another export/projection algebra gap

## 2026-06-30T21:17:05+02:00

Trigger: temporal activity bands added to shared TemporalEvidenceWindow
Candidate demos: demo 14 temporal dogfood; no new report shelf; possible README-only update rejected
Selected/improved demo: /realm/inbox/demos_polylogue/14-devloop-temporal-dogfood
Artifact action: regenerated devloop-events.temporal-window.json and updated README/summary with activity_bands top dense hours
Proof/caveat: proof: live temporal-devloop reports 272 events and activity_bands in shared temporal_window JSON; top dense bands include 18:00Z with 18 devloop checkpoints and 8 git commits; caveat: dense event count is cadence signal, not semantic progress score
Next demo question: next slice should use activity_bands to compare cadence against proof/commit quality, or move back to projection/render algebra

## 2026-06-30T21:57:14+02:00

Trigger: chatlog export workaround productized
Candidate demos: product-native dialogue export; prune old custom operator-readable variants; richer layout/render DSL
Selected/improved demo: product-native dialogue export in .agent/demos/chatlog-exports/current/*/product-read
Artifact action: updated regenerate.sh to emit dialogue.md/dialogue.json for both large Codex devloop sessions and refreshed the product-read outputs
Proof/caveat: proof: read --view dialogue live samples include user/assistant timestamps and omit tool outputs; dialogue JSON validates; caveat: older custom markdown variants remain as comparison artifacts until a later consolidation pass
Next demo question: Should the next demo slice consolidate the chatlog-export shelf around product-native variants, or first improve projection/render DSL for layout-level export control?

## 2026-06-30T22:01:32+02:00

Trigger: chatlog export current shelf consolidated
Candidate demos: compact product-native current shelf; compact dialogue JSON; declarative export package layout
Selected/improved demo: current chatlog-export shelf with clearly separated product-read and full-chatlog variants
Artifact action: moved legacy custom markdown and raw JSONL out of current, added current README, regenerated product outputs, and later corrected the shelf so full-chatlog exports are generated unconditionally by the read package
Proof/caveat: proof: current product JSON validates and later full-chatlog artifacts return all stored messages; caveat: this earlier checkpoint was superseded because full transcript generation belongs in the normal package for requested whole-session exports
Next demo question: Should the next product slice make compact dialogue JSON/export packages first, or continue broader projection/render DSL cleanup?

## 2026-06-30T22:06:47+02:00

Trigger: compact dialogue machine payload
Candidate demos: compact dialogue JSON/YAML; declarative export package layout; broader projection/render DSL
Selected/improved demo: read --view dialogue compact machine payload plus refreshed chatlog-export product artifacts
Artifact action: dialogue JSON/YAML now emit session identity plus dialogue turns only; regenerated product-read dialogue.json for both large Codex sessions
Proof/caveat: proof: focused CLI tests passed 59; live dialogue JSON has expected compact keys and dropped to 3.3M/3.6M; caveat: export package layout is still script-defined rather than declarative product configuration
Next demo question: Should the next slice introduce a declarative export package spec, or keep shaving individual projection/render gaps surfaced by demos?

## 2026-06-30T22:16:11+02:00

Trigger: declarative read package generation
Candidate demos: devtools read-package; product read-plan surface; more demo package specs
Selected/improved demo: chatlog-export read-package.json executed by devtools workspace read-package
Artifact action: added generic read-package command and rewired chatlog export regenerate.sh to use declarative view/format/path/spec artifacts
Proof/caveat: proof: focused tests passed 3; render all --check passed; live package --json summary parsed and includes six artifact byte counts; caveat: command is still in devtools workspace, not a first-class product read-plan verb
Next demo question: Should the next slice promote read-package planning into product CLI/API, or use the spec to consolidate additional demos first?

## 2026-06-30T23:07:58+02:00

Trigger: agent affordance usage analysis
Candidate demos: Serena/codebase-memory utility; tool-family failure rates; recency-normalized MCP adoption; product gap from slow raw actions scans
Selected/improved demo: .agent/demos/agent-affordance-usage
Artifact action: added README reasoning report plus focused-tool-counts, by-origin, recent-7d, samples, and archive-origin CSV evidence; refreshed current demo shelf manifest
Proof/caveat: proof: current archive v18 actions data; recent window shows Serena find_symbol 10 actions across 3 sessions with zero structured failures; caveat: raw tool names are not normalized and direct SQL was slow/lock-prone during daemon writes
Next demo question: Build reusable affordance-usage projection/query command so this demo regenerates without direct SQL and with family-normalized tool identities

## 2026-07-01T01:28:57+02:00

Trigger: dialogue projection output bounded
Candidate demos: CLI audit full dialogue payload; bounded dialogue projection; chatlog export product-read
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: Added read_dialogue_bounded_json output and README finding showing --max-tokens 120 reduces one large devloop dialogue JSON to 2.3 KB with omission counts
Proof/caveat: Proof is active archive v18 live command; caveat: token count is whitespace-estimated, not provider tokenizer exact
Next demo question: Next choose facets/read-view-discovery latency or promote bounded dialogue into chatlog export package defaults.

## 2026-07-01T01:35:15+02:00

Trigger: bounded dialogue projection promoted into read-package
Candidate demos: chatlog export package defaults; CLI surface audit bounded dialogue proof; future composition/layout DSL
Selected/improved demo: chatlog export current product-read dialogue.json
Artifact action: Added max_tokens to read-package artifacts, regenerated both current chatlog export packets, and refreshed demo indexes.
Proof/caveat: dialogue.json is now 1.8-2.3KB with explicit projection/omission metadata; dialogue.md remains full concise prose. The package still renders full messages.json, so future demo defaults may need additional projection controls for structural payloads.
Next demo question: Should read-package grow a general projection/layout policy object rather than individual fields as more read options become demo defaults?

## 2026-07-01T01:46:02+02:00

Trigger: read-package projection object replaced flat max_tokens
Candidate demos: chatlog export package; repeatable package docs; future render/layout policy
Selected/improved demo: read-package projection policy plus chatlog export current package
Artifact action: Replaced flat max_tokens with projection.max_tokens, added projection limit keys, regenerated current chatlog export package, and documented repeatable local packages in docs/export.md.
Proof/caveat: Proof: focused tests 12 passed, render all --check passed, live regenerated dialogue JSON stayed 1.8-2.3KB with omission metadata. Caveat: render/layout policy is not yet represented in read-package; this slice covered projection policy only.
Next demo question: Next choose whether render policy belongs in read-package now, or return to agent affordance analytics as a product query/report projection.

## 2026-07-01T01:52:31+02:00

Trigger: affordance usage normalized tool identities
Candidate demos: agent affordance usage demo; codebase-memory/Serena utility analysis; future indexed tool-use projection
Selected/improved demo: .agent/demos/agent-affordance-usage
Artifact action: Normalized tool identities in devtools workspace affordance-usage, refreshed the default report and codebase-detail-30d companion report, and updated the local analysis note.
Proof/caveat: Proof: focused tests 4 passed and live active-archive reports now carry normalized tool identities plus raw_tool_names. Caveat: broad all-time scans remain too expensive; indexed/materialized tool-use projection is still future substrate work.
Next demo question: Next choose indexed/materialized tool-use projection, or switch back to read/package render policy if export demos need it first.

## 2026-07-01T02:02:24+02:00

Trigger: affordance usage canonical actions projection
Candidate demos: agent affordance usage; canonical action substrate; future indexed tool-use projection
Selected/improved demo: .agent/demos/agent-affordance-usage
Artifact action: Switched reusable affordance report to canonical actions view, refreshed reports and analysis, and recorded warm/cold timing evidence.
Proof/caveat: Proof: focused tests 4 passed; static checks passed; render all --check passed; live archive reports regenerated. Caveat: one cold/contended default read still took 55.098s, so materialized/indexed projection remains a future option if repeated.
Next demo question: Next shift to meta/process devloop improvement per operator request, especially canonical primitive convergence and conductor packet cleanliness.

## 2026-07-01T02:18:31+02:00

Trigger: read-package render policy
Candidate demos: chatlog export package; projection/render composition; future layout/timestamp render knobs
Selected/improved demo: .agent/demos/chatlog-exports
Artifact action: Added render.fields to read-package specs, regenerated current chatlog export product-read artifacts, and documented that spec.json is narrowed to selection/projection/render through render policy.
Proof/caveat: Proof: focused read-package tests 14 passed; static checks passed; live package dry-run shows --fields selection,projection,render; regenerated both session packages; validated both spec.json files contain only selection/projection/render; render all --check passed. Caveat: render policy currently supports fields only.
Next demo question: Next choose whether to add more render knobs when product CLI exposes them, or move to CLI surface latency/read-view discovery from current demo pressure.

## 2026-07-01T02:29:28+02:00

Trigger: CLI surface audit generator
Candidate demos: current CLI audit shelf; bounded dialogue proof; future read-view discovery/latency audit
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: Regenerated with devtools workspace cli-surface-audit; default command set has 11 outputs, all exit 0, and stale unbounded read_dialogue_json.stdout is removed.
Proof/caveat: Proof: active archive v18, 13,119 sessions, 3,960,227 messages; include_unbounded_dialogue=false; command-matrix.json records timings/bytes. Caveat: this is representative CLI smoke, not exhaustive CLI certification.
Next demo question: Next meta slice should make process scaffolding enforce current curated demos and remove duplication/noisy packet patterns.

## 2026-07-01T02:49:31+02:00

Trigger: CLI startup lazy config
Candidate demos: CLI surface audit; import-time profile; future read-view/query latency
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: Made config a lazy root command and regenerated CLI audit current shelf; root_help timing is now 585 ms in command-matrix.json versus prior 1000 ms.
Proof/caveat: Proof: importtime no longer pulls config->paths->storage/archive layout for root help; config/json focused tests 29 passed; root help snapshots 4 passed; devtools verify --quick passed. Caveat: read_dialogue_bounded_json and facets remain multi-second paths.
Next demo question: Next product slice should profile query/read-view startup rather than root command registration.

## 2026-07-01T03:12:48+02:00

Trigger: CLI read-view registry latency
Candidate demos: CLI surface audit current demo; read-view registry metadata path; remaining facets/dialogue latency
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: Regenerated after lazy read-view runtime/AppEnv/projection/payload imports; read_views_json is 575.844ms and importtime avoids polylogue.api/dateparser/services/ui/storage/payload/projection on the metadata-only path.
Proof/caveat: Proof: devtools workspace cli-surface-audit on active archive v18; devtools verify --quick passed in commit 600016ff8. Caveat: facets_json and read_dialogue_bounded_json remain multi-second archive-backed paths.
Next demo question: Choose whether to optimize remaining archive-backed query paths or promote affordance usage into a reusable query/report projection.

## 2026-07-01T03:39:34+02:00

Trigger: Tool affordance usage projection
Candidate demos: analyze tools now exposes a fast call-count projection shape and filters; exact ToolUsageInsight accepts origin/provider filters and empty archives
Selected/improved demo: polylogue analyze tools --format json --limit 5; polylogue analyze tools --mcp-server serena --format json --limit 5
Artifact action: Proof: ruff/mypy touched files; diagnostics tools tests 4 passed; tool_usage focused tests 5 passed; active archive query-plan evidence shows exact actions aggregation still scans ~1.56M tool_use blocks with temp B-trees.
Proof/caveat: Caveat: live timing proof was not claimed because borg backup and lynchpin materialization were active; exact detail/coverage remains a materialized/indexed projection target, not solved by this slice.
Next demo question: Next: add schema-backed/materialized tool usage rollups or indexed generated columns so exact Serena/codebase-memory affordance analytics are fast and non-siloed.

## 2026-07-01T04:04:17+02:00

Trigger: Tool outcome query contract
Candidate demos: observed-event tool outcome basis; stable analyze tools JSON; affordance analytics demo refresh after pressure clears
Selected/improved demo: docs/schemas/cli-output/tool-counts.schema.json
Artifact action: Published ToolCountPayload schema, regenerated CLI output schema README and CLI reference, and wired analyze tools JSON through the typed payload model.
Proof/caveat: Proof: ruff/mypy touched files passed; diagnostics tools tests 7 passed; CLI output schema focused tests 18 passed; render all --check passed after regenerating CLI reference/pages. Caveat: no live latency claim while devloop-status reports host pressure.
Next demo question: Next: when pressure clears, regenerate agent-affordance demo using the observed-events basis or design the schema-backed exact tool rollup.

## 2026-07-01T04:11:43+02:00

Trigger: General query tool outcome aggregation
Candidate demos: observed-event DSL predicates/grouping; analyze tools schema; future live affordance demo
Selected/improved demo: observed-events where kind:tool_finished AND handler:mcp | group by status | count
Artifact action: Added observed-event payload fields tool/handler/status, aggregate group support, SQL count lowering, and OpenAPI query example regeneration.
Proof/caveat: Proof: ruff/mypy touched files passed; query expression aggregate tests 3 passed; query metadata tests 2 passed; render all --check passed after regenerating OpenAPI. Caveat: this proves substrate/query semantics on fixtures, not live latency under current host pressure.
Next demo question: Next: when pressure clears, use this DSL expression against active archive to refresh agent-affordance demos; otherwise design indexed rollup if repeated live use remains slow.

## 2026-07-01T06:11:07+02:00

Trigger: affected_count made raw debt structurally aggregable
Candidate demos: raw debt summary, replayable repair queue, parsed-without-session classifier demos
Selected/improved demo: archive-debt-summary
Artifact action: created .agent/demos/archive-debt-summary with README, regenerate.sh, raw JSON, CSV summary, and ANALYSIS.md; refreshed demo indexes
Proof/caveat: proves current active archive has 378 affected raw debt artifacts, 78 replayable acquired-unparsed rows, and 276 open/info non-replay rows; live performance proof remains blocked
Next demo question: decide whether to drain the 78 replayable rows when host pressure clears or improve parsed-without-session classification next

## 2026-07-01T08:16:12+02:00

Trigger: raw materialization scoped repair proof
Candidate demos: archive-debt-summary current shelf
Selected/improved demo: archive-debt-summary
Artifact action: regenerated archive-debt-summary after scoped raw repairs
Proof/caveat: raw materialization debt now 374 total / 74 actionable; four Codex raw artifacts materialized; scoped maintenance still times out after writes
Next demo question: investigate parse_from_raw/ingest-batch finalization hang after successful raw write

## 2026-07-01T11:10:13+02:00

Trigger: product analyze tools filtered query proof
Candidate demos: refresh agent-affordance demo through product CLI, not devtools-only reports
Selected/improved demo: agent-affordance-usage/product-analyze-tools
Artifact action: added Serena and codebase-memory observed-event JSON payloads plus current README/analysis interpretation
Proof/caveat: Serena returns two observed-event rows under active borg pressure; codebase-memory returns zero on observed-events basis, with command/detail evidence preserved separately
Next demo question: next affordance slice should decide whether codebase-memory command/detail evidence belongs in a product projection rather than devtools detail-pattern reports

## 2026-07-01T11:33:16+02:00

Trigger: agent affordance evidence basis harmonization
Candidate demos: agent-affordance-usage product analyze tools; action-evidence basis; codebase-memory utility eval
Selected/improved demo: agent-affordance-usage/product-analyze-tools
Artifact action: Added product action-evidence proof for codebase-memory command-detail usage using --basis actions --days 7 --detail-pattern codebase-memory
Proof/caveat: Proof: live active archive v19 command returned one claude-code codebase-memory/command-detail row; caveat: 30-day detail scan timed out under borg, so current demo uses 7-day bounded scope
Next demo question: Next decide whether indexed/materialized action-evidence rollups are needed for 30d/all-time utility evaluation

## 2026-07-01T11:50:49+02:00

Trigger: action-evidence 30-day product proof
Candidate demos: 7-day action proof; 30-day action proof; all-time action proof; future materialized rollup
Selected/improved demo: 30-day action-evidence product payload under .agent/demos/agent-affordance-usage/product-analyze-tools
Artifact action: replaced stale 7-day payload with codebase-memory-action-evidence-30d.json and refreshed demo indexes
Proof/caveat: proof: live active-archive command completed under timeout with archive root /home/sinity/.local/share/polylogue, v19, 13,103 sessions; caveat: filtered proof, not broad all-tool SLO
Next demo question: next demo question: evaluate whether all-time/high-cardinality action-evidence questions need a materialized rollup

## 2026-07-01T12:59:45+02:00

Trigger: agent affordance comparison productization
Candidate demos: serena/codebase-memory comparison; product analyze-tools proof; demo shelf refresh
Selected/improved demo: agent-affordance-usage/product-analyze-tools
Artifact action: Added live --compare-family serena/codebase-memory JSON artifacts and refreshed demo shelf manifests
Proof/caveat: Proof: focused CLI/schema tests passed, mypy passed, live compare-family commands succeeded on active v19 archive; caveat: borg pressure blocks broad SLO claims
Next demo question: Next compare-family improvement should be an indexed/materialized action-evidence rollup only if all-time/high-cardinality utility evaluation becomes frequent

## 2026-07-01T13:13:49+02:00

Trigger: read projection/render ownership surface
Candidate demos: read-view option inventory; projection/render contract inventory; CLI surface audit refresh
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: regenerated CLI surface audit after read --views JSON gained projection_contract summaries
Proof/caveat: proof: focused CLI/profile/projection tests pass and active v19 archive demo output shows context-image/raw projection_contract fields; caveat: this exposes the current contract, it does not yet replace every flag with a typed projection expression
Next demo question: next decide whether to add render layout controls to the spec surface or switch to archive-debt classification once borg clears

## 2026-07-01T13:18:07+02:00

Trigger: read render layout spec controls
Candidate demos: render layout explicit spec control; CLI surface audit refresh; later render-layout execution semantics
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: regenerated CLI surface audit after read --help gained --render-layout and read --spec accepted explicit layout override
Proof/caveat: proof: focused read help/spec tests, mypy, ruff, live read --spec, and refreshed active v19 CLI audit pass; caveat: --render-layout currently makes the composed spec explicit and does not force every renderer to implement multiple layouts
Next demo question: next choose between archive-debt classification if borg clears, or another projection/render contract slice if pressure remains

## 2026-07-01T13:22:59+02:00

Trigger: read render timestamp spec controls
Candidate demos: read projection contract demos; CLI surface audit; temporal read spec proof
Selected/improved demo: CLI surface audit current shelf plus /tmp/polylogue-timestamp-policy-spec.json live proof
Artifact action: refreshed .agent/demos/cli-surface-audit/current after adding --timestamps
Proof/caveat: Proof covers spec composition and CLI contract; it does not claim broad archive convergence while raw materialization debt remains
Next demo question: Should read rendering grow a declarative composition/layout DSL next, or first collapse more legacy flags into projection spec controls?

## 2026-07-01T13:32:08+02:00

Trigger: read render expression shorthand
Candidate demos: projection/render expression proof; CLI surface audit; temporal+chronicle read spec
Selected/improved demo: CLI surface audit current shelf plus /tmp/polylogue-render-expression-spec.json live proof
Artifact action: refreshed .agent/demos/cli-surface-audit/current after adding --render expression
Proof/caveat: Proof covers CLI parsing and RenderSpec composition on active v19 archive; raw materialization debt remains so this is not a full convergence claim
Next demo question: Next: collapse more read flags into projection/render spec only where the typed contract can represent them cleanly

## 2026-07-01T13:44:40+02:00

Trigger: archive debt summary refresh
Candidate demos: archive debt summary demo; raw debt status/review; actionable replay item caveat
Selected/improved demo: current archive-debt-summary demo under .agent/demos
Artifact action: regenerated archive-debt-summary and demo shelf manifests against active v19 archive
Proof/caveat: Proof covers current structured debt classification: 277 affected, 1 actionable/replayable, 276 classified open; it does not run the 1.5 GiB parse-pending repair under borg pressure
Next demo question: Next: either add a product-level archive debt summary/read projection or wait for pressure to clear before targeted raw replay

## 2026-07-01T13:54:26+02:00

Trigger: archive debt text evidence refs
Candidate demos: archive debt text proof; archive-debt-summary demo refresh
Selected/improved demo: archive-debt-summary
Artifact action: Refreshed archive-debt-summary after text debt output started rendering affected_count, sampled evidence refs, and row caveats
Proof/caveat: Focused CLI test/static checks plus active --only-actionable raw-materialization output show raw/file/blob evidence refs in normal text
Next demo question: Potential next slice: run targeted raw replay when safe, or add a compact debt summary command/view if repeated operator use wants less row detail

## 2026-07-01T14:15:34+02:00

Trigger: targeted raw replay blocked oversized row
Candidate demos: archive-debt-summary; raw replay guard
Selected/improved demo: archive-debt-summary
Artifact action: Refreshed archive-debt-summary after oversized raw replay became blocked instead of actionable
Proof/caveat: Live maintenance command now fails fast with RepairReportedFailure; debt list --only-actionable returns zero raw-materialization rows; full debt list shows blocked=1 affected_blocked=1
Next demo question: Next: implement streaming Claude Code grouped-record repair if this raw row must converge, or build a compact blocked-debt summary view

## 2026-07-01T14:19:32+02:00

Trigger: oversized raw replay guard
Candidate demos: archive-debt-summary; raw materialization debt list; guarded maintenance replay
Selected/improved demo: archive-debt-summary
Artifact action: regenerated .agent/demos/archive-debt-summary via regenerate.sh and devloop-refresh-demos
Proof/caveat: Proof: live active archive reports actionable=0 blocked=1 and guarded replay returns failed/blocked immediately; caveat: underlying 1.5 GiB row still needs a streaming provider-specific repair path
Next demo question: Should the next slice implement streaming Claude Code raw-row repair or improve blocked-debt UX first?

## 2026-07-01T14:56:45+02:00

Trigger: safe oversized Claude Code raw materialization actuator
Candidate demos: archive-debt-summary; raw materialization live proof; stream-safe repair actuator
Selected/improved demo: archive-debt-summary refreshed after live targeted replay
Artifact action: regenerated .agent/demos/archive-debt-summary from active archive
Proof/caveat: active archive v19 now has replayable_acquired_unparsed=0 and raw debt actionable/blocked counts are zero; remaining 276 rows are informational/open classifications, not full convergence
Next demo question: show parsed-non-session/materialized-alias classification semantics next

## 2026-07-01T15:09:23+02:00

Trigger: raw materialization classified debt semantics
Candidate demos: archive-debt-summary; ops status raw component; devloop-status raw materialization line
Selected/improved demo: archive-debt-summary refreshed with classified non-actionable raw gap semantics
Artifact action: regenerated .agent/demos/archive-debt-summary from active archive
Proof/caveat: direct CLI fallback and devloop-status report degraded/non-actionable instead of stale/pending; normal daemon-backed status needs daemon restart after commit
Next demo question: decide whether materialized-alias and parsed-non-session-artifact should become ready-classified debt or disappear from broad debt totals

## 2026-07-01T15:26:12+02:00

Trigger: classified raw gap semantics
Candidate demos: archive-debt-summary; ops status raw component; devloop-status convergence fields
Selected/improved demo: archive-debt-summary refreshed with classified raw gaps as ready/non-debt
Artifact action: regenerated archive-debt-summary and demo shelf after ArchiveDebtStatus gained classified
Proof/caveat: focused tests/static checks pass; active v19 archive reports join_gaps=276, debt=0, classified=276, open=0; daemon-backed status needs restart after commit
Next demo question: next decide whether classified rows should be hidden from default debt list text or kept visible as info evidence

## 2026-07-01T15:39:48+02:00

Trigger: archive debt classified visibility
Candidate demos: classified debt evidence filter; actionable zero-debt filter; archive-debt-summary
Selected/improved demo: archive-debt CLI/status filter
Artifact action: added repeatable --status filter and rendered classified/affected_classified totals in text output
Proof/caveat: focused CLI/operation tests, mypy, ruff, live --status classified and --status actionable active-v19 proofs pass
Next demo question: next decide whether default debt list should grow an unresolved-only alias or whether status filters are sufficient

## 2026-07-01T15:43:09+02:00

Trigger: archive debt unresolved default
Candidate demos: default unresolved debt view; classified evidence filter; archive-debt-summary
Selected/improved demo: archive-debt CLI default and --status classified proof
Artifact action: changed ops debt list default to open/actionable/blocked while preserving explicit --status classified
Proof/caveat: focused CLI/operation tests, mypy, ruff, live default raw-materialization zero rows, and live classified filter returns 3 rows/276 affected
Next demo question: next choose whether to refresh archive-debt-summary demo artifacts or move to projection/render DSL work

## 2026-07-01T15:50:53+02:00

Trigger: archive debt demo refresh
Candidate demos: archive-debt-summary unresolved/classified payload split; daemon status proof
Selected/improved demo: archive-debt-summary
Artifact action: regenerated ignored current demo shelf with raw-materialization-unresolved.json and raw-materialization-classified.json
Proof/caveat: active v19 archive proof: default unresolved rows=0; classified rows=3/affected=276; daemon raw component ready
Next demo question: next product slice can return to projection/render DSL or agent-affordance rollup depending on pressure and fresh evidence

## 2026-07-01T16:01:50+02:00

Trigger: read projection spec controls
Candidate demos: read --projection expression; CLI surface audit current proof
Selected/improved demo: .agent/demos/cli-surface-audit/current
Artifact action: regenerated current CLI surface audit so read_temporal_spec_json exercises both --render and --projection expressions
Proof/caveat: focused tests/static checks/live active-v19 CLI spec proof pass; render all pending
Next demo question: next choose whether more read flags fold into projection expression or shift to agent-affordance analytics

## 2026-07-01T16:20:34+02:00

Trigger: daemon FTS readiness truthfulness
Candidate demos: healthz ready; FTS debt zero; ops status envelope truthful; FTS search usable
Selected/improved demo: .agent/task-history/fts-readiness-repair-20260701
Artifact action: repaired canonical active archive FTS via existing maintenance target and captured before/after evidence
Proof/caveat: active v19 archive: messages_fts 5,066,827/5,066,827, /healthz/ready ready, ops debt --kind fts zero, focused status regression passed
Next demo question: next choose between insight/version mismatch repair, agent-affordance analytics, or further read projection flag folding based on fresh evidence

## 2026-07-01T16:55:51+02:00

Trigger: session insight repair truthfulness
Candidate demos: session_insights maintenance now distinguishes targeted session rebuild from aggregate/thread-marker refresh
Selected/improved demo: .agent/task-history/session-insight-repair-20260701
Artifact action: fixed misleading dry-run/full-rebuild fallback and safely cleared aggregate debt on the active v19 archive
Proof/caveat: commit c45334268; 56 affected storage tests passed; devloop daemon ready; FTS 5,066,827/5,066,827; tag rollups ready; stale thread rows 0
Next demo question: remaining 88 missing-profile sessions total 797,306 messages; next slice should make large-session profile materialization memory-safe/degraded before broad repair

## 2026-07-01T17:45:27+02:00

Trigger: large-session profile materialization
Candidate demos: bounded large-session profiles; thread parent repair; active archive insight readiness
Selected/improved demo: .agent/task-history/large-session-profile-materialization-20260701
Artifact action: recorded live repair artifacts, final dry-run, final insight status, daemon restart
Proof/caveat: active archive /home/sinity/.local/share/polylogue index v19: session_insights dry-run affected_rows=0; profiles 13104/13104; threads 5235/5235 ready; daemon /healthz/ready ok; caveat: session_profiles aggregate verdict remains degraded because degraded reasons are explicit
Next demo question: choose next demo/product slice from live query/export affordance analytics or CLI surface audit

## 2026-07-01T19:52:07+02:00

Trigger: agent affordance analytics proof closure
Candidate demos: product analyze-tools proof; source-derived observed-events proof; schema/materialization redesign candidate; FTS debt follow-up
Selected/improved demo: .agent/demos/agent-affordance-usage/product-analyze-tools
Artifact action: refreshed product-analyze-tools JSON artifacts including serena-observed-events-30d.json and updated README interpretation
Proof/caveat: Active archive /home/sinity/.local/share/polylogue index v20; session_observed_events table remains zero, but analyze tools --basis observed-events --mcp-server serena --days 30 returns source-derived paired tool-result events with status unknown when result rows lack structured outcome fields
Next demo question: Next demo question: remove or narrow persisted run/observed/context tables by proving which query units can be lowered from canonical blocks/session_events/topology on demand

## 2026-07-01T20:16:04+02:00

Trigger: terminal observed-events from blocks
Candidate demos: query DSL no longer depends on empty session_observed_events for selective tool_finished rows
Selected/improved demo: .agent/demos/agent-affordance-usage/product-terminal-observed-events
Artifact action: active archive /home/sinity/.local/share/polylogue, index schema v20; row proof ~1.9s, aggregate proof ~3.7s
Proof/caveat: status remains unknown when tool_result lacks structured success/error
Next demo question: cluster PR integration and continue run/context projection audit

## 2026-07-01T20:25:33+02:00

Trigger: source-derived run/context query units
Candidate demos: runs and context-snapshots no longer require empty run-projection tables for main/session-start rows
Selected/improved demo: .agent/demos/query-runtime-projections/source-derived-run-context
Artifact action: active archive /home/sinity/.local/share/polylogue, index schema v20; session_runs=0, session_context_snapshots=0, session_observed_events=0; run proof ~2.4s, context proof ~5.2s
Proof/caveat: source-derived rows are main run/session_start only; subagent runs and richer cwd/context still need deeper evidence
Next demo question: use integration first-wave plan, then continue narrowing/removing run-projection materialization tables

## 2026-07-01T23:12:10+02:00

Trigger: pages cache and docs-site quality concern
Candidate demos: docs-site generated pages cache blocked two integration pushes; operator reports remembered broken/stale links; pages-preview is part of PR checks
Selected/improved demo: audit docs/pages generated site, link integrity, cache role, and whether pages belongs in quick gate
Artifact action: add workload item now; investigate as a future docs/product-quality slice after current PR integration settles
Proof/caveat: current evidence proves local pages cache can be stale and costly; it does not yet prove deployed docs links are broken, so the next slice needs link-check/browser or static crawler evidence
Next demo question: Should pages rendering remain a quick-gate source cache check, or become a dedicated docs-site verification lane with generated-cache cleanup?

## 2026-07-01T23:28:33+02:00

Trigger: docs pages site audit proof
Candidate demos: generated docs site links; pages preview reliability; docs cache false failures
Selected/improved demo: PR #2500 docs-site link repair
Artifact action: PR #2500 opened with relative generated links, Markdown link rewriting, nav validation, and render pages --check link validation
Proof/caveat: Proof: focused renderer tests, render pages --check, and quick gate passed; CI/preview still pending on GitHub
Next demo question: Which docs/product slice should follow after CI: browser-capture preview smoke, fuller docs curation, or CLI surface audit?

## 2026-07-01T23:37:27+02:00

Trigger: CLI surface audit wait-time artifact
Candidate demos: CLI command latency/shape; audit brokenness; demo shelf freshness
Selected/improved demo: refresh current CLI surface audit demo
Artifact action: run devtools workspace cli-surface-audit into .agent/demos/cli-surface-audit/current and then refresh demo indexes
Proof/caveat: Proof will be artifact files plus demo-shelf verification; caveat: this is command-shape evidence, not a full UX/browser review
Next demo question: Should the next product slice promote this into a richer CLI audit with thresholds and aesthetic notes?

## 2026-07-01T23:38:42+02:00

Trigger: temporal devloop stale path finding
Candidate demos: temporal dogfood; conductor packet correctness; scratch/current removal
Selected/improved demo: fix temporal-devloop to read .agent/conductor-devloop
Artifact action: temporal-devloop currently points at .agent/scratch/current/2026-06-30-devloop and misses active OPERATING-LOG/EVENTS
Proof/caveat: Proof: command emitted operating_log_missing and source_counts.devloop_log_events=0 despite active conductor log existing
Next demo question: Fix command defaults and tests so temporal analytics dogfood uses canonical conductor state

## 2026-07-02T00:09:15+02:00

Trigger: observed event materialization integrity proof
Candidate demos: agent affordance product analyze-tools; schema/source-of-truth audit
Selected/improved demo: agent-affordance-usage/product-analyze-tools
Artifact action: regenerated current product JSON artifacts in place
Proof/caveat: serena family comparison returns in 1.553s solo on active v20 archive; observed-event caveat says source-derived tool_finished outcomes
Next demo question: commit the focused SQL/caveat fix, then continue run/context projection audit

## 2026-07-02T07:05:41+02:00

Trigger: two-session Codex export demo regeneration
Candidate demos: bounded product-native chatlog export; full transcript as a normal package artifact;
read-package layout policy
Selected/improved demo: `.agent/demos/chatlog-exports/current`
Artifact action: regenerated both Codex sessions through
`devtools workspace read-package`; bounded `dialogue.md` and `dialogue.json` to
`projection.max_tokens = 5000`; replaced stale inbox provenance manifest with a
current no-inbox policy manifest; refreshed `.agent/demos` manifest/index
Proof/caveat: proof: active-archive regeneration produced ~39 KB Markdown and
~60 KB JSON dialogue artifacts for both sessions, no encrypted reasoning hits,
and no matching Codex export staging files under `/realm/inbox`; caveat: the
default layout is a first-window bounded dialogue projection, not a semantically
selected summary of the whole devloop
Next demo question: should chatlog export expose a reusable named layout for
bounded operator-readable dialogue windows, or should the next slice shift to
embeddings/CLI diagnostics?

## 2026-07-02T10:36:16+02:00

Trigger: user flagged curr_state chatlog exports as too lean
Candidate demos: capped product-read packet; full transcript packet; raw JSONL copy
Selected/improved demo: full-chatlog sibling plus product-read concise views
Artifact action: regenerated /realm/inbox/curr_state/chatlog-exports-current and .agent/demos/chatlog-exports/current; refreshed demo indexes
Proof/caveat: full messages returned 19288/19288 and 23204/23204; product-read remains capped and labeled; raw JSONL body is not copied
Next demo question: Should full-chatlog exports move from huge JSON/Markdown files toward a streaming/chunked read-package layout?

## 2026-07-03T03:14:10+02:00

Trigger: raw-materialization closure after v23 repair
Candidate demos: archive-debt proof refresh; claim-vs-evidence coverage repair; tool-episode projection
Selected/improved demo: .agent/demos/archive-debt-summary
Artifact action: regenerated archive-debt-summary with schema v23, 16,494 sessions, 0 unresolved/actionable raw-materialization debt, 389 classified join-gap artifacts, and indexed summary.json
Proof/caveat: proof: demo regeneration uses debt payloads and quick status; devloop-review reports raw debt zero and demo shelf has 4 summaries with complete coverage; caveat: 391 raw/index join gaps remain as classified alias/non-session evidence
Next demo question: next demo question: can claim-vs-evidence produce a publishable cross-provider failure acknowledgment report that includes Codex/GPT rows and a clear sample frame?

## 2026-07-03T03:25:23+02:00

Trigger: claim-vs-evidence origin-stratified current artifact
Candidate demos: Regenerated .agent/demos/claim-vs-evidence on active schema v23 after fixing sample-frame bias. Proof: 41,774 classifiable failures, 100 unpaired coverage gaps, 5,000 inspected rows including 1,250 codex-session rows.
Selected/improved demo: --artifact
Artifact action: .agent/demos/claim-vs-evidence/claim-vs-evidence.report.json
Proof/caveat: --proof
Next demo question: devtools test tests/unit/devtools/test_claim_vs_evidence.py; devtools workspace claim-vs-evidence --limit 5000 --out-dir .agent/demos/claim-vs-evidence --json; devtools workspace demo-shelf

## 2026-07-03T03:35:50+02:00

Trigger: claim-vs-evidence auditable sample previews
Candidate demos: Added bounded next_text_preview fields to claim-vs-evidence samples and regenerated the active archive artifact, so classification examples can be audited without a separate read command.
Selected/improved demo: --artifact
Artifact action: .agent/demos/claim-vs-evidence/claim-vs-evidence.report.json
Proof/caveat: --proof
Next demo question: devtools test tests/unit/devtools/test_claim_vs_evidence.py; devtools workspace claim-vs-evidence --limit 5000 --out-dir .agent/demos/claim-vs-evidence --json; devtools workspace demo-shelf

## 2026-07-03T03:37:51+02:00

Trigger: claim-vs-evidence classifier rationale
Candidate demos: Added classification_reason and matched_marker to claim-vs-evidence samples, regenerated the active artifact, and verified samples now expose why acknowledged/ambiguous/silent decisions were made.
Selected/improved demo: --artifact
Artifact action: .agent/demos/claim-vs-evidence/claim-vs-evidence.report.json
Proof/caveat: --proof
Next demo question: devtools test tests/unit/devtools/test_claim_vs_evidence.py tests/unit/scripts/test_agent_forensics.py; devtools workspace claim-vs-evidence --limit 5000 --out-dir .agent/demos/claim-vs-evidence --json; devtools workspace demo-shelf

## 2026-07-03T03:54:53+02:00

Trigger: claim-vs-evidence performance and fables synthesis
Candidate demos: claim-vs-evidence current artifact; DSL composition demo; uplift experiment; CLI-to-daemon perf path
Selected/improved demo: .agent/demos/claim-vs-evidence
Artifact action: .agent/demos/claim-vs-evidence/claim-vs-evidence.report.json; .agent/includes/fables-poly-findings.md
Proof/caveat: devtools test tests/unit/devtools/test_claim_vs_evidence.py passed; active archive regen schema v23 inspected 5,000/41,774 paired failures with 100 unpaired gaps; regen latency 43.09s remains a perf caveat
Next demo question: prioritize CLI-to-daemon fast path or DSL projection/composition after committing this batch

## 2026-07-04T18:44:49+02:00

Trigger: browser-backed visual tape proof
Candidate demos: query-tour, reader-evidence-tour, browser-capture-tour, future live-reader-follow
Selected/improved demo: docs/examples/visual-tapes/browser-capture-tour.tape and browser-capture-tour.gif
Artifact action: added browser-capture-tour to default visual-tapes inventory; regenerated public tape; captured GIF with vhs
Proof/caveat: proof: focused visual-vhs tests passed, render visual-tapes reports 4 specs, browser-provider smoke captured deterministic ChatGPT/Claude fixtures through Chrome/extension/receiver/popup without raw debug leak, and vhs generated the GIF; caveat: this proves browser capture, not web reader following an ingested live session
Next demo question: next close-out question: should polylogue-3tl.5 implement a deterministic reader-visible live-follow lane, or should Beads narrow the acceptance criterion to browser-backed capture plus reader smoke?
