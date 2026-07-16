## [assistant 352]
According to the A27 prompt, the requested unit is **not lineage** and not “same session continued elsewhere.” It is a derived **episode**: one logical piece of work stitched across sessions, tools, artifacts, and time, using thematic/temporal evidence where no shared provider identity exists. The prompt’s key constraint is asymmetric cost: false merges are worse than missed stitches, so only topology-proven or corroborated edges should collapse into one rendered episode by default. fileciteturn0file0

Snapshot caveat: I am grounding this in the uploaded July 5 bundle and extracted code snapshot. Live code may have moved; verify paths/lines locally before filing beads or editing DDL.

## Recommendation

**[proposal] Defended recommendation:** implement episodes as a **derived `index.db` read model**, with operator decisions stored as **`user.db.assertions` stitch hypotheses/overrides**. The scorer can rebuild candidates from durable evidence; user confirmations, splits, quarantines, and labels are irreplaceable and belong in `user.db`.

**[proposal] Runner-up rejected:** store episodes directly as durable assertions only. Rejected because automatic episode stitching is probabilistic, high-volume, and rebuildable; putting every candidate in `user.db` would pollute the human/agent overlay tier. `user.db` should store judgments about episodes, not the whole derived candidate graph.

## Layer 1 — Today’s substrate

**[evidence] Polylogue already has most raw ingredients for episodes, but no first-class episode unit.** The A27 prompt explicitly anchors the design to `session_links`, `embeddings.db`, `session_profiles`, repo/cwd profiles, and cross-source telemetry. fileciteturn0file0

**[evidence] `session_links` already stores topology-proven cross-session edges with `link_type`, `inheritance`, `status`, `method`, `confidence`, `evidence_json`, and timestamps.** That is the correct lower tier for within-provider replay lineage, continuation, fork, compaction, and subagent edges; it should feed episodes but not be renamed into episodes. Code: `polylogue/storage/sqlite/archive_tiers/index.py:376-400`.

**[evidence] `session_profiles` already gives each session a derived semantic/repo/cost/timing profile: `logical_session_id`, `first_message_at`, `last_message_at`, `repo_paths_json`, `repo_names_json`, counts, costs, durations, workflow shape, terminal state, evidence payloads, inference payloads, and search text.** This is the natural row source for time-window, repo/cwd, summary-text, and cost/effort signals. Code: `polylogue/storage/sqlite/archive_tiers/index.py:799-869`.

**[evidence] `sessions` already carries direct git-ish fields: `git_branch`, `git_repository_url`, and `commit_hash`.** These are weak/explicit metadata sources for the artifact signal. Code: `polylogue/storage/sqlite/archive_tiers/index.py:39-75`.

**[evidence] Repo/cwd telemetry exists in normalized tables: `session_working_dirs`, `repos`, `session_repos`, and `session_commits`.** `session_commits` already stores `commit_sha`, `repo_id`, `detection_type`, `method`, `confidence`, and `evidence_json`, with detection types including `time_window`, `file_overlap`, `explicit_ref`, and `origin_reported`. Code: `polylogue/storage/sqlite/archive_tiers/index.py:470-519`.

**[evidence] Tool/file/path evidence is partially present in blocks/actions.** The `blocks` table has generated `tool_command` and `tool_path` columns and tool result error/exit-code fields; the `actions` view pairs tool use/result blocks by `tool_id` and `session_id`. Code: `polylogue/storage/sqlite/archive_tiers/index.py:182-220` and `index.py:336-343`.

**[evidence] Raw artifact classification exists in `source.db.raw_artifacts` with `artifact_id`, `raw_id`, `origin`, `source_path`, `artifact_kind`, `cohort_id`, `link_group_key`, and `sidecar_agent_type`.** This is not enough for “produced artifact by session action” yet, but it proves artifact identity is already a first-class storage concern. Code: `polylogue/storage/sqlite/archive_tiers/source.py:84-108`.

**[evidence] Embeddings exist at message grain, not session-summary grain.** `embeddings.db` has `message_embeddings`, `message_embeddings_meta`, and `embedding_status`, with a 1024-dimensional vector table keyed by `message_id`. Code: `polylogue/storage/sqlite/archive_tiers/embeddings.py:8-32`; docs also describe `embeddings.db` as rebuildable but expensive. `docs/schema.md:68-74`.

**[proposal] Episode scoring should initially use message/session summary vectors by deriving a session-summary embedding from already embedded messages, but the design should leave room for a dedicated `session_summary_embeddings` table later.** The current table is message-keyed, so session-level cosine is a derived aggregation unless a new embedding family is added.

**[evidence] `user.db.assertions` is already the right place for operator/agent judgments.** The unified assertions table stores `scope_ref`, `target_ref`, `kind`, `value_json`, `body_text`, `author_kind`, `evidence_refs_json`, `status`, `confidence`, `staleness_json`, `context_policy_json`, and supersession metadata. Code: `polylogue/storage/sqlite/archive_tiers/user.py:7-31`.

**[evidence] Assertion kinds already include `annotation`, `saved_query`, `recall_pack`, `decision`, `caveat`, `lesson`, `handoff`, `judgment`, `run_state`, `prompt_eval`, `transform_candidate`, and `pathology`; statuses include `candidate`, `accepted`, `rejected`, `deferred`, `superseded`, and `deleted`.** That is enough vocabulary for “stitch hypothesis accepted/rejected/split/quarantined” without inventing a new user-tier overlay. Code: `polylogue/core/enums.py:399-447`.

**[evidence] The read algebra already has `Query × Projection × Render`, structural units including `assertion`, `file`, and `run`, and named views that should move toward composable projection algebra.** The swarm brief records this baseline and the current limitation that projection is still mainly named-view-based. fileciteturn1file16

**[evidence] The composer design already wants query results to become recall entries with resolved specs and result fingerprints, then named macros/cohorts usable in set algebra.** That is adjacent to episodes because an episode result should be a reusable relation, not one printed transcript. fileciteturn1file1

**[proposal] Today’s best approximation is: query likely sessions by repo/time/text, inspect `session_commits`, inspect raw artifacts/files/actions, then create an assertion or recall pack by hand.** That is useful but not yet an episode substrate.

## Layer 2 — Near-term substrate change

**[proposal] Add `episodes` as a derived `index.db` model plus `episode_members` and `episode_edges`.** A27 only asks for `episodes` and `episode_members`, but `episode_edges` is worth adding because the merge-confidence model is edge-local while the episode is a connected component. If you only store member confidence, you lose why member A was attached to member B.

Recommended DDL shape:

```sql
CREATE TABLE IF NOT EXISTS episodes (
    episode_id              TEXT PRIMARY KEY,
    member_set_hash         BLOB NOT NULL CHECK(length(member_set_hash) = 32),
    scorer_version          INTEGER NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'candidate'
                            CHECK(status IN ('linked','corroborated','candidate','quarantined','rejected','operator_confirmed','operator_split')),
    primary_repo_id         TEXT,
    title                   TEXT,
    summary                 TEXT,
    start_at_ms             INTEGER,
    end_at_ms               INTEGER,
    member_count            INTEGER NOT NULL DEFAULT 0 CHECK(member_count >= 0),
    produced_refs_json      TEXT NOT NULL DEFAULT '[]',
    cost_rollup_json        TEXT NOT NULL DEFAULT '{}',
    effort_rollup_json      TEXT NOT NULL DEFAULT '{}',
    confidence              REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    confidence_tier         TEXT NOT NULL CHECK(confidence_tier IN ('linked','corroborated','candidate')),
    evidence_json           TEXT NOT NULL DEFAULT '{}',
    anti_evidence_json      TEXT NOT NULL DEFAULT '{}',
    materialized_at_ms      INTEGER NOT NULL,
    input_high_water_mark   TEXT,
    materializer_version    INTEGER NOT NULL
) STRICT;

CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_member_set_hash
ON episodes(member_set_hash);

CREATE TABLE IF NOT EXISTS episode_members (
    episode_id              TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    member_ref              TEXT NOT NULL,
    member_kind             TEXT NOT NULL CHECK(member_kind IN ('session','commit','pr','issue','artifact','raw_event','query_run')),
    role                    TEXT NOT NULL DEFAULT 'evidence'
                            CHECK(role IN ('seed','exploration','rubber_duck','implementation','verification','commit','pr','artifact','telemetry','evidence')),
    confidence              REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    confidence_tier         TEXT NOT NULL CHECK(confidence_tier IN ('linked','corroborated','candidate')),
    evidence_json           TEXT NOT NULL DEFAULT '{}',
    anti_evidence_json      TEXT NOT NULL DEFAULT '{}',
    position                INTEGER NOT NULL DEFAULT 0 CHECK(position >= 0),
    PRIMARY KEY (episode_id, member_ref)
) STRICT;

CREATE TABLE IF NOT EXISTS episode_edges (
    episode_id              TEXT NOT NULL REFERENCES episodes(episode_id) ON DELETE CASCADE,
    src_ref                 TEXT NOT NULL,
    dst_ref                 TEXT NOT NULL,
    confidence              REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
    confidence_tier         TEXT NOT NULL CHECK(confidence_tier IN ('linked','corroborated','candidate')),
    signals_json            TEXT NOT NULL DEFAULT '{}',
    anti_signals_json       TEXT NOT NULL DEFAULT '{}',
    method                  TEXT NOT NULL,
    PRIMARY KEY (episode_id, src_ref, dst_ref)
) STRICT;
```

**[proposal] Store episodes in `index.db`, not `user.db`.** `index.db` already owns parsed tree, search, topology, and materialized read models and is rebuildable from `source.db`; `user.db` is the irreplaceable assertion/overlay tier. `docs/schema.md:53-66`, `docs/schema.md:76-86`.

**[proposal] Store operator decisions as assertions targeted at `episode:<id>` or `episode_edge:<hash>`.** Use `kind=judgment` or a new `AssertionKind.STITCH_HYPOTHESIS` only if you want a closed vocabulary name. The current assertion schema already supports `target_ref`, `scope_ref`, evidence refs, confidence, lifecycle status, and supersession. Code: `user.py:12-31`; `enums.py:424-445`.

**[proposal] Add new user-state target kinds for `episode`, `episode_edge`, `query_run`, `query_result`, `analysis_run`, `artifact`, and `commit`.** Current target kinds stop at session/message/work_event/thread/block/attachment/paste_span and storage-only phase; topology edge is explicitly not listed until stable identity exists. Code: `polylogue/core/user_state_targets.py:1-21`, `user_state_targets.py:52-115`.

**[proposal] Use content-hashed sorted member refs for idempotent re-stitch.** `member_set_hash = sha256("\n".join(sorted(member_ref)))`; `episode_id = "episode:" + scorer_version + ":" + hex(member_set_hash[:16])` or deterministic opaque id. If the scorer changes but the member set is unchanged, the episode can update confidence/evidence without duplicating the logical unit.

**[proposal] Keep candidate episodes rebuildable and operator-confirmed episodes stable.** Automatic materialization can drop and rebuild `candidate` rows. Rows with accepted/split/rejected operator assertions should be replayed as constraints during rebuild.

## Merge/confidence model

**[proposal] The episode scorer should build pairwise candidate edges, apply a hard false-merge floor, then connected-component only over eligible edges.** Do not directly cluster by embedding similarity. Do not collapse all sessions inside a broad time window. The safe unit is an edge with signal evidence.

### Signals

**[proposal] Signal 1: repo/cwd equality is the hard prior.** Compute from, in order: `session_repos.repo_id`, `session_working_dirs.path`, `session_profiles.repo_paths_json`, and fallback `sessions.git_repository_url`/`git_branch`. Same canonical repo root or same repo id gives a strong positive prior. Different canonical repo roots is negative evidence unless a shared hard artifact overrides it.

**[proposal] Signal 2: time-window adjacency is repo-conditioned.** For same repo, use a slow-decay kernel: a 6-hour gap can still be plausible, and a same-day sequence may be a single episode. Across repos, use fast decay: even 5 minutes can be a context switch. This should be asymmetric around explicit artifact continuity: a commit/PR edge can bridge a long gap; time alone cannot.

A concrete scoring sketch:

```text
time_score =
  if same_repo:
      exp(-gap_hours / 8.0)
  elif shared_hard_artifact:
      exp(-gap_hours / 3.0) * 0.7
  else:
      exp(-gap_minutes / 10.0) * 0.2
```

**[proposal] Signal 3: embedding cosine over session summaries is soft semantic evidence.** Use `session_profiles.search_text`, `evidence_search_text`, or a generated summary as the text. If only message embeddings exist, derive a session vector by weighted mean over substantive authored message embeddings, excluding tool/protocol/context material via `material_origin`. Current message-level embeddings make this possible but not ideal. Code: `embeddings.py:8-32`; `docs/data-model.md:76-85`.

**[proposal] Signal 4: shared-artifact overlap is the strongest positive signal.** Artifacts include file paths from tool calls, `tool_path`, command strings, git SHAs, branch names, PR/issue numbers, error fingerprints, raw artifact paths, and commit refs. This signal is invariant across tool handoffs where semantic embeddings go weak, exactly as A27 states. fileciteturn0file0

**[proposal] Persist every signal contribution in edge-level `signals_json`.** Example:

```json
{
  "repo": {
    "score": 0.95,
    "matched": true,
    "source": "session_repos",
    "repo_id": "https://github.com/...␟/realm/project/polylogue"
  },
  "time": {
    "score": 0.72,
    "gap_ms": 14400000,
    "kernel": "same_repo_exp_gap_hours_8"
  },
  "embedding": {
    "score": 0.81,
    "method": "session_summary_mean_message_embedding",
    "model": "voyage-...",
    "source": "embeddings.db"
  },
  "artifact": {
    "score": 1.0,
    "overlap": [
      {"kind": "commit_sha", "value": "4b9389d75"},
      {"kind": "path", "value": "polylogue/storage/sqlite/archive_tiers/index.py"},
      {"kind": "issue", "value": "gh-2547"}
    ]
  }
}
```

### Tiers and false-merge floor

**[proposal] Use three positive tiers: `linked`, `corroborated`, `candidate`.**

`linked`: topology-proven or explicit-identity edge. Examples: `session_links` continuation/fork/subagent; explicit parent/child session id; explicit “continued from” or “spawned subagent” metadata. This tier can render as one episode by default.

`corroborated`: at least two independent positive signals, including one hard non-semantic signal. Allowed pairs: repo + artifact, artifact + time, repo + commit, explicit PR/issue + time, path overlap + repo. This tier can render as one episode by default.

`candidate`: embedding + time only, or weak repo + time, or semantic similarity without hard artifact/repo. This tier should be visible in “possible stitches,” but not collapsed into one episode by default.

**[proposal] A candidate edge must never become default-rendered merely by high cosine.** This is the main false-merge floor. Embeddings are good recall; artifact/repo evidence is what earns merging.

**[proposal] Anti-stitch signals should subtract confidence and can quarantine edges.** Anti-stitches include different canonical repo root, disjoint file sets, contradictory goals, non-overlapping PR/issue identifiers, explicit cycle-break/quarantine marker from topology, conflicting branch roots, and large time gaps with no shared artifact.

**[proposal] Use hard vetoes sparingly.** Different repo root is a strong negative, not an absolute veto, because a commit/PR/artifact can bridge across tools or repositories in monorepo/submodule/cross-repo work. A quarantined topology cycle-break marker should be an absolute veto unless operator overrides.

**[proposal] Make missed-stitches cheap and false-merges expensive in UI.** Default episode views should include only `linked` and `corroborated`; a sidebar or section can show `candidate stitches not included`. Operator can promote candidates; rejected candidates feed the scorer as negative examples.

A concrete tiering function:

```text
linked if topology_edge.status != quarantined
       and link_type in {continuation, branch, fork, subagent, compaction}

corroborated if
  positive_independent_signals >= 2
  and max(repo_score, artifact_score) >= hard_signal_threshold
  and anti_score < quarantine_threshold

candidate if
  embedding_score >= semantic_threshold
  and time_score >= weak_time_threshold
  and anti_score < hard_veto_threshold
```

**[proposal] Confidence should be monotonic in hard evidence, but not purely additive.** Use a small calibrated logistic model later, but start with a transparent weighted formula:

```text
base =
  0.35 * repo_score +
  0.20 * time_score +
  0.20 * embedding_score +
  0.45 * artifact_score -
  0.60 * anti_score

confidence = clamp(base, 0, 1)
```

Then apply tier gates. The numbers should be configurable and recorded in `scorer_version`.

## Payoffs

**[proposal] Episode→commit/PR attribution should be first-class via `produced_refs_json`.** A commit with no in-window session can still be an episode member if it shares hard artifacts with sessions: paths, branch, issue/PR id, error fingerprint, or explicit commit mention. This satisfies A27’s “lynchpin cross-source telemetry” note: telemetry can enter the episode even when no matching AI session exists. fileciteturn0file0

**[proposal] Episode-level cost/effort rollup should dedup lineage and honor `material_origin`.** Cost should roll up from sessions/messages using `session_profiles` and cost tables, but avoid double-counting prefix-shared lineage using `session_profiles.logical_session_id` and `session_links.inheritance`. Authored effort should separate human-authored, assistant-authored, operator command, runtime protocol, tool result, generated context pack, and generated analysis pack using `material_origin`. Code/docs: `docs/data-model.md:76-85`, `docs/data-model.md:94-98`, `docs/data-model.md:135-145`.

**[proposal] Episode read-view should render an interleaved cross-tool transcript.** It should sort segments by time, group by member, badge every segment with origin/tool/session/commit/artifact, and attach stitch-evidence anchors at the boundaries: “Claude Code implementation stitched to ChatGPT rubber-duck because same repo + path overlap + issue #2547 + 42min gap.”

**[proposal] Episode read-view should be a projection preset over the read algebra, not a hardcoded silo.** Existing docs already want `Query × Projection × Render`, with projection moving beyond named views. The episode view should add an `episodes` family/unit plus a `layout=episode` render preset. fileciteturn1file16

Example render:

```text
Episode: readiness verifier split (#2547)
confidence: corroborated 0.91
produced: commit:4b9389d75, pr:#2547

[Cursor] exploration, 13:10–13:24
  ... evidence window ...

[ChatGPT] rubber-duck, 13:31–13:48
  ... summary / decision ...

[Claude Code] implementation, 14:02–14:56
  ... tool calls, diffs, test results ...

[Git] commit 4b9389d75
  files: ...
  stitch evidence: same repo, path overlap, commit mention, 34m gap
```

**[proposal] Episode should become a query unit.** Add terminal source `episodes where ...`, plus fields `repo`, `produced`, `confidence`, `tier`, `member.origin`, `path`, `commit`, `issue`, `cost`, `duration`, `status`. This should compose with set algebra and cohorts.

Example queries:

```text
episodes where repo:polylogue and produced:commit and confidence_tier:corroborated
  | group by produced.kind
  | count
```

```text
episodes where artifact.path:"polylogue/storage" and anti_evidence:any
  | read layout:episode
```

```text
episodes where member.origin:chatgpt and member.origin:claude-code
  | sort by cost desc
  | limit 20
  | read
```

**[proposal] Episode stitch hypotheses should round-trip through assertions.** Store the automatic episode in `index.db`; create a candidate assertion only when shown to the operator, exported into a report, or manually acted on. Operator actions: confirm, split, reject edge, quarantine cycle, rename/title, attach produced ref, mark as exemplar. These assertions feed future scorer calibration.

## Layer 3 — Full direction

**[proposal] Episode is the missing middle unit between “session lineage” and “analysis/workflow.”** It sits above provider replay lineage and below high-level analysis runs. It lets Polylogue answer: “what did solving X actually involve across Cursor, ChatGPT, Claude Code, terminal, git, PR, and telemetry?”

**[evidence] The earlier design direction already wants query objects, recall entries, saved macros/cohorts, set algebra, preview, and analysis/report rendering.** The composer report says committed runs should write recall entries with query text, resolved spec, result fingerprint, and timestamp, and named recall entries become macros usable as set-op operands. fileciteturn1file1 The swarm brief says query/read/render should be composable and that set algebra/macros are on the fnm roadmap. fileciteturn1file16

**[proposal] Full direction: `raw evidence → lineage → episodes → cohorts → analyses → reports`.** Episodes should not replace query objects or analysis runs. They provide a reusable work-unit grain for downstream analyses.

**[proposal] Episode graph should eventually include non-session members.** `episode_members.member_kind` should allow `commit`, `pr`, `issue`, `artifact`, `raw_event`, and later `sinex_event`/`activitywatch_window`/`terminal_recording`. A27 explicitly requires that a commit with no in-window session can still be an episode member. fileciteturn0file0

**[proposal] The full system should support “work replay” as reconstruct/reproduce, not deterministic replay.** Episode reconstruction can render commands, diffs, files, commits, tool results, and failure points. It should say “reconstruction” or “reproduction plan” unless exact filesystem/process state is captured.

**[proposal] Add episode-aware context packs.** If an agent asks to continue work, Polylogue should compile the current episode: relevant sessions, recent commits, produced artifacts, accepted assertions, open candidate stitches, failures, and next likely beads. This directly improves post-compaction recovery and multi-agent handoff.

**[proposal] Add episode-level evaluation.** Once episodes exist, you can measure cross-tool effort: “which episodes used rubber-duck chats before implementation?”, “how much did Claude Code implementation cost after ChatGPT design?”, “which commits had no visible AI-session antecedent?”, “which episodes contain unacknowledged failed tool results?” That is a much better analytic substrate than session-only reports.

## Beads I would add

**[proposal] `episode-derived-unit` — first-class episode tables and materializer.** Design/implement `episodes`, `episode_members`, and `episode_edges` in `index.db`, with content-hashed member sets, scorer version, confidence tier, evidence/anti-evidence JSON, and rebuild semantics.

**[proposal] `episode-merge-scorer-v1` — four-signal scorer.** Implement repo/cwd, time-window, embedding-summary, and shared-artifact overlap signals; persist each signal contribution; enforce linked/corroborated/candidate tiers.

**[proposal] `episode-false-merge-floor` — anti-stitch and quarantine.** Implement negative evidence, hard quarantine markers, no-default-render for candidate-only stitches, and operator override hooks.

**[proposal] `episode-assertion-overrides` — stitch hypotheses in user.db.** Add target kinds for episode/episode_edge and an operator flow to confirm/split/reject; feed accepted/rejected assertions back into scorer calibration.

**[proposal] `episode-produced-ref-attribution` — commit/PR/artifact membership.** Generalize `session_commits` evidence into episode produced refs; allow commits/PRs with no matching session to enter episodes through artifact overlap.

**[proposal] `episode-cost-effort-rollup` — lineage/material-origin-aware totals.** Roll up cost/duration/tokens using logical-session dedup and `material_origin` filters.

**[proposal] `episode-read-view` — interleaved transcript projection.** Add `episodes where ...` and `read layout:episode` rendering with per-segment origin badges and stitch-evidence anchors.

**[proposal] `episode-corpus-validation` — scorer precision audit.** Sample candidate/corroborated/linked stitches, require operator review, measure false-merge/missed-stitch rates, and pin thresholds before default rendering.

## Open questions for the operator

**[proposal] How aggressive should default stitching be?** My recommendation is conservative: only `linked` and `corroborated` render as one episode; `candidate` remains visible but separate.

**[proposal] What is the first target corpus?** I would start with Polylogue repo work because repo/cwd/commit/PR/path evidence is strongest there.

**[proposal] Should episode id be stable across scorer versions?** I recommend stable member-set hash plus scorer-version metadata, not scorer-version in the visible id. The same member set is the same episode hypothesis; confidence can change.

**[proposal] Should operator-confirmed episodes become immutable snapshots?** I recommend “soft immutable”: preserve confirmed member set unless explicitly split/merge; new candidates can be proposed adjacent.

**[proposal] What should count as a hard artifact?** Commit SHA and PR/issue id are obvious. File path overlap is strong only after path normalization and repo-root matching. Error fingerprints need a schema before they become hard.

**[proposal] Should cross-repo episodes be supported in v1?** I recommend yes, but only via artifact/PR/commit evidence, not time+embedding.

## What’s missing

**[evidence] I did not run the daemon or inspect a live archive.** All claims are from the uploaded July 5 bundle and extracted code snapshot.

**[evidence] I did not verify whether `session_profiles` convergence is currently complete on the live archive.** The previous chatlog mentions `w79` session-profile convergence as an unaudited/open thread, so any design relying heavily on `session_profiles` should verify freshness locally before launch. fileciteturn1file15

**[evidence] I did not inspect a live `embeddings.db` for coverage.** The schema exists, but session-summary embedding quality and coverage are unverified.

**[evidence] I did not inspect Sinex/Lynchpin telemetry schemas.** A27 asks to anchor to lynchpin cross-source telemetry, but the uploaded task only names the need; the current bundle snippets do not provide a concrete telemetry schema. fileciteturn0file0

**[evidence] I did not produce a migration PR or bead mutations.** This is a design answer only.

**[proposal] The first implementation should deliberately under-stitch, then use operator-confirm/reject assertions to calibrate.** The product wins by avoiding embarrassing false merges first; better recall can come later.
