## [assistant 375]
According to the 2026-07-05 A18 task prompt, the invariant you want is specifically about a self-ingesting Polylogue archive: browser-captured GPT/webui R&D chats and dev sessions become archive material, distilled findings become assertions, and those assertions may later be injected into agent context. The danger is the recovery-digest class: an agent-authored derived claim becomes “evidence” for another agent-authored derived claim, until the loop launders a hallucination into context as if it were grounded truth. fileciteturn1file0

My defended recommendation: **make “injectability” depend on a materialized provenance verdict, not on `context_policy_json.inject` alone.** The context scheduler should only inject an assertion when all of these are true: the row asks for injection, the author/trust state allows injection, the closed-loop predicate is false, the provenance graph is acyclic, and the cited content hashes still match. The runner-up I reject is “never inject agent-authored assertions unless a human manually rewrites them.” That is safe but too tight: it destroys scalable external annotation, agent-produced candidate analyses, and cross-session synthesis. The right posture is not “agents never speak into future context”; it is “agents may produce candidates, but only grounded/promoted candidates can become memory.”

## Layer 1: today’s substrate

[evidence] The current `user.db.assertions` table already has most of the necessary fields: `author_ref`, `author_kind`, `evidence_refs_json`, `status`, `visibility`, `confidence`, `staleness_json`, `context_policy_json`, and timestamps. It also defaults `context_policy_json` to `{"inject":false}`. Code anchor: `polylogue/storage/sqlite/archive_tiers/user.py:12-31`.

[evidence] The assertion vocabulary already covers the right semantic space. `AssertionKind` includes `annotation`, `correction`, `saved_query`, `recall_pack`, `decision`, `caveat`, `lesson`, `handoff`, `judgment`, `run_state`, `prompt_eval`, `transform_candidate`, and `pathology`; statuses include `active`, `candidate`, `accepted`, `rejected`, `deferred`, `superseded`, and `deleted`. Code anchor: `polylogue/core/enums.py:399-447`.

[evidence] The context-policy default is already conservative: `AssertionContextPolicy.default()` returns `{"inject": False}`, and `from_raw()` sets `inject` to false when omitted. Code anchor: `polylogue/core/assertions.py:43-66`.

[evidence] Existing derived candidates already follow the safety pattern you want. Transform candidates are mirrored as non-injected candidate assertions with `context_policy={"inject": False, "promotion_required": True}`; deterministic pathology findings do the same. Code anchors: `polylogue/storage/sqlite/archive_tiers/user_write.py:1008-1053` and `polylogue/storage/sqlite/archive_tiers/user_write.py:1060-1115`.

[evidence] Operator judgment already exists. `judge_assertion_candidate()` only accepts candidate rows, writes an explicit `judgment` assertion authored by `user`, and promotes accepted/superseded candidates through `_promote_candidate_assertion()`. Code anchors: `polylogue/storage/sqlite/archive_tiers/user_write.py:1245-1326` and `polylogue/storage/sqlite/archive_tiers/user_write.py:1338-1370`.

[evidence] There is a gap: today, `list_assertion_claims(..., context_inject=True)` filters by `claim.context_policy.get("inject")` after reading rows; it does not also require human authorship, non-closed-loop grounding, acyclic provenance, or fresh hashes. Code anchor: `polylogue/storage/sqlite/archive_tiers/user_write.py:1531-1588`.

[evidence] The current public ref layer supports `session`, `message`, `block`, `commit`, `agent`, `user`, `repo`, `assertion`, `run`, `context-snapshot`, `observed-event`, etc., and the evidence-ref DTO supports session/message/block pointers. Code anchors: `polylogue/core/refs.py:8-40`, `polylogue/core/refs.py:76-206`. That is enough to start, but it is not enough to treat raw source bytes, GitHub API snapshots, external docs, and citation anchors as first-class grounding without a small extension.

[evidence] The “PR #123 merged” recovery-digest problem is not hypothetical. The audit log says the recovery-digest fabrication was already killed under #2482, and that pathologies now derive from structured tool-result fields rather than regex prose. fileciteturn3file0 The code also explicitly says test/check regexes do not assert outcomes; outcomes come from structured keystone tool-result fields. Code anchor: `polylogue/insights/transforms.py:77-80`.

[evidence] The broader project posture is already aligned with this invariant: Polylogue is framed as a “system of record for AI work,” source.db and user.db are durable tiers, index/embeddings are derived, and adjacent roadmap threads include context scheduler, outcome-grounded analytics, and construct-validity metadata. fileciteturn2file8

[proposal] Today’s minimal enforceable rule is:

```text
If assertion.author_kind != 'user'
AND assertion.status IN ('active', 'accepted', 'candidate')
AND context_policy.inject is true or could become true
AND the assertion lacks a non-agent, non-derived grounding anchor,
THEN force status='candidate' and context_policy.inject=false.
```

That can be implemented today as a conservative scheduler-side check before injection, without a schema migration. It will be slightly too tight because current refs do not fully distinguish “agent said X” from “tool result proves X,” but too tight is the right default for context injection.

## Layer 2: near-term substrate change

[proposal] Add a small derived/provenance cache, not a new silo. Assertions remain canonical in `user.db.assertions`; the new piece is a materialized **assertion grounding verdict** computed by the converger before the context scheduler runs.

The near-term DDL shape I recommend is:

```sql
CREATE TABLE IF NOT EXISTS assertion_citation_anchors (
    assertion_id        TEXT NOT NULL,
    evidence_ref        TEXT NOT NULL,
    resolved_ref        TEXT,
    resolved_kind       TEXT NOT NULL,
    grounding_class     TEXT NOT NULL CHECK (
        grounding_class IN (
            'agent_session',
            'human_message',
            'human_judgment',
            'tool_result',
            'source_raw',
            'git_commit',
            'git_tree',
            'external_doc',
            'external_issue',
            'external_pr',
            'assertion',
            'unknown'
        )
    ),
    compatible_claim    INTEGER NOT NULL DEFAULT 0 CHECK(compatible_claim IN (0,1)),
    author_kind         TEXT,
    content_hash_hex    TEXT,
    observed_hash_hex   TEXT,
    drifted             INTEGER NOT NULL DEFAULT 0 CHECK(drifted IN (0,1)),
    resolver_version    INTEGER NOT NULL,
    resolved_at_ms      INTEGER NOT NULL,
    PRIMARY KEY(assertion_id, evidence_ref, resolved_ref)
) STRICT;

CREATE TABLE IF NOT EXISTS assertion_provenance_edges (
    src_assertion_id    TEXT NOT NULL,
    dst_assertion_id    TEXT NOT NULL,
    edge_kind           TEXT NOT NULL DEFAULT 'cites',
    status              TEXT CHECK(status IN ('resolved','repaired','quarantined') OR status IS NULL),
    evidence_ref        TEXT NOT NULL,
    created_at_ms       INTEGER NOT NULL,
    PRIMARY KEY(src_assertion_id, dst_assertion_id, evidence_ref)
) STRICT;

CREATE TABLE IF NOT EXISTS assertion_grounding_verdicts (
    assertion_id          TEXT PRIMARY KEY,
    closed_loop           INTEGER NOT NULL CHECK(closed_loop IN (0,1)),
    provenance_cycle      INTEGER NOT NULL CHECK(provenance_cycle IN (0,1)),
    drifted               INTEGER NOT NULL CHECK(drifted IN (0,1)),
    externally_grounded   INTEGER NOT NULL CHECK(externally_grounded IN (0,1)),
    verdict               TEXT NOT NULL CHECK (
        verdict IN ('grounded','closed_loop','cycle','drifted','unknown')
    ),
    computed_at_ms        INTEGER NOT NULL,
    resolver_version      INTEGER NOT NULL
) STRICT;
```

[proposal] The important field is `compatible_claim`, not just `grounding_class`. A raw GPT transcript can prove “the assistant said X,” but it cannot prove “PR #123 merged.” A tool-result block with exit code can prove a command outcome. A git commit anchor can prove a commit exists. A human judgment can release a candidate. This compatibility bit is the difference between a real construct-validity guard and a checkbox.

### The closed-loop predicate

[proposal] This is the exact SQL predicate I would put behind `assertion_grounding_verdicts`. It assumes `assertion_citation_anchors` and `assertion_provenance_edges` have been populated by a resolver that expands `evidence_refs_json`, follows `assertion:<id>` citations, resolves message/block/session refs, records content hashes, and classifies anchors.

```sql
WITH RECURSIVE
seed(assertion_id) AS (
    SELECT a.assertion_id
    FROM assertions a
    WHERE COALESCE(a.author_kind, 'agent') != 'user'
      AND COALESCE(a.status, 'active') IN ('active', 'accepted', 'candidate')
),

walk(root_assertion_id, assertion_id, path, depth, cycle) AS (
    SELECT
        s.assertion_id,
        s.assertion_id,
        '|' || s.assertion_id || '|',
        0,
        0
    FROM seed s

    UNION ALL

    SELECT
        w.root_assertion_id,
        e.dst_assertion_id,
        w.path || e.dst_assertion_id || '|',
        w.depth + 1,
        CASE
            WHEN instr(w.path, '|' || e.dst_assertion_id || '|') > 0 THEN 1
            ELSE 0
        END
    FROM walk w
    JOIN assertion_provenance_edges e
      ON e.src_assertion_id = w.assertion_id
    WHERE w.depth < 32
      AND w.cycle = 0
      AND COALESCE(e.status, 'resolved') != 'quarantined'
),

anchors AS (
    SELECT
        w.root_assertion_id,
        c.evidence_ref,
        c.grounding_class,
        c.compatible_claim,
        c.drifted
    FROM walk w
    LEFT JOIN assertion_citation_anchors c
      ON c.assertion_id = w.assertion_id
),

agg AS (
    SELECT
        w.root_assertion_id AS assertion_id,

        MAX(w.cycle) AS provenance_cycle,

        COALESCE(SUM(
            CASE
                WHEN a.compatible_claim = 1
                 AND a.drifted = 0
                 AND a.grounding_class IN (
                    'human_message',
                    'human_judgment',
                    'tool_result',
                    'source_raw',
                    'git_commit',
                    'git_tree',
                    'external_doc',
                    'external_issue',
                    'external_pr'
                 )
                THEN 1 ELSE 0
            END
        ), 0) AS external_ground_count,

        COALESCE(SUM(
            CASE
                WHEN a.grounding_class = 'agent_session'
                  OR a.grounding_class = 'assertion'
                  OR a.grounding_class = 'unknown'
                  OR a.grounding_class IS NULL
                THEN 1 ELSE 0
            END
        ), 0) AS closed_or_unknown_count,

        COALESCE(MAX(CASE WHEN a.drifted = 1 THEN 1 ELSE 0 END), 0) AS any_drift
    FROM walk w
    LEFT JOIN anchors a
      ON a.root_assertion_id = w.root_assertion_id
    GROUP BY w.root_assertion_id
)

SELECT
    assertion_id,
    CASE
        WHEN provenance_cycle = 1 THEN 'cycle'
        WHEN any_drift = 1 THEN 'drifted'
        WHEN external_ground_count = 0 THEN 'closed_loop'
        ELSE 'grounded'
    END AS verdict,
    CASE WHEN external_ground_count = 0 THEN 1 ELSE 0 END AS closed_loop,
    provenance_cycle,
    any_drift AS drifted,
    CASE WHEN external_ground_count > 0 THEN 1 ELSE 0 END AS externally_grounded
FROM agg;
```

[proposal] The closed-loop predicate is:

```sql
closed_loop = 1
WHERE author_kind != 'user'
  AND external_ground_count = 0
```

with cycle and drift treated as stronger quarantine reasons. In other words: if an agent-authored claim is supported only by agent sessions, other assertions that themselves bottom out in agent sessions, unknown refs, or nothing, it cannot be injected.

### Quarantine state machine

[proposal] Store quarantine as ordinary assertion state plus context policy. Do not add `AssertionStatus.QUARANTINED` yet; that creates another lifecycle axis and requires more surface work. Use the already-existing candidate machinery.

```text
NEW ASSERTION
  |
  | converger computes verdict
  v
AGENT + CLOSED_LOOP
  -> status='candidate'
  -> context_policy.inject=false
  -> context_policy.promotion_required=true
  -> context_policy.quarantine_reason='closed_loop'
  -> staleness.recursive_safety='closed_loop'

AGENT + CYCLE
  -> status='candidate'
  -> context_policy.inject=false
  -> context_policy.promotion_required=true
  -> context_policy.quarantine_reason='provenance_cycle'

AGENT + DRIFTED
  -> keep current status if useful for review, but force inject=false
  -> context_policy.quarantine_reason='evidence_drift'
  -> staleness.hash_drift=true

AGENT + EXTERNALLY_GROUNDED
  -> may remain active/accepted/candidate according to ordinary lifecycle
  -> still not inject-eligible unless promoted or explicitly allowed by policy

USER JUDGMENT ACCEPTS
  -> candidate becomes accepted/superseded and a user-authored judgment row is written
  -> promoted row has author_kind='user'
  -> promotion_required removed
  -> inject remains false unless the operator or policy explicitly requests injection

ADDED EXTERNAL COMPATIBLE CITATION
  -> resolver recomputes verdict='grounded'
  -> quarantine_reason cleared
  -> inject may be re-enabled only if author/trust gate also passes
```

[evidence] This deliberately follows existing code: derived candidates are already stored as `candidate`, private, non-injected, promotion-required rows; operator judgments already write `author_kind='user'` judgment rows and create promoted active assertions. Code anchors: `user_write.py:1008-1053`, `user_write.py:1060-1115`, `user_write.py:1245-1370`.

[proposal] I would change `_promote_candidate_assertion()` only slightly: after a candidate is accepted, do not blindly inherit the old context policy. Instead, normalize it through a release function:

```text
release_policy(candidate, judgment):
    inject = false by default
    remove promotion_required
    remove quarantine_reason
    set promoted_by = judgment.assertion_id
    set recursive_safety = 'operator_released'
```

If the operator wants the accepted assertion injected, that should be an explicit `judge --accept --inject` or separate `assertions set-policy --inject true`, not an accidental side effect. This respects the existing code line that currently preserves `inject` as false unless explicitly set (`user_write.py:1348-1350`).

### Author-kind differential trust gate

[proposal] The context scheduler’s injection predicate should become:

```sql
SELECT a.*
FROM assertions a
JOIN assertion_grounding_verdicts v
  ON v.assertion_id = a.assertion_id
WHERE json_extract(a.context_policy_json, '$.inject') = 1
  AND COALESCE(a.status, 'active') IN ('active', 'accepted')
  AND v.verdict = 'grounded'
  AND v.closed_loop = 0
  AND v.provenance_cycle = 0
  AND v.drifted = 0
  AND (
        COALESCE(a.author_kind, 'user') = 'user'
        OR json_extract(a.context_policy_json, '$.promotion_required') = 0
      );
```

[proposal] The rule in prose:

```text
author_kind=user:
    inject-eligible if policy asks for injection and safety verdict is grounded/fresh.

author_kind=agent / detector / transform / model:
    not inject-eligible by default.
    must be promoted by user judgment or externally grounded and policy-released.

author_kind=detector:
    can be treated differently only for deterministic detectors whose evidence anchors are tool_result/source_raw/git and whose detector version is recorded.
```

This is stricter than current `context_inject` filtering, which checks only the policy flag after rows are loaded. Code anchor: `user_write.py:1582-1588`.

### Provenance-cycle guard

[evidence] Polylogue already has a topology status vocabulary with `QUARANTINED`, and `session_links.status` already accepts `quarantined` at the storage layer. Code anchors: `polylogue/core/enums.py:314-321`; `polylogue/storage/sqlite/archive_tiers/index.py:376-400`.

[proposal] Reuse that vocabulary, but not necessarily the `session_links` table, for query/finding provenance. A query→finding→query cycle is not a session-link edge; it is an assertion/provenance edge. So the shared enum value is right, while the table should be `assertion_provenance_edges` or, in the full version, `provenance_edges`.

Cycle detection:

```sql
WITH RECURSIVE walk(root, node, path, cycle) AS (
    SELECT src_assertion_id, dst_assertion_id,
           '|' || src_assertion_id || '|' || dst_assertion_id || '|',
           0
    FROM assertion_provenance_edges
    WHERE COALESCE(status, 'resolved') != 'quarantined'

    UNION ALL

    SELECT walk.root,
           e.dst_assertion_id,
           walk.path || e.dst_assertion_id || '|',
           CASE
             WHEN instr(walk.path, '|' || e.dst_assertion_id || '|') > 0 THEN 1
             ELSE 0
           END
    FROM walk
    JOIN assertion_provenance_edges e
      ON e.src_assertion_id = walk.node
    WHERE walk.cycle = 0
      AND COALESCE(e.status, 'resolved') != 'quarantined'
)
UPDATE assertion_provenance_edges
SET status = 'quarantined'
WHERE src_assertion_id IN (
    SELECT root FROM walk WHERE cycle = 1
);
```

[proposal] The converger must then refuse to auto-promote any assertion whose provenance closure includes a quarantined edge. This is the same doctrine as session topology: cycles should be surfaced so the operator can quarantine the archive slice rather than silently linearize it. Code anchor for existing topology cycle posture: `polylogue/insights/topology.py:135-145`.

### Auto-downgrade on drift

[evidence] The durable tiers already store hashes at the right levels: `source.db.raw_sessions.blob_hash` stores acquired raw bytes; `index.db.sessions.content_hash`, `messages.content_hash`, `paste_spans.content_hash`, and history sidecars carry content hashes. Code anchors: `source.py:15-33`, `source.py:51-59`, `index.py:39-75`, `index.py:92-126`, `index.py:570-583`, `source.py:146-156`.

[proposal] Each citation anchor should store the hash it was resolved against. On converger pass:

```sql
UPDATE assertion_citation_anchors
SET drifted = 1
WHERE content_hash_hex IS NOT NULL
  AND observed_hash_hex IS NOT NULL
  AND content_hash_hex != observed_hash_hex;
```

Then:

```sql
UPDATE assertions
SET context_policy_json = json_set(
        COALESCE(context_policy_json, '{"inject":false}'),
        '$.inject', json('false'),
        '$.quarantine_reason', 'evidence_drift'
    ),
    staleness_json = json_set(
        COALESCE(staleness_json, '{}'),
        '$.hash_drift', json('true')
    )
WHERE assertion_id IN (
    SELECT assertion_id
    FROM assertion_grounding_verdicts
    WHERE drifted = 1
);
```

[proposal] Conservative failure mode: if the resolver cannot compute the current observed hash, treat the anchor as `unknown`, not as fresh. Missing evidence cannot release injection.

## Layer 3: full direction

[proposal] The full design is a general **provenance DAG**, not an assertion-only patch. Nodes are `session`, `message`, `block`, `raw`, `artifact`, `query_definition`, `query_run`, `result_relation`, `cohort`, `assertion`, `annotation_batch`, `analysis_run`, `report`, `commit`, `external_doc`, and `human_judgment`.

[proposal] Edges are `cites`, `derived_from`, `supports`, `contradicts`, `supersedes`, `produced`, `consumed`, `rendered_as`, `queried_by`, `promoted_by`, and `included_in_context`. Every edge has `status`, `confidence`, `resolver_version`, `content_hash`, and `claim_kind`.

[proposal] The closed-loop predicate then becomes graph-theoretic:

```text
An assertion is recursively safe iff every path from the assertion to terminal evidence
has at least one compatible, non-drifted grounding node outside the agent-authored archive-output class,
and the assertion’s provenance closure is acyclic.
```

[proposal] Query objects make this much more powerful. The composer design already proposes that every committed run writes a recall entry with query text, resolved spec, result fingerprint, and timestamp, and that named entries become macros. fileciteturn1file7 In the full design, these recall entries become `query_definition`, `query_run`, and `result_relation` nodes. A report can cite the exact query run and result relation it used; an assertion can target a query result; a later agent can re-query the annotations without laundering the report itself.

[evidence] This matches the swarm direction: the read baseline is `Query × Projection × Render`, with query units including assertions, messages, actions, files, runs, observed events, and context snapshots; the architecture direction says the client should be thin, the daemon hot, and the composer should provide preview/completion against the resident substrate. fileciteturn1file17

[proposal] In the full direction, “source.db raw bytes” is not one undifferentiated truth class. It is a typed anchor:

```text
raw_transcript_anchor: proves text was present in captured source
tool_result_anchor: proves a tool returned status/output/exit code
git_anchor: proves repo state / commit / tree
external_doc_anchor: proves fetched external document content
human_judgment_anchor: proves operator accepted or asserted a claim
```

This prevents the subtle loophole: an agent transcript saying “PR #123 merged” should not ground a `pr_merged` claim; it only grounds an `assistant_said` claim. That is the generalized fix for the recovery-digest incident.

## Why the predicate prevents laundering

[proposal] Assume an agent-authored assertion A says “PR #123 merged.” Its evidence refs point to a prior agent session B that also says “PR #123 merged,” and B’s own evidence points to another agent summary C. The resolver classifies B/C as `agent_session` or `assertion`, finds no compatible `git_commit`, `external_pr`, `tool_result`, or `human_judgment`, and sets `external_ground_count=0`. The verdict becomes `closed_loop`. The quarantine transition forces `status='candidate'` and `context_policy.inject=false`. The scheduler’s gate requires `verdict='grounded'`, so A cannot enter future context. Therefore no future agent receives A as retrieved evidence unless an operator judges it or a compatible external citation is added.

[proposal] If A cites itself indirectly through a query result, the recursive CTE detects the cycle and marks the edge/verdict `quarantined`. The converger refuses auto-promotion through quarantined provenance. So a query→finding→query loop cannot bootstrap itself into memory.

[proposal] If A was once grounded but the cited content hash moves, drift forces `inject=false` before the scheduler reads it. So stale external grounding cannot silently continue to release an assertion.

## Failure modes

[proposal] Too loose means fabrication leaks. Examples:

An agent-authored assertion with `context_policy.inject=true` cites only an assistant report, and the scheduler injects it because the current gate only checks the policy flag.

A raw transcript quote is treated as proof of the world claim it contains; “assistant said PR #123 merged” becomes “PR #123 merged.”

An assertion cites another assertion that cites it back through a saved query/result, and the resolver does not walk the closure.

A citation’s content hash changes after reingest, but the assertion remains injectable because no drift pass downgrades it.

A detector-authored assertion is treated as safe merely because `author_kind='detector'`, despite being regex over prose rather than structured tool-result evidence. The #2482 recovery-digest incident is exactly the warning here. fileciteturn3file0

[proposal] Too tight means legitimate synthesis is suppressed. Examples:

A user explicitly accepts an agent candidate, but the system still treats it as closed-loop because the original evidence was agent-authored.

A claim about “what the agent said” is blocked even though raw transcript bytes are exactly the correct grounding.

A deterministic detector over structured tool-result blocks is blocked because `author_kind=detector` is not `user`.

A cross-session synthesis cites many agent sessions plus one strong git/tool-result anchor, but the predicate requires every citation to be external instead of requiring at least one compatible external grounding path.

A session-level ref is treated as agent-only even when the citation anchor points to a human-authored message span inside it. The fix is citation anchors with span/hash/claim-kind compatibility, not blunt session refs.

## What to file as beads

[proposal] I would file this under the 37t context/memory loop and connect it to 37t.12. The current bundle says 37t is the context/memory loop, and that 37t.11 is the context scheduler and single arbiter of what enters agent context; it also says 37t.12 was created as the missing judgment-queue surface over user_write.py’s already-existing substrate. fileciteturn2file11turn2file10

Proposed beads:

`37t.recursive-safety-invariant`: Context scheduler blocks closed-loop agent assertions from injection.

`37t.assertion-citation-anchors`: Resolve assertion evidence refs into typed anchors with grounding_class, compatible_claim, hash, and resolver version.

`37t.provenance-cycle-quarantine`: Add assertion/query/finding provenance edge cycle detection using `quarantined` status.

`37t.evidence-drift-downgrade`: Recompute cited hashes and force inject:false on drift.

`37t.scheduler-trust-gate`: Replace `context_policy.inject`-only filtering with `inject && author/trust && grounded && acyclic && fresh`.

`37t.annotation-batch-safety`: Bulk imported agent annotations land as candidates with promotion_required until grounded or judged.

`9l5.claim-grounding-compatibility`: Claim-kind → allowed grounding-class registry, so raw transcript bytes cannot prove PR/git/tool claims.

## Open questions for the operator

[proposal] Should user-authored messages inside captured GPT/webui chats count as `human_message` grounding, or only explicit Polylogue `judge`/`assertion` rows authored by `user:local`?

[proposal] For source raw bytes: should they break the closed-loop predicate for all claims, or only for claim kinds they can actually support? I strongly recommend the latter.

[proposal] Should accepted user judgments default to `inject:false` and require a second explicit injection decision, or should `judge --accept` optionally carry `--inject`?

[proposal] Should the provenance verdict live in `user.db` as durable safety state, or in a derived tier with fail-closed behavior when missing? I lean derived verdict plus durable policy mutation when quarantined.

[proposal] What is the first supported external grounding class: git commit/tree, GitHub PR/issue API snapshot, local source raw bytes, or human judgment? I would start with human judgment + git/tool-result because those close the recovery-digest class fastest.

## What’s missing

[evidence] I used the uploaded 2026-07-05 bundle and the extracted code snapshot; live code may have moved, especially after the later bead-set merges mentioned in the transcript. Verify line anchors locally before filing or implementing. fileciteturn3file2

[evidence] I did not run the daemon, inspect a live `user.db`, or test actual assertion evidence refs against the 38 GB archive.

[evidence] The current code snapshot does not appear to have a first-class `raw:<id>` public ref accepted by `normalize_public_ref_text`; existing transform evidence converts raw-derived positions into session/message/block evidence refs. Code anchors: `polylogue/core/refs.py:8-40`, `polylogue/insights/transforms.py:114-137`, `polylogue/storage/sqlite/archive_tiers/user_write.py:1122-1128`.

[proposal] The SQL above assumes a resolver/materialized `assertion_citation_anchors` table. Without that table, a weaker scheduler-side check can still force all non-user agent claims to `candidate + inject:false`, but it cannot distinguish “agent said X” from “tool/git/raw evidence proves X” with enough precision.
