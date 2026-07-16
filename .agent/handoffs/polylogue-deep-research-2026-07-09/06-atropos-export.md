---
created: "2026-06-28T00:00:00Z"
purpose: "Spec: export Polylogue agent traces in Nous Research Atropos ScoredDataGroup format"
status: "active"
project: "polylogue"
---

# Atropos Export Spec (`polylogue export atropos`)

## Context

Goal: emit Polylogue agent sessions in Nous Research **Atropos** RL-environment
data format so traces round-trip through Atropos's own JSONL/HTML/Gradio tooling.
Atropos is "a Language Model Reinforcement Learning Environments framework for
collecting and evaluating LLM trajectories." Scores come from Polylogue's
deterministic pathology detectors (`get_pathologies`) and work-event outcomes.

Sources:
- https://github.com/NousResearch/atropos
- https://github.com/NousResearch/atropos/blob/main/README.md
- https://github.com/NousResearch/atropos/blob/main/atroposlib/type_definitions.py
- https://github.com/NousResearch/atropos/blob/main/atroposlib/envs/base.py
- https://github.com/NousResearch/atropos/blob/main/atroposlib/envs/README.md
- https://pypi.org/project/atroposlib/

---

## 1. Atropos data format — `ScoredDataGroup` (verbatim schema)

The core training-data unit is `ScoredDataGroup`, a `TypedDict` (one *group* =
one prompt with N candidate rollouts; the JSONL writer emits one group per line).

```python
class ScoredDataGroup(TypedDict):
    tokens: List[List[int]]                                  # per-member token ids
    masks: List[List[int]]                                   # per-member loss mask (-100 / token-id convention)
    scores: List[float]                                      # per-member scalar reward
    advantages: Optional[List[List[float]]]                 # per-token advantages
    ref_logprobs: Optional[List[List[float]]]
    messages: Optional[List[List[Message]]]                 # per-member OpenAI-style message list
    generation_params: Optional[Dict[str, Any]]
    inference_logprobs: Optional[List[List[float]]]
    group_overrides: Optional[Dict]
    overrides: Optional[List[Dict]]
    images: Optional[Any]
    distill_token_ids: Optional[List[List[List[int]]]]
    distill_logprobs: Optional[List[List[List[float]]]]
```

Supporting types (`atroposlib/type_definitions.py`):

```python
number = int | float

class Message(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: Content                  # str (or multimodal content blocks)
    reward: Optional[float]

Item = Any
```

Key structural facts:
- **All top-level fields are parallel lists indexed by group member** (rollout
  candidate). `len(tokens) == len(masks) == len(scores) == N`.
- **`tokens`/`masks`/`scores` are the only hard-required fields** for the RL
  trainer path. Everything else is `Optional`.
- **Environments own tokenization** — Atropos does not tokenize for you; each env
  produces its own `tokens`/`masks`. (README/envs README: "Environments are
  responsible for tokenization … flexibility to assign token-level rewards.")
- **`messages` is the human-readable / multimodal carrier**: a per-member list of
  OpenAI-format `Message` dicts. When `config.include_messages` is set and
  `messages is None`, `base.py` auto-fills it by decoding `tokens`:
  ```python
  group["messages"] = [
      [{"role": "user", "content": self.tokenizer.decode(group["tokens"][i])}]
      for i in range(len(group["tokens"]))
  ]
  ```
  i.e. messages can be supplied directly and Atropos will *not* overwrite them.

### Serialization / save path
- `process` subcommand does inference-only rollouts and, with
  `--env.data_path_to_save_groups out.jsonl`, writes **one JSON-serialized
  `ScoredDataGroup` dict per line** (`jsonl_writer.write(group)` in the
  `handle_send_to_api` path). It also emits a sibling `out.html`.
- Saved per line = the dict above (Optional fields present as `null`/omitted).

### Viewer / round-trip surface
- `process` auto-generates `out.html` (browser-viewable, no server).
- `view-run` launches a **Gradio UI** to inspect batches of rollouts.
- The README does not formally enumerate viewer-minimal fields, but the HTML/
  Gradio views render **prompts, responses, and scores** — i.e. they read
  `messages` (or decoded `tokens`) + `scores`. For a *human round-trip viewer*
  the load-bearing fields are **`messages` + `scores`**; for the *trainer*
  round-trip they are **`tokens` + `masks` + `scores`**.

---

## 2. Round-trip: minimum viable group

Two fidelity tiers:

| Tier | Required fields | Consumer |
|------|-----------------|----------|
| **Viewer** (our default) | `messages` (list[list[Message]]), `scores` (list[float]); plus `tokens`/`masks` as `[]`/`null` placeholders to satisfy schema typing | HTML / Gradio `view-run` |
| **Trainer** | `tokens`, `masks`, `scores` (real token ids + loss mask) | RL trainer pull |

Polylogue has **no token ids and no loss masks** — sessions are stored as text +
typed blocks, never tokenized, and we are not an RL rollout producer. So the
honest target is the **Viewer tier**: emit real `messages` + derived `scores`,
and leave `tokens`/`masks` empty (`tokens: [], masks: []` or `null`). This is
schema-valid (Atropos "does not validate cross-field semantics beyond schema
typing"; trainers validate alignment themselves). A `--tokenize` opt-in (see §4)
can later populate `tokens`/`masks` via a HF tokenizer for trainer-tier output.

---

## 3. Polylogue → Atropos field mapping

### Group construction choice
A Polylogue session is a **single trajectory**, not a sampled group of N
candidate rollouts. Two valid mappings:

- **(A) one-session-per-group, N=1** (recommended default): each group has
  exactly one member = the session's full turn sequence. `scores=[session_score]`,
  `messages=[[...turns...]]`. Clean, lossless, and matches "trace round-trip".
- **(B) lineage-as-group**: use `get_logical_session` / topology so a
  fork/resume/subagent family becomes one group with N members (the sibling
  branches), each scored independently. Richer "compare candidate rollouts" view
  but depends on lineage normalization (#2467). Defer to a `--group-by lineage`
  flag.

Default = (A).

### Messages mapping (`get_messages(session_id)` → `list[Message]`)

`MessageRecord` (`polylogue/storage/runtime/archive/records.py`) → Atropos
`Message`:

| Atropos `Message` | Polylogue source | Notes |
|-------------------|------------------|-------|
| `role` | `MessageRecord.role` (`Role`) | map to `{system,user,assistant,tool}`. Polylogue roles normalize; `tool`/tool-result blocks → `"tool"`. Unknown/protocol rows (`material_origin`/`message_type`) optionally filtered. |
| `content` | `MessageRecord.text` (+ `blocks` for tool_use/thinking) | concatenate text; optionally serialize tool-call blocks into content or a follow-on `tool` message. |
| `reward` | usually `None` at message level | per-token/per-message reward not modeled; leave `None`. Could carry a per-turn pathology hit later. |

Practical: pull via `Polylogue.get_messages` / `get_messages_paginated`; honor
`material_origin` to drop generated/context-pack protocol rows from the authored
transcript (mirrors existing authored-user accounting).

### Scores mapping (the RL signal)

`scores: List[float]` — one scalar per group member. Derive a single session
score in **[-1.0, +1.0]** (Atropos convention is signed/normalized reward;
trainers re-normalize per-group). Composition:

1. **Pathology penalty** (`get_pathologies` → `PathologyReport.findings`,
   `polylogue/insights/pathology.py`). Each `PathologyFinding` has
   `kind ∈ {wasted_loop, missed_review, stale_context}`, `severity ∈
   {low,medium,high}`, `occurrence_count`. Map severity→weight
   (`low=0.1, medium=0.25, high=0.5`), sum `weight*occurrence` capped, subtract.
2. **Work-event outcome** (`SessionWorkEventInsight`,
   `polylogue/insights/archive.py:297`; run projection statuses in
   `run_projection.py`: `RunStatus`/`ObservedEvent.kind` like
   `tool_finished`, `check_failed`, `test_failed`). Reward terminal success
   (run `status == completed`, tests passing) positively; failed/abandoned
   negatively. `find_abandoned_sessions` / `find_stuck_sessions` give negative
   signal too.
3. Default neutral score `0.0` when no signal.

Suggested formula (v1, documented + versioned like detectors):
```
score = clamp(
    base_outcome              # +1 completed, -0.5 abandoned/stuck, 0 unknown
    - pathology_penalty,      # sum severity-weighted, capped at 1.0
    -1.0, +1.0)
```
Emit the component breakdown into `group_overrides` (metadata) so the score is
auditable and reproducible — pathology detectors are deterministic and versioned
(`PATHOLOGY_DETECTOR_VERSION`), so the score is too.

### Metadata mapping (Atropos optional fields)
- `group_overrides`: `{polylogue_session_id, origin, score_breakdown,
  pathology_detector_version, exporter_version}` — provenance + reproducibility.
- `generation_params`: `{model_name}` from `MessageRecord.model_name` when
  homogeneous.
- `images`: omit (or wire `AttachmentRecord` image attachments later).
- `tokens`/`masks`/`advantages`/`*_logprobs`/`distill_*`: empty/`null` at Viewer
  tier.

### Gaps / honest limits
- **No token ids, no loss masks** — Polylogue never tokenizes. Trainer-tier
  output requires an opt-in tokenizer pass; Viewer tier is the truthful default.
- **No per-token rewards / advantages** — scores are session-level only.
- **N=1 groups** are degenerate for GRPO-style group-relative advantage (a group
  of one has no relative signal). That's fine for *trace viewing*; lineage groups
  (mapping B) are the path to multi-member groups.
- **Role fidelity**: Polylogue tool-use is block-structured; flattening to
  OpenAI `tool` role messages is lossy but viewer-adequate.

---

## 4. `polylogue export atropos` design

### Surface
CLI command (export folds into read/export surface per project intent;
implement as `polylogue export atropos` or `polylogue read --format atropos`).

```
polylogue export atropos [QUERY] \
  --origin <origin> --since --until --tag --repo \   # same scope filters as get_pathologies
  --output groups.jsonl \                            # JSONL of ScoredDataGroup, one/line
  --group-by session|lineage \                       # default: session (N=1)
  [--tokenize <hf-model>] \                           # opt-in trainer tier: fill tokens/masks
  [--include-protocol]                               # keep generated/context rows (default: drop)
```

### Output
- **`groups.jsonl`**: one `ScoredDataGroup` JSON object per line — directly the
  shape `process --env.data_path_to_save_groups` produces, so Atropos's own
  `view-run` Gradio UI and `.html` tooling consume it unchanged.
- Optional sibling `groups.html` could be generated by *running Atropos's own*
  HTML step, or we skip and rely on `view-run`.

### Scope (per-session vs whole-archive)
Whole-archive = stream all matched sessions, one group per line (memory-bounded,
paginated via `get_messages_paginated`). Per-session = single-line file. Same
code path; scope is just the filter set (reuse `MCPSessionQueryRequest` filters
that `get_pathologies` already uses) — so pathology scope and export scope align.

### Score derivation pipeline (per session)
1. `get_messages` → build `messages` list (filter protocol rows unless
   `--include-protocol`).
2. `pathology_report(scope=single session)` → penalty.
3. work-event/run-projection status → base outcome.
4. compose `score` (formula §3), record breakdown in `group_overrides`.
5. assemble `ScoredDataGroup` (Viewer tier: `tokens=[]`, `masks=[]`).

### Verification plan
- **Schema validity**: load each emitted line, assert it satisfies the
  `ScoredDataGroup` TypedDict (parallel-list lengths equal; `scores` floats in
  [-1,1]; `messages` roles in the allowed Literal set). Unit test in
  `tests/unit/` over a `corpus_seeded_db` / demo archive.
- **Determinism**: export twice over the same archive → byte-identical JSONL
  (pathology detectors + outcome are deterministic; assert it).
- **Round-trip (real Atropos)**: in a throwaway venv `pip install atroposlib`,
  run Atropos `view-run` (Gradio) against `groups.jsonl` and confirm messages +
  scores render; capture as manual/operator evidence (Atropos is not a repo dep).
  Cheaper automated proxy: re-load with `atroposlib`'s `ScoredDataGroup` import
  if exposed, else validate against a vendored TypedDict copy in tests.
- **Mapping correctness**: golden test — a fixture session with a known
  `wasted_loop` pathology must yield the expected negative score and the finding
  in `group_overrides.score_breakdown`.

### Sequencing / placement
- New semantics first: a substrate/insight helper
  `build_atropos_group(session_id, scope) -> ScoredDataGroup` (insights or a new
  `polylogue/export/atropos.py`), then the CLI surface adapts (project rule:
  substrate before surfaces). Vendor a minimal `ScoredDataGroup`/`Message`
  TypedDict for typing (don't take a runtime `atroposlib` dependency).
- Open an issue first (non-trivial, cross-module: messages + pathology + outcome
  + new export surface).
