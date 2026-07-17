# 02. polylogue-37t.15 — Force non-user assertion writes through candidate/non-injected policy

Priority: **P1**  
Lane: **agent-write-safety**  
Readiness: **ready-now / code-local**

Depends on packet(s): polylogue-s7ae.6

## Why this is urgent / critical-path

The archive is about to let agents coordinate, post messages, and contribute memory. Non-user writes must not become trusted active context by default.

## Static diagnosis / likely mechanism

Root cause: `upsert_assertion` defaults `status=None` to `AssertionStatus.ACTIVE`, regardless of `author_kind`. `upsert_blackboard_note` passes `author_kind` through and does not provide a status, so agent-authored blackboard notes become active assertions. On conflict, `upsert_assertion` also overwrites `status` and `context_policy_json`, so a repeated non-user upsert can resurrect or re-inject a row.

Source anchors: `polylogue/storage/sqlite/archive_tiers/user_write.py:31-47` defines the ACTIVE default; `:641-671` blackboard note calls `upsert_assertion`; `:901-971` normalizes status/author/context and updates status/context on conflict; `:1245+` contains candidate judgment paths that should remain terminal.

## Implementation plan

Patch the single choke point: `polylogue/storage/sqlite/archive_tiers/user_write.py::upsert_assertion`.

Implementation shape:
1. Normalize `author_kind` before resolving status/context.
2. Add `_is_user_author(author_kind) -> bool`, initially exact normalized `"user"` only.
3. Fetch existing `status` and `context_policy_json`, not only `created_at_ms`.
4. For non-user authors:
   - if no existing row or existing row is still unjudged, force `status=CANDIDATE`;
   - force `context_policy.inject=false` and `context_policy.promotion_required=true`, overriding caller input;
   - if the existing row is terminal judged (`REJECTED`, `DEFERRED`, `SUPERSEDED`, `DELETED`, and probably `ACCEPTED` unless explicitly user-updated), preserve the existing status/context and do not let agent input revive it.
5. User-authored writes keep current behavior, including explicit active/injected assertions where caller policy allows it.
6. Do not add separate checks in MCP or blackboard handlers; they should remain clients of the invariant.

## Test plan

Add focused storage/API tests:
- agent `upsert_assertion(... author_kind="agent")` with no status lands as `CANDIDATE`, not `ACTIVE`.
- agent call with `status=ACTIVE` and `context_policy={"inject": true}` still stores candidate + non-injected.
- `post_blackboard_note(author_kind="agent")` mirrors to an assertion with candidate/non-injected policy.
- rejected candidate re-upserted by an agent remains rejected and non-injected.
- user-authored assertion remains active by default to avoid breaking current user-state flows.

## Verification command / proof

`devtools test tests/unit/storage/test_user_state_contracts.py tests/unit/storage/test_archive_tiers_assertions.py -k 'assertion or blackboard or candidate'`

## Pitfalls

Do not fix only `blackboard_post`; MCP, CLI, daemon, and future coordination messages must all inherit the policy from one place. Avoid a broad ontology change; this is a safety invariant, not a new assertion product.

## Files/functions to inspect or touch

- `polylogue/storage/sqlite/archive_tiers/user_write.py:31-47`
- `polylogue/storage/sqlite/archive_tiers/user_write.py:641-671`
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901-971`
- `polylogue/mcp/server_mutation_tools.py`
- `polylogue/api/user_state*`
