# 161. polylogue-rsad — MCP agent ergonomics: oversized responses, boilerplate affordances, metadata-only summaries

Priority/type/status: **P2 / bug / open**. Lane: **06-agent-context-coordination**. Release: **D-agent-context-coordination**. Readiness: **needs-acceptance-criteria**.

## What the bead says

Field report from a Sinex-side agent doing design archaeology over the archive (2026-07-06). The MCP surface fought the agent at every step; each item below is a concrete reproducible friction:

1. Affordance boilerplate dominates small responses: an EMPTY search result (hits: []) returned ~6KB of action_affordances — the affordance catalog rides every response instead of being a capability clients fetch once. Result: even trivial queries blow past agent token limits or waste context.
2. get_messages with limit=2 returned 375KB (claude-ai session 142a482e): full text + blocks of giant messages with no truncation/word-cap parameter honored at the message level. Agents need max_chars-per-message or excerpt mode on get_messages (list-level max_words exists but not here).
3. get_session_summary returns METADATA ONLY (id/title/count/actions) — the name promises a content summary; either rename (get_session_meta) or make it summarize.
4. list_sessions sort=started_at -> hard error 'QuerySpecError' with no hint of valid sort values; error detail is just the exception name.
5. list_sessions returns DUPLICATE items (same session id repeated up to 9x in one page — observed on aistudio-drive exocortex listing; presumably one item per match/branch, undocumented and sorted-confusing).
6. Multi-word search query with origin filter returned 0 hits where per-word substring (contains) clearly matches — AND-semantics or tokenization is too strict, and nothing in the response explains why (no per-term hit counts).

## Steps to Reproduce
Each numbered item above names its call shape; 1/2/4/5 reproduce against the live archive as of 2026-07-06.

## Acceptance Criteria
Affordances become opt-in (parameter or separate tool) or one-line refs; get_messages gains per-message truncation honored for role-filtered reads; get_session_summary either summarizes or is renamed; sort errors enumerate valid values; list results deduplicate by session id (or document the multiplicity); search responses carry per-term diagnostics when hits=0. An agent should be able to do the archaeology workflow (find design chats by keyword across origins, skim user messages) in <10 calls without any oversized-response fallback.

## Acceptance criteria gap

This active bead lacks acceptance criteria in the export. Add checkable acceptance criteria before coding unless this packet explicitly supplies a temporary gate.

## Static mechanism / likely defect

MCP responses can flood agent context with large payloads and repetitive boilerplate. The fix is ergonomic and safety-critical: metadata-first summaries with continuation handles.

## Source anchors to inspect first

- No precise source anchor was localized in this static pass. Start from the bead description and repository search.

## Implementation plan

1. Add a response-budget policy to MCP read tools: max bytes/items/tokens, with metadata-only fallback.
2. Return continuation handles for large results and explicit next-read/open instructions.
3. Strip repeated boilerplate; include concise schema/fields only on demand.
4. Test prompts and client payloads with realistic large sessions.

## Tests to add

- Oversized query returns summary + continuation, not full payload.
- Continuation reads exact next page.
- Small responses remain direct.
- No boilerplate repeats across common tool calls.

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools test tests/unit/coordination tests/unit/mcp -q`

## Pitfalls

- Do not add direct context injection or trusted memory writes outside the scheduler/candidate policy.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
