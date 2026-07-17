Title: "Six-tool-era standing agent manual: regenerate the cold-start manual and continuity recipes for the replaced MCP surface"

Result ZIP: `mcp-01-agent-manual-r01.zip`

## Mission

Polylogue's MCP surface is being cut over from 103 tools to a six-tool
transaction surface (default read role: `query`, `read`, `get`, `explain`,
`context`, `status`; continuation as input/result-ref protocol; write/judge/
run/operate role-gated; URI resources for stable objects; MCP prompts for
workflows). That cutover is a LOCAL lane running in parallel (bead
`polylogue-t46.8` + children — read them). Separately, an external package
("beads-06", covering beads `polylogue-3gd.2`/`3gd.3`) already delivered an
executable agent-manual + installation kit against the 103-tool surface:
project-owned cold-start manual, typed recipe/capability declarations,
role-scoped MCP manifests, native installation for Claude Code / Codex /
Gemini CLI / Hermes, and verification lanes. Check whether its content is in
your snapshot under `.agent/handoffs/external-agent-campaigns/`
(`beads-06` results) or already merged into source; adjudicate and REUSE its
architecture — do not design a competing manual system.

Your job: produce the six-tool-era manual CONTENT and generation pipeline so
that when the cutover lands, agents get a correct, cache-stable, installable
manual with zero "fetch the manual first" turns:

1. **Manual content, generated from declarations**: normal invocation of
   each of the six tools with realistic argument examples; the continuation
   protocol (how to resume a truncated result — exact request shape);
   result-limit semantics and recovery; result-ref citation discipline;
   role ladder (what write/judge/run/operate add and how confirm gates
   work); URI resource addressing; source coverage (the 10 Origin tokens
   with one-line meaning each); continuity workflows (resume a session,
   forensic lookup, prior-art search, cost audit) as step-by-step recipes
   using ONLY the six tools. Every example must be derivable/compilable
   against declarations — follow the beads-06 pattern of verification lanes
   that resolve generated routes against production declarations.
2. **Query-language teaching**: a compact DSL section whose every example
   round-trips the real parser (`archive/query/expression.py`) — extract
   valid examples from repo tests; include the strict-command-floor rule and
   the three intent signals (find keyword / quoted expression / field
   syntax).
3. **Install/delivery**: reconcile with beads-06's SessionStart/native
   installation mechanisms — what changes for the six-tool era, what is
   untouched; per-client (Claude Code, Codex, Gemini, Hermes) deltas.
4. Tests: manual-compilation lane (examples resolve against declarations;
   parser round-trip for DSL examples), staleness check (manual regenerates
   deterministically; drift fails a `--check`).

## Constraints

- If the six-tool declarations are not yet in your snapshot, write the
  manual against the contract stated above and the t46.8.1 declaration
  models (`polylogue/mcp/declarations/` if present), parameterizing tool
  schemas so integration is mechanical; state exactly what must be
  re-generated post-cutover.
- The manual is public-repo content: no private paths/data in examples.

## Deliverable emphasis

HANDOFF.md: adjudication verdict on beads-06 reuse (what you kept/changed),
manual generation pipeline, the complete manual itself (rendered), per-client
install deltas, post-cutover regeneration checklist, and which parts are
blocked on cutover-final schemas.
