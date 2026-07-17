Title: "Authority design: the write/judge/run/operate role matrix, confirm semantics, and destructive-operation gating for the six-tool MCP era"

Result ZIP: `mcp-02-role-matrix-r01.zip`

## Mission

Analysis/adjudication job (analysis contract; produce decision inputs, not a
patch). The six-tool MCP cutover collapses the default read surface, but the
role-gated families (write, judge, run, operate) need a coherent authority
design that today exists only as scattered mechanisms. Produce the
decision-complete role/authority matrix the cutover lane and the operation-
gateway program can implement without re-litigating.

Inventory first (all in the snapshot):

- Current role ladder: read ⊂ write ⊂ admin (`polylogue/mcp/server.py`
  role split); every register_* family in `server_tools.py`,
  `server_insight_tools.py`, context/mutation/personal-state/maintenance/
  coordination modules — classify each current tool into its six-tool-era
  home (read-tool projection / write family / judge family / run family /
  operate family / retired).
- Confirm gates: bead `polylogue-jn40` (interim confirm parameters on
  destructive tools — 10 tools across write/admin found unprotected in
  dogfood-2); the MCP confirm-gate asymmetry finding.
- The executable-mutation-authority direction: bead `polylogue-t46.9`
  (OperationSpec as the ONE mutation boundary: resolve → authorize →
  preview → bound confirmation → apply → receipt → postflight, across
  CLI/API/MCP/daemon/maintenance) and the Testsuite Diet architecture
  decision `04-destructive-and-authentication-boundaries.md` (in
  `.agent/handoffs/.../testdiet/context/testsuite_diet/architecture/` in the
  snapshot). Your matrix must be a projection of that decided contract, not
  a new policy system.
- The judgment lane: beads `polylogue-37t.12` (canonical judgment
  transaction), `41ow` (TOCTOU), and the assertion/candidate/injection
  model (`user.db` assertions, `context_policy_json`).

Deliver (documents, not code):

1. **The matrix**: for every write/judge/run/operate verb — object kinds,
   required role/capability, preview obligation (what a preview must show),
   confirmation binding (how confirm tokens bind to the previewed effect,
   not just a boolean), receipt content, idempotency/retry semantics,
   and failure states. Explicitly mark which verbs are destructive-
   irreversible (excision, deletion, live repair) vs reversible.
2. **Confirm semantics**: one uniform confirm protocol proposal (bound
   preview-hash confirm), its MCP tool-schema shape, and the migration from
   jn40's interim booleans.
3. **Gap census**: every current mutation path that BYPASSES the intended
   gateway (grep MCP/CLI/daemon for direct repository/store mutation calls;
   the t46.9 bead lists known bypasses) — table with file:line, verb,
   proposed home.
4. **Adjudication questions**: the ≤5 genuine operator decisions remaining
   (e.g. should `judge` be its own role or a write capability?), each with
   a recommended default and rationale.

## Deliverable emphasis

REPORT.md (the matrix + semantics), EVIDENCE.md (census with file:line
citations), DECISIONS.md (recommended defaults), NEXT-ACTIONS.md (exact
implementation order for the cutover lane and t46.9). Every claim cites
snapshot files/beads by path/id.
