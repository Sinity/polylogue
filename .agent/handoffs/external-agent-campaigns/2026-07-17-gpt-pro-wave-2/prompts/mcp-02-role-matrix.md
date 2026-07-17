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


---

## Context and authority

You are a long-running ChatGPT Pro analysis worker. A recent, complete
Polylogue project-state archive will be attached. Retrieve and inspect it
broadly; attachment size alone is not a reason to ignore evidence. This prompt
defines the question. The snapshot's current source, repository instructions,
complete relevant Beads records, and cited history are the evidence authority,
in that order when older plans drift.

## Working contract

- Investigate the actual source and tracker state before recommending changes.
- Separate observed facts, source-supported inference, unresolved uncertainty,
  and recommendation. Quote paths/symbols/Bead ids precisely but do not fill the
  report with copied source.
- Adjudicate contradictions and duplicates; do not create a parallel product
  model or generic architecture merely to make the report look complete.
- Translate findings into decision-ready actions: exact owning areas, ordering,
  acceptance criteria, falsification evidence, and what a local implementer
  should verify.
- Do not claim live browser, daemon, archive, deployment, or test evidence you
  cannot access.

## Deliverable

Create the exact `Result ZIP` named near the top under `/mnt/data/`. It must
contain `REPORT.md`, `EVIDENCE.md`, `DECISIONS.md`, and `NEXT-ACTIONS.md`.
Include compact machine-readable tables as JSON/CSV only when they add genuine
integration value. Do not copy the input archive into the result. Attach the
finished ZIP to the conversation through a working user-clickable link; files
left only in an internal temporary directory are not delivered.

Reopen and validate the ZIP, then report its SHA-256, size, and members. The
final chat answer must itself explain the important conclusions and decisions,
limitations, missing evidence, and the likely value of another iteration before
linking the package.

Do not perform an adversarial review unless explicitly requested. On an
ordinary **iterate/continue** request, preserve sound findings, resolve the
highest-value remaining uncertainty, and regenerate a complete package
revision. On an explicit **adversarial review** request, try to falsify the
prior report: seek contrary source/history evidence, unsupported certainty,
missed stakeholders/call sites, duplicate or incompatible designs, weak
acceptance criteria, and recommendations that do not survive current code.
Repair legitimate findings, regenerate the cohesive package, and report the
delta, residual disputes, and expected value of another pass.
