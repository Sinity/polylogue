Title: "WebUI v2 vertical: cost/usage explorer that keeps reported, derived, priced, and estimated lanes distinct"

Result ZIP: `webui-06-cost-usage-r01.zip`

## Mission

Build the cost/usage explorer for WebUI v2 (scaffold interface per webui-01;
state assumptions if its result is absent). Polylogue's cost model is
deliberately multi-lane and the UI must NOT collapse lanes into one number —
that collapse is a named anti-goal (`docs/cost-model.md`, read it fully).

Ground truth and traps (all verifiable in the snapshot):

- Authoritative token/usage rows: `session_model_usage` and provider-usage
  events. Bead `polylogue-r7p6`: `session_profiles` token columns undercount
  Codex 1000× — never source from profiles.
- Codex `input` INCLUDES cached tokens (~96% of it) and `output` includes
  reasoning; disjoint-lane billing or you double-count (this bug once caused
  7.69× cost inflation — commit `3938bc6c2` history).
- Claude subscription semantics: cache reads effectively free on Max/Pro;
  `cost_usd` is API-list-equivalent and overstates subscription spend — the
  UI should offer both "API-equivalent" and "subscription-credit" views
  where the data model exposes them.
- Bead `polylogue-f2qv.6` (open): profiles/costs not yet reconciled to exact
  provider usage — your UI must render the reconciliation state rather than
  pretending agreement.

Deliver:

1. Overview: spend/tokens by time bucket × origin × model, with lane
   selector (reported $ / catalog-priced $ / estimated) and an explicit lane
   legend explaining what each lane means (copy semantics from
   docs/cost-model.md, don't paraphrase loosely).
2. Cache economics view: cache-read share per model/origin over time, cache
   hit ratio, "what cached-token accounting would look like if billed naively"
   toggle — this is the view an agentic-cost-obsessed reader inspects first;
   make it exact and beautiful.
3. Session drill-down: per-session usage panel (API calls, token lanes,
   models used, per-model rows) linking into the webui-02 read page.
4. Honest absence: sessions/origins lacking usage evidence render as
   "no usage evidence", never $0. Aggregates must state coverage (N of M
   sessions carry usage rows).
5. SSR first page + islands; server-computed aggregates ONLY (client never
   sums rows — coverage/qualification must come from the server); Vitest +
   Python route tests including a coverage-qualification regression test.

## Constraints

- Continuation-paged JSON per the shared QueryTransaction vocabulary
  (`archive/query/transaction.py`).
- If an aggregate endpoint you need is missing server-side, add the smallest
  daemon route over existing operations/repository methods — no new SQL in
  the web layer; check `api/archive.py` cost/usage verbs first.
- Zero CDN; sanitized fixtures.

## Deliverable emphasis

HANDOFF.md: lane semantics table as implemented, JSON contracts, coverage/
qualification mechanism, screenshots-in-words of the cache-economics view,
f2qv.6 interaction notes (what changes when it lands), superseded web_shell
files, decisions the integrator could overturn.
