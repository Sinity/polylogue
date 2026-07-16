#!/usr/bin/env python3
"""Flesh out thin P3/P4 beads, batch C: audits, demos, infra + cpf descs. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:110] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:250])


def label(bid, *labs):
    for lab in labs:
        subprocess.run(["bd", "label", "add", bid, lab], capture_output=True, text=True)


bd(
    "update",
    "polylogue-9e5.5",
    "--design",
    "Method (the at44 pattern generalized — user_settings was found dead exactly this way): for "
    "every table across the five tiers, classify READ (rg for SELECT/FROM in polylogue/ excluding "
    "migrations/DDL), WRITE (INSERT/UPDATE/DELETE), and DDL-only. Output matrix: table x "
    "{read,write,ddl} x {runtime,test-only,none}. Dead = DDL exists, zero runtime read+write "
    "(user_settings-class); zombie = written never read (pure cost); mystery = read never written "
    "(depends on external writer — verify). Feeds a7xr kill list + schema-bump batching (60i5): "
    "dead-table drops ride the next same-tier migration window.",
    "--acceptance",
    "Committed matrix covers every CREATE TABLE across all five tiers; each dead/zombie row has a "
    "verdict (drop in next bump / keep with reason / wire like at44); re-runnable script so the "
    "matrix cannot rot. Verify: script re-run clean vs committed artifact.",
)
label("polylogue-9e5.5", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.6",
    "--design",
    "Census every digest producer/consumer: content-hash identity (core/hashing.py NFC-normalized "
    "session hash), blob store SHA-256 (storage/blob_store.py, raw_id), attachment hashes (#2469 "
    "path), embeddings recipe/chunk hashes, FTS nothing, backup manifests, dolt/beads external. "
    "For each: algorithm, normalization, what is INCLUDED/EXCLUDED (the session hash excludes user "
    "metadata BY DESIGN — tagging must not re-import), collision/migration story. Output: one doc "
    "table + a lint that new hashlib call sites must register (prevents ad-hoc hash flavors).",
    "--acceptance",
    "Committed census covers every hashlib/sha call site in polylogue/ (rg-verified count "
    "matches); each row states inclusion contract + consumer; the register-or-fail lint runs in "
    "devtools verify --quick or documented as follow-up. Verify: rg census diff clean.",
)
label("polylogue-9e5.6", "horizon:frontier")

bd(
    "update",
    "polylogue-9e5.7",
    "--design",
    "Map every long-lived daemon loop (convergence driver daemon/convergence.py, watcher/ingest "
    "loops in daemon/cli.py, fts_automerge.py, embedding catch-up, cursor-lag samplers, http server "
    "thread) against: which SQLite connection/lock class it holds, blocking vs async, backoff "
    "shape, and what starves it (the single-writer invariant means one hot loop can starve the "
    "rest). Output: a lock/starvation table + the top-3 starvation risks with reproduction "
    "sketches. Evidence-first: instrument with the existing daemon events/otlp tables rather than "
    "new machinery. Pre-read docs/retro/2026-05-24-1498-cascade.md (standing rule before touching "
    "convergence).",
    "--acceptance",
    "Committed table covers every loop the daemon spawns (enumerated from daemon/cli.py startup, "
    "cross-checked against live polylogued thread/task dump); each starvation risk has evidence or "
    "an explicit not-reproducible note. Verify: artifact + one live daemon observation session.",
)
label("polylogue-9e5.7", "horizon:frontier")

bd(
    "update",
    "polylogue-s8q",
    "--design",
    "Parked P4 while prod polylogued.service is inactive on sinnix — the trust problem: after a "
    "deploy, nothing proves the running daemon matches the repo state (version, schema versions, "
    "config) and recent captures are queryable. Shape when unparked: polylogue readiness gains a "
    "deploy-attestation view (running commit/version, tier schema versions vs expected, last "
    "ingest per origin within SLO); sinnix service unit exports version metadata; a post-switch "
    "check in the sinnix deploy flow queries it. Anchors: storage/archive_readiness.py + "
    "daemon/http.py status routes + the sinnix module owning polylogued.service.",
    "--acceptance",
    "(Provisional until unparked) After a deploy, one command reports version-match + "
    "schema-match + fresh-ingest-per-origin; a deliberate version skew is detected. Verify: "
    "readiness test + one real deploy observation.",
)
label("polylogue-s8q", "horizon:vision")

bd(
    "update",
    "polylogue-2jj",
    "--design",
    "Real closed issues as a coding-agent benchmark: sample N closed polylogue GH issues with "
    "verifiable outcomes (merged PR + tests), reconstruct the pre-fix repo state (base commit "
    "before the fix PR), and package issue text + repo ref + the fix PR's test as SpecCards "
    "(fs1.10 schema — internal schema first, adapters second per the D07 doctrine). The archive "
    "adds what SWE-bench lacks: the ORIGINAL agent sessions that solved each issue become "
    "reference trajectories (recorded_reward semantics from fs1.5). Leakage gate: agents "
    "evaluated on these must not have the fix in training/context — timestamp partitioning "
    "documented per card.",
    "--acceptance",
    "(Vision — no fabricated AC) Requires: fs1.10 SpecCard schema landed; a first hand-built "
    "card set (~10 issues) proving the reconstruction recipe; leakage policy written. States "
    "WHY: turns the repo's own history into an honest agent-eval asset no public benchmark "
    "provides.",
)
label("polylogue-2jj", "horizon:vision")

bd(
    "update",
    "polylogue-7fj",
    "--design",
    "Beads is already a Dolt DB with full history (.beads/, bd dolt). Ingest shape: a source "
    "family reading dolt commit history (or issues.jsonl git history as the cheap first pass) "
    "into a beads-issue origin — each issue's field-change timeline becomes queryable events "
    "correlated with sessions (which session created/closed/edited which bead — polylogued "
    "already tails agent sessions; join on time+repo). Unlocks: 37t.13 (beads<->assertions "
    "boundary), work-graph joins (1vpm), and 'what did the backlog look like when session X ran'. "
    "Decide grain deliberately: issue-state snapshots per change, not just current state.",
    "--acceptance",
    "bd issue history for this repo ingests as sessions/events with stable native ids "
    "(idempotent re-ingest); one join query answers 'sessions that touched bead X' on the live "
    "archive. Verify: devtools test -k beads + one live correlation query.",
)
label("polylogue-7fj", "horizon:mid")

bd(
    "update",
    "polylogue-212.5",
    "--design",
    "The reflexive capture proof: run an agent session ABOUT polylogue while browser-capture + "
    "hooks record it, then produce the archive's account of that same session (timeline, tool "
    "calls, cost, claims) as a Demo Finding Packet (212.7 contract). The packet juxtaposes what "
    "the agent claimed in-session vs what the archive recorded — the honest-mirror demo. All "
    "substrate exists (capture e2e verified 2026-06-29; hooks channel live); this is composition "
    "+ writeup, gated only by 212.7's packet shape.",
    "--acceptance",
    "A committed packet under .agent/demos/ where the recorded session's evidence (tool timing, "
    "exit codes, cost) annotates the session's own narrative; regeneration instructions work "
    "cold. Verify: packet passes the 212.7 shape check + cold-reader gate.",
)
label("polylogue-212.5", "horizon:mid")

bd(
    "update",
    "polylogue-5en",
    "--design",
    "Branch-local devloop exists for the daemon (dev-loop payload in daemon/http.py "
    "_dev_loop_payload; devloop-review warns on stale run dirs). Remaining surfaces to verify or "
    "wire: web shell (does the branch daemon serve branch web assets?), browser extension "
    "(MV3 reload against a branch receiver port), MCP (branch server instance without clobbering "
    "the prod-registered one). Deliverable: a table of surface x branch-isolation-status + the "
    "gaps wired or beaded. Known trap to encode: the branch daemon serves STALE code from the "
    "old run dir after switching branches — restart required (devloop-runtime memory).",
    "--acceptance",
    "Each surface (daemon/web/extension/MCP) has a documented branch-local recipe that two "
    "concurrent branches can run without cross-talk; stale-run-dir detection warns in at least "
    "the daemon + web cases. Verify: two-branch smoke run.",
)
label("polylogue-5en", "horizon:frontier")

bd(
    "update",
    "polylogue-6bu",
    "--design",
    "Docs-site failure modes seen live: pre-push render all --check rebuilds gitignored "
    ".cache/site so a stale cache breaks push (run render pages first — recorded gotcha), and "
    "PR #2500 repaired link rot. Make it a lane: link-integrity check (internal anchors + "
    "cross-page refs) over the rendered site, cache-invalidation rule documented, and the check "
    "wired where render all --check runs so drift fails fast instead of at push time.",
    "--acceptance",
    "Broken internal link fails the lane with the offending page:anchor named; stale-cache "
    "false-failures documented with the one-command fix; lane runs in verify --quick or "
    "render all --check. Verify: seed one broken link, watch it fail.",
)
label("polylogue-6bu", "horizon:frontier")

bd(
    "update",
    "polylogue-6l6",
    "--design",
    "Grab-bag polish set — split on claim if any slice grows: (a) docs theming pass (ui/theme.py "
    "tokens applied to docs site), (b) release-proof check (3tl.7 install matrix is the heavy "
    "half; this is the docs claim-consistency half — versions/commands in docs match pyproject), "
    "(c) control-plane doc currency (docs/devtools.md vs command_catalog.py drift — the "
    "doc-commands lint exists, extend to prose). Each slice is independent; none blocks the "
    "others.",
    "--acceptance",
    "Each slice either done or split to its own bead; docs claims about version/commands verified "
    "against live surfaces (render checks green). Verify: devtools render all --check + "
    "doc-commands lint.",
)
label("polylogue-6l6", "horizon:frontier")

# cpf.1/2/3 have design+AC but empty descriptions - add the WHY line
bd(
    "update",
    "polylogue-cpf.1",
    "--description",
    "WHY: TEXT timestamps in durable DDL re-introduce the exact ambiguity the four-time-kinds "
    "doctrine exists to kill (tz-unknown, lexicographic-vs-temporal sort divergence). A lint at "
    "DDL-review time is orders cheaper than a copy-forward migration later.",
)
bd(
    "update",
    "polylogue-cpf.2",
    "--description",
    "WHY: writer-class modules carry implicit invariants (single-writer, tier ownership, "
    "twin-sync) that new contributors/agents violate silently; a docstring convention + layering "
    "check makes the contract visible where the code is edited.",
)
bd(
    "update",
    "polylogue-cpf.3",
    "--description",
    "WHY: injected context (recall packs, preambles, assertions) is a prompt-injection surface — "
    "a deny-lexicon tripwire fixture set proves the trust boundary holds as the injection "
    "surfaces multiply (37t rollout makes this load-bearing).",
)

print("--- batch C done")
