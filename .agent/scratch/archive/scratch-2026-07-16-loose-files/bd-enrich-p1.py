#!/usr/bin/env python3
"""Surgical enrichment of shallow ready-P1 beads: file anchors + verify commands. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:120] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:300])


bd(
    "update",
    "polylogue-37t.15",
    "--append-notes",
    "2026-07-06 anchors (verified live): the chokepoint is polylogue/storage/sqlite/archive_tiers/"
    "user_write.py upsert_assertion — and GOOD NEWS vs the design caution: assertions have a SINGLE "
    "write path (rg shows upsert_assertion only in user_write.py + scenarios/corpus.py); there is no "
    "async storage twin to mirror for this fix. Entry surface to regression-test: polylogue/mcp/"
    "server_mutation_tools.py:140 blackboard_post -> api post_blackboard_note -> upsert_assertion. "
    "Verify: devtools test -k 'user_write or blackboard' plus a new test asserting agent-authored "
    "post lands CANDIDATE+inject:false and a judged-rejected row is not resurrected.",
)

bd(
    "update",
    "polylogue-s7ae.6",
    "--description",
    "Commit 32ff31651 (coordination substrate, ~1376 LOC) merged with only verify --quick + focused "
    "tests green; the full devtools verify was aborted at 74% with scattered unclassified failures. "
    "Until each failure is classified coordination-caused vs pre-existing, the deploy gate for s7ae "
    "stays closed — unclassified inherited failure state is exactly what the verification doctrine "
    "forbids shipping on.",
)

bd(
    "update",
    "polylogue-0v9p",
    "--append-notes",
    "2026-07-06 anchors: detected language facts are DERIVED (rebuildable) -> index-tier DDL in "
    "polylogue/storage/sqlite/archive_tiers/index.py + an insights/registry.py descriptor for the "
    "rollup surface; operator language preferences/corrections are DURABLE -> user.db (the at44/w8db "
    "settings lane, or an assertion kind if per-object). Candidate detector: lingua or fasttext-lid "
    "at block grain, batch during convergence (a ConvergenceStage like insights). Verify: devtools "
    "test -k language plus one live-archive spot query showing per-block lang + confidence on a "
    "known Polish/English mixed session.",
)

bd(
    "update",
    "polylogue-arso",
    "--append-notes",
    "2026-07-06 anchors: durable variant storage -> numbered additive migration under polylogue/"
    "storage/sqlite/migrations/user/ (variants are operator-valuable transformed content, "
    "user-tier per the durability axis) + DDL constants in polylogue/storage/sqlite/archive_tiers/"
    "user.py; typed models near polylogue/core/ (follow AssertionKind pattern: literal_check embeds "
    "the closed vocab into SQL); ref resolution in polylogue/core/refs.py (ObjectRefKind is the "
    "closed vocabulary to extend — variant:/variant-node:); read/write methods on the repository "
    "mixins polylogue/storage/repository/. Registration traps memory applies if MCP/CLI surfaces "
    "are added. Verify: devtools test -k variant + migration roundtrip via devtools lab schema "
    "roundtrip.",
)

bd(
    "update",
    "polylogue-83u.3",
    "--append-notes",
    "2026-07-06 anchors: extension source is browser-extension/ (MV3, manifest.json at root); "
    "receiver/spool path is polylogue/daemon/http.py POST routes + the browser_capture spool "
    "writer. Option (a) upload-body interception happens in the extension service worker "
    "(webRequest/fetch hook); option (b) re-fetch happens receiver-side with the page session's "
    "auth constraints — document the MV3 service-worker lifecycle limits before choosing. Verify: "
    "an end-to-end capture of a session with an uploaded file lands attachment bytes with real "
    "SHA-256 + acquisition_status (the #2469 write path), byte_count > 0.",
)

bd(
    "update",
    "polylogue-d1y",
    "--append-notes",
    "2026-07-06 anchors: CLI command lands under polylogue/cli/commands/ (new hooks.py; init.py "
    "shows the settings-file-touching pattern); hook event names/coverage doc at docs/hooks.md; "
    "liveness monitoring reads hook-event arrivals source-side (source.db hook events tables — see "
    "artifact_taxonomy/runtime.py). Verify: devtools test -k hooks; manual: hooks install "
    "--dry-run on a copy of settings.json is idempotent (second run zero diff), hooks status "
    "reports wired-vs-observed per event.",
)

bd(
    "update",
    "polylogue-pj8",
    "--append-notes",
    "2026-07-06 anchors: prompt registration in polylogue/mcp/server_prompts.py (existing prompt "
    "plumbing — extend, do not invent); the registration-traps memory applies: EXPECTED_TOOL_NAMES "
    "analog for prompts + contract + render openapi/cli-output-schemas regen if schemas change. "
    "Skill recipes land in the repo skills dir consumed by harness config. Verify: devtools test "
    "-k prompt + MCP discovery test listing the ~6 intent prompts; then one live agent session "
    "using resume-context end-to-end.",
)

print("--- P1 enrichment done")
