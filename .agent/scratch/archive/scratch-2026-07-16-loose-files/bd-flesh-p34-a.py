#!/usr/bin/env python3
"""Flesh out thin P3/P4 beads, batch A: jnj cluster + CLI-adjacent. All target fields verified empty. Run once."""

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
    "polylogue-jnj",
    "--design",
    "Rule-per-concern map the children implement: output dialect (jnj.3), read-ref semantics (jnj.4), "
    "demo/import split (jnj.6), vocabulary hygiene (jnj.7), runtime config surface (jnj.9), bare-root "
    "behavior (jnj.13). The shared invariant: one concern = one rule expressed identically across CLI, "
    "MCP, API, and daemon — no surface-local exceptions. Each child cites the exact file where its "
    "rule currently forks. Epic closes when no concern has surface-divergent behavior and the rules "
    "are written down in docs/cli-reference.md or a surface-algebra doc.",
)
label("polylogue-jnj", "horizon:mid")

bd(
    "update",
    "polylogue-jnj.3",
    "--design",
    "Anchors: polylogue/cli/query_output.py + query_output_contracts.py (query-side rendering), "
    "polylogue/cli/shared/formatting.py (plain/tty detection), scattered --json flags on verbs. "
    "Target dialect rule: --format {table,json,jsonl,md} as ONE root-level option consumed by every "
    "verb through a single render dispatch; --json survives as alias for --format json; --to/--out "
    "(file destination) is orthogonal to dialect and never implies a format change. Pitfall: new "
    "Click params on query verbs must go LAST (positional-shift trap).",
    "--acceptance",
    "Every read verb accepts --format with identical semantics; --json is an alias; format x "
    "destination are independent; output-contract schemas regenerated (devtools render "
    "cli-output-schemas). Verify: devtools test -k 'format or output' + one golden per dialect.",
)
label("polylogue-jnj.3", "horizon:frontier")

bd(
    "update",
    "polylogue-jnj.4",
    "--design",
    "Anchor: polylogue/cli/read_view_handlers.py (read --view semantics) vs the direct 'read "
    "session:REF' path in cli/query_group.py — the direct path bypasses read-view profile "
    "resolution, so the same ref renders differently depending on invocation shape. Fix: route "
    "direct ref reads through the same read-view resolver (profile selection, fold budgets, "
    "variant handling) so 'read session:X' == 'find id:X then read' output-identical.",
    "--acceptance",
    "Direct ref read and query-then-read produce byte-identical output for the same session and "
    "view; read-view profiles apply on both paths. Verify: devtools test -k read_view + a golden "
    "comparing both invocations on the demo corpus.",
)
label("polylogue-jnj.4", "horizon:frontier")

bd(
    "update",
    "polylogue-jnj.6",
    "--design",
    "Rule: 'import' ingests real user data; 'demo' owns synthetic/showcase flows entirely "
    "(polylogue demo seed/verify already exist). Migrate import --demo into demo seed (alias with "
    "deprecation window is acceptable ONLY if one release; prefer clean cut per surgical-renewal "
    "doctrine). Anchors: cli/commands/ import + demo command modules; docs/cli-reference.md regen.",
    "--acceptance",
    "import has no --demo flag; demo seed covers the flow; docs + output schemas regenerated; no "
    "other surface (MCP/daemon) references import-demo. Verify: devtools render all --check + "
    "devtools test -k demo.",
)
label("polylogue-jnj.6", "horizon:frontier")

bd(
    "update",
    "polylogue-jnj.7",
    "--design",
    "The provider->origin retirement's CLI-help slice: rg for provider-vocabulary tokens in "
    "user-visible help/error strings (cli/ commands, click option help=, UsageError text). Origin "
    "is the public vocabulary (core/enums.py Origin); provider tokens are wire-boundary only. "
    "Related: 9e5.8 owns the full retirement sequencing; this bead is ONLY the help/error-string "
    "surface so it ships independently.",
    "--acceptance",
    "No provider-family token appears in polylogue --help output tree or UsageError messages where "
    "an origin token is meant; docs/cli-reference.md regenerated. Verify: a help-tree grep script "
    "run in CI-able form + devtools render all --check.",
)
label("polylogue-jnj.7", "horizon:frontier")

bd(
    "update",
    "polylogue-jnj.9",
    "--design",
    "Today runtime/deployment config is env-var folklore (POLYLOGUE_ARCHIVE_ROOT, "
    "POLYLOGUE_FORCE_PLAIN, worker counts, pytest paths...). Target: one documented settings "
    "surface — polylogue config list/get/set backed by config.py's 5-layer resolution, showing "
    "WHICH layer wins per key (resolver-explain, same shape as w8db's DB prefs). Env vars stay "
    "authoritative for deployment; the surface makes them discoverable + explains precedence. "
    "Anchor: polylogue/config.py (inventory-driven diagnostics already exist there).",
    "--acceptance",
    "config list shows every recognized key, its value, and winning layer; unknown-key set warns; "
    "docs page generated from the same inventory (no hand-maintained table). Verify: devtools test "
    "-k config + render all --check.",
)
label("polylogue-jnj.9", "horizon:mid")

bd(
    "update",
    "polylogue-jnj.13",
    "--design",
    "Bare 'polylogue' currently prints help via the strict command floor (cli/query_group.py "
    "_bare_root_error_message handles bare WORDS; bare NO-ARGS shows Click help). Target triage "
    "surface instead: archive status one-liner (daemon fresh? converged?) + five most recent "
    "sessions (id, origin, title, age) + the three most useful next commands. Keep it fast "
    "(<200ms: one indexed query, no insight loads) and plain-safe.",
    "--acceptance",
    "Bare invocation renders triage in under 200ms on the live archive; falls back to help text "
    "when no archive exists; strict-floor bare-word behavior unchanged (polylogue foo still "
    "UsageError). Verify: devtools test -k bare + timing spot-check.",
)
label("polylogue-jnj.13", "horizon:frontier")

bd(
    "update",
    "polylogue-7le",
    "--design",
    "The three session->HTML paths to consolidate: (1) polylogue/rendering/core_messages.py + "
    "rendering/blocks.py (canonical block/message renderers), (2) daemon web shell "
    "(polylogue/daemon/web_shell*.py) which re-renders for the SPA, (3) the CLI read/export HTML "
    "view path (read_view_handlers.py --view html lane). Target: rendering/ is the single "
    "block->HTML authority; web shell and CLI consume it via ProjectionSpec x RenderSpec; "
    "bby.11's webui v2 then inherits one renderer. Do AFTER bby.11 stack decision to avoid "
    "renovating a surface scheduled for replacement — coordinate scope with bby.11's scaffold.",
    "--acceptance",
    "One HTML rendering entry point; web-shell and CLI HTML outputs diff-clean vs before (or "
    "intentionally improved with goldens updated); no duplicated block-type dispatch tables "
    "remain. Verify: devtools test -k 'render or html' + golden diffs.",
)
label("polylogue-7le", "horizon:mid")

print("--- batch A done")
