#!/usr/bin/env python3
"""Create the two new beads from feedback + close out ejm3. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:140] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:300])
    return out


# --- public claims ledger under 3tl ---
bd(
    "create",
    "Public claims ledger: every README/launch claim carries a status and an evidence ref",
    "--parent",
    "polylogue-3tl",
    "--type",
    "task",
    "-p",
    "2",
    "-l",
    "area:legibility,horizon:frontier,tech-tree",
    "-d",
    "Turn radical honesty into a product surface: every public claim (README, docs site, launch post, "
    "category one-liner) must be exactly one of proven (backed by a finding/proof artifact), capability "
    "(code exists, no measured-result claim), aspirational (roadmap only), or retired (no longer true). "
    "This is the discipline that keeps the flight-recorder positioning from becoming marketing fog — the "
    "product whose pitch is 'every metric resolves to bytes' cannot itself ship unresolvable claims. "
    "Complements 3tl.12 (README de-persuasion pass) by making the honesty machine-checkable instead of "
    "a one-time edit.",
    "--design",
    "A docs/claims.yml ledger (or user-tier finding-backed equivalent once rxdo.4 FINDING lands): "
    "claim id, text, status, evidence ref (finding id / artifact path / measurement), last-verified date. "
    "README/docs quantitative claims link to a ledger entry by id. CI lint: a quantitative or comparative "
    "public claim without a ledger ref fails; a ledger entry with status=proven whose evidence ref does "
    "not resolve fails. Upgrade path: ledger entries become user.db findings once analysis provenance "
    "(rxdo) exists, so public claims share the same lifecycle as internal findings.",
    "--acceptance",
    "claims.yml exists and covers every quantitative/comparative claim in README + docs site; CI gate "
    "rejects unreferenced claims; each status has at least one real entry or an explicit none; the "
    "flight-recorder category claim itself is ledgered (initially capability, not proven). "
    "Verify: the CI lint run + a grep sweep of README claims against ledger ids.",
)

# --- thinking-vs-doing drift measure under 9l5 ---
bd(
    "create",
    "Thinking-vs-doing drift: experimental coverage-gated measure of reasoning share vs tool-active share",
    "--parent",
    "polylogue-9l5",
    "--type",
    "task",
    "-p",
    "3",
    "-l",
    "area:analytics,horizon:mid,tech-tree",
    "-d",
    "Early signal that a model got worse (or a harness got wasteful) for YOUR work: compare reasoning/"
    "thinking effort against tool-active time, trended by model family, repo, workflow shape, and month. "
    "Catches 'the model feels smarter but does less' and 'the upgrade inflated hidden reasoning cost' "
    "from the operator's own corpus before it congeals into a vague preference. Explicitly an "
    "experimental/suppressed measure — never a public quality score, never a composite productivity "
    "number (9l5.16 anti-goal applies).",
    "--design",
    "Candidate definitions, each emitted only where provider fields support it, else "
    "insufficient_evidence: thinking_token_share = reasoning_tokens/total_output_tokens (Codex output "
    "includes reasoning — see token-semantics memory; Claude thinking blocks where present); "
    "thinking_wall_share = model_thinking_duration_ms/session_wall_ms; tool_active_share = "
    "tool_duration_ms/session_wall_ms. Registered in the measure registry (9l5.7) with a MeasureSpec "
    "carrying coverage gates + construct-validity notes; trend surfaces ride 9l5.8 temporal analytics. "
    "Depends on activity_spans (9l5.13) for tool-active intervals.",
    "--acceptance",
    "Measure registered with coverage gate semantics (per-provider availability matrix); emits "
    "insufficient_evidence rather than fabricating where fields are absent; a trend query by model/"
    "month works over the live archive; no composite score surface. Verify: measure-registry tests + "
    "one live trend run.",
    "--deps",
    "blocks:polylogue-9l5.7,blocks:polylogue-9l5.13",
)

print("--- creates done")
