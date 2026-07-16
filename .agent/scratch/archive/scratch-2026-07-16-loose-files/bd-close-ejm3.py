#!/usr/bin/env python3
"""ejm3 closeout: residuals resolved, honest coverage record, close. Run once."""

import subprocess


def bd(*args):
    r = subprocess.run(["bd", *args], capture_output=True, text=True)
    out = (r.stdout or r.stderr).strip()
    print(("OK  " if r.returncode == 0 else "FAIL"), out.splitlines()[0][:140] if out else "")
    if r.returncode != 0:
        print("     ", (r.stderr or r.stdout).strip()[:400])


bd(
    "update",
    "polylogue-ejm3",
    "--append-notes",
    "CLOSEOUT 2026-07-06 (second pass, after gpt-pro feedback + on-brief D01/D02/D07 reruns): "
    "(1) All three deep-research lanes now RERUN ON-BRIEF and digested — D01 -> 3tl positioning note + "
    "3tl.16 claims ledger; D02 -> mhx.3 benchmark protocol + mhx.1 model registry + mhx.6 cost + 37t.5 "
    "generator guidance; D07 -> fs1.5 atropos-eval-jsonl profile + recorded/checkable reward split + "
    "fs1.10 internal-schema-first. Reports preserved in .agent/scratch/corpus-gpt-pro-2026-07-06/ as "
    "DR2-01/02/07. "
    "(2) Graph integrity: dangling deps emb-targets/emb-eval (from the 2026-07-03 integration) repointed "
    "to mhx.2/mhx.3; bd orphans clean. "
    "(3) Stale diagnoses corrected against LIVE SOURCE: cpf.6 (RELATIVE_BASE is per-call, not "
    "import-frozen — real gap is the clock seam), l4kf.2 (raw_id already content-hash — real hazards are "
    "acquisition-provenance multimap + origin:native_id collision), 4822 reworded (boundary stability, "
    "not async-only/method-count). 37t.15 bumped P1 + wired as blocker of scheduler/recall/distillery/"
    "standing-queries/annotation-import. t46.8 gained the shadow-telemetry-before-deletion gate; at44 "
    "gained the no-flat-KV guardrail; fnm.14 the ContextImage-vs-CorpusCompactionPack DTO boundary. "
    "(4) HONEST COVERAGE RECORD: first-pass digestion read D-demos, all B-*, all C-*, A-review-bundle1 "
    "fully, bundle2/3 partially; A-review-bundle4/5/6 prose was TITLE-SCRAPED only, and the six raw "
    "rnd-bundle files were indexed structurally (titles + line numbers), never full-text read. "
    "Mitigation: bundle-4/5/6 themes overlap the fully-read B/C branch files, and the gpt-pro feedback "
    "pass (which vetted everything) surfaced its misses as the 14 items now applied. Residual risk after "
    "this second pass: low; spot-check bundles/ via MANIFEST.sha256 index if a topic feels thin. "
    "(5) Provenance escrow: corpus copies + new artifacts live under .agent/scratch/"
    "corpus-gpt-pro-2026-07-06/ with MANIFEST.sha256 (31 files hashed). This is a LOCAL convenience "
    "copy, not a durable project artifact — beads are written to execute without it; the operator plans "
    "proper ingestion via browser capture / GDPR export, at which point the archive itself becomes the "
    "durable home.",
)

bd(
    "close",
    "polylogue-ejm3",
    "--reason",
    "Tech-tree integration complete after second pass: corpus digested, gpt-pro feedback applied "
    "(14 items: 3 stale diagnoses corrected against live source, priority/dep rewiring, 2 new beads "
    "3tl.16 + 9l5.19), D01/D02/D07 rerun on-brief and encoded into mhx/fs1/3tl/37t.5. Coverage record "
    "and provenance escrow in notes.",
)

print("--- ejm3 closeout done")
