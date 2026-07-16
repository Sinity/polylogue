---
created: "2026-07-10"
purpose: "Fable (Claude) working notes for the 2026-07-10 strategy/verifiability dialogue with the codex agent. Personal scratchpad; positions + corrections + delegation analysis. Dialogue ledger: 2026-07-10-agent-dialogue.md"
status: "active"
project: "polylogue"
session: "claude-code-session:3347cf34-ca12-45ae-918f-781c7f96a704"
---

# Fable notes — 2026-07-10

## Session arc
1. Recon (grok doc + bd prime + gate board + dirty tree): frontier A-trust-floor 67%; wip = f2qv.2; dirty tree = complete-looking f2qv.2 impl (codex parser disjoint-lane cache subtraction + regression tests + cost-model docs) — NOT mine, do not touch.
2. Read codex strategy session `codex-session:019f49d8-0185-7c43-8793-db6e57db13e1` via prod MCP (worked; 220 msgs; ends in one-week plan, plan-mode, unexecuted).
3. Gave feedback on plan v1 (5 concerns: parallel meta-system/no-s7ae, 2-3x oversubscribed, v29 ungrounded, benchmark epistemics, cpf.5 temporal gap).
4. Investigated `polylogue agents` (s7ae coordination substrate, landed #2534 on 07-04): live demo showed value (identified me, f2qv.2 attribution 0.95, 3 codex peers) + warts (systemd as "build" resources, conductor-path handoff refs, harness launcher metadata ingested verbatim, huge payload).
5. Read codex writeup `.agent/scratch/2026-07-10-broad-project-strategy-and-verifiability.md`; verified its corrections; wrote dialogue entry [1]; drafted gpt-pro-sol prompts.

## Corrections to self (verified 2026-07-10)
- `cpf.5` CLOSED 2026-07-08 — my "verified live bug" was stale (grok-doc era). Lesson: bead status from memory files decays in ~48h in this repo; `bd show` before citing.
- `INDEX_SCHEMA_VERSION = 29` (index.py:36) — canonical moved 24→29 via #2588/#2597/#2607/#2622 while deployed archive stays v24. I flagged "v29" as possibly hallucinated; wrong half, but the operational half (no live rebuild mid-campaign) stood.
- `persist_session_commits` = placeholder no-op (session_commit.py:455) — codex claim confirmed.
- **Affordance-review misread (dialogue [3] concession):** I read only the first ~40 lines of the 07-09 review and presented the machine `kill` classification (59 MCP + 34 CLI) as the review's conclusion. Actual reviewed decisions: keep 56/59 + all 34; retire only `v6vy`; `moyt` consolidation pending parity. I committed the exact matrix-authority error the dialogue diagnoses. Lesson: never cite a review doc's headline without reading its decision table; generated labels ≠ adjudicated decisions.

## Dialogue state (as of entry [3])
- Consensus 1–15 recorded, pending operator ratification. Codex accepted 1–8 + vacuity name; I accepted A1 (census gate), A2 (composite content cursor — best design artifact of the dialogue), A3 (s7ae.1 successor), A4/10 (no broad retirement), 14 (planned_work internal-first), 15 (honest accounting naming).
- Pilot census ran (read-only FTS, during codex's backup window — light, different tier): claude-code 5,448 claim-msgs/1,456 sessions (3.7/sess); codex 3,321/210 (15.8/sess) → new methodological point: sampling-unit clustering; per-session cap or cluster-robust intervals required. Fed into prompt 02.
- Codex is executing a LIVE source-tier migration (002→003) with verified backup, operator-approved ("you should migrate"). I hold heavy archive reads until it reports completion.
- gpt-pro-sol now: 00 README (codex-amended), 01–05 (mine; 02/03 amended post-consensus; 05 codex-corrected), 06 strategy-falsification (codex), 10 cloud-lanes matrix (codex), sinex-* prompts (codex), polylogue-cloud/ executor packets 01–04 (codex) + LAUNCH.md (mine, operator-facing runbook).

## Quota-burst execution state (entry [4]/[5] era)
- Operating plan read (inspiration only); codex adjudicated its board stale (PRs 2626/2627 merged, b0b.1 closed via #2630).
- Codex completed live source migration v2→v3 (verified backup at /realm/data/captures/polylogue/backups/2026-07-10-source-v2-pre-v3/). Archive read constraints lifted.
- **My repo prep (operator-assigned):** bead polylogue-ooqh (execution-grade, claimed) → worktree /realm/tmp/worktrees/cloud-bootstrap → PR #2631 (setup.sh render-all fix + visible warning [codex review applied: cause-neutral phrasing] + pytest bounds + docs mirror). verify --quick green (39.4s). Lint green. Merge pending checks.
- bd-reimport-guard observed working live: worktree add clobbered fresh ooqh claim, guard auto-restored.
- First wave locked at 3 lanes (k6fm/kj22/v6vy) — consensus; C1 doubles as envelope benchmark (incl. testmon-seed timing).
- Consensus at 16 items (added: ≤2 claims/session Receipts sampling cap).
- DONE: PR #2631 merged (c68585b8b), bead ooqh closed, worktree/branches cleaned.
- **Eligibility census v0 (entry [7]):** gate PASSES — L2-eligible: claude 4,450 (1,912 after ≤2 cap), codex 2,713 (223 after cap); ≥40 floor cleared 5–30×. Cap discards 92% of codex candidates (clustering confirmed). Surprise: ~82% of claims have a verifier between last-edit and claim — the open question is verifier OUTCOMES, not verifier absence. Script: scratchpad/eligibility_census.py.
- Codex working browser control (operator note); division of labor holding.

## My standing positions (post-adjudication)
- Codex writeup is strong; consensus items 1–8 in dialogue ledger.
- Remaining pushbacks: (1) eligible-claim census before n=120 commitment; (2) labeler-family × origin bias reporting; (3) surface retirement (9e5.25/.26, 59 kill MCP tools) belongs in campaign — prune before proof-obligation enumeration; (4) campaign should dogfood cost/outcome accounting; (5) keep the wrong Hermes superset fixture as the unknown-future-version test case.
- Named pathology: **fixture/matrix vacuity** — 3 instances this week (closure matrix shape-checks, Hermes invented-superset fixture, a7xr.13 contract shadows). Anti-vacuity (delete real proof ⇒ report goes red) should be doctrine, not just a gap-compiler feature.

## Cloud-delegation matrix (drafted for operator, adjudication pending w/ codex)
Cloud coding agents (Claude Code Web / Codex Cloud; per docs/cloud-agents.md: synthetic fixtures only, POLYLOGUE_ARCHIVE_ROOT=/tmp, no real corpus, no /realm/data):
- SAFE: WorkflowProofSpec/gap-compiler MVP; kj22 (fuzz collection); k6fm (verify-run git metadata); s7ae envelope per-view allowlists + contract tests; actions ranked-pairing parity fixture; MCP surface retirement (9e5.25 mechanics: EXPECTED_TOOL_NAMES, contracts, render regen); web API metadata/preview split + tests vs demo seed; Playwright harness scaffolding vs demo-seeded daemon (journeys tuned locally later vs live-shaped scale).
- NOT CLOUD: Receipts extraction/census (real corpus, privacy); Hermes real-fingerprint capture (do locally, ship structural fingerprint file to cloud); live canaries; anything live-archive; coordination live proofs (s7ae.5).
- GPT-5.6 Pro (no execution; full chisel source+beads): design/spec/review/packet production → /realm/inbox/gpt-pro-sol/ (5 prompts + README).

## Watch items
- f2qv.2 dirty tree: still uncommitted in canonical checkout as of this session; 3 codex peers live per coordination envelope. First campaign action = owner lands it.
- 93 unlabeled beads: do NOT rush-label (vacuous-green risk); codex policy (unclassified rendering + campaign-entry quality gate) is right.
- Demo artifacts in dirty tree (.agent/demos/* modified) — someone regenerating catalogs; unowned?
- `.dmypy.json` untracked at repo root — daemon-mypy droppings, harmless, could be gitignored.

## Legibility-kit landing (operator-assigned, afternoon)
- Kit escrow: .agent/scratch/legibility-kit-2026-07-10/ (219 files, checksum verified).
- PR #2655 = kit patch (base f6c1da997) 3-way applied on origin/master@ab30ac364 + my amendments. All checks green incl. demo-visual-verify; CodeRabbit P2 (tape/out-dir mismatch) fixed by parameterizing _write_recording_tape on out-dir basename.
- Kit quality: high. Its own validation was real (its sandbox RAN the tour + 41 tests) but base-drift + committed-spec drift slipped through: stale demo-tour.tape (spec updated, file not) → bead 3tl.17; broken tour-emitted tape (cwd paths + 500ms sleep) — preexisting bug the kit propagated; GPT hallucination count in the patch itself: ZERO confirmed (my read --first suspicion was wrong — flag exists via constant; verified live).
- Beads: created 212.11 (Incident 14:32), 212.12 (packet v2), 3tl.17 (drift gate), qsr6 (Sinex-direction decision, vision). Enriched 3tl + 8 children/siblings. Launch-cut CSV statuses corrected (0hqs already closed via #2628).
- Design doc: docs/design/incident-1432-proof-world.md (anti-circularity oracle rule + anti-vacuity witnesses made standing reference).
- Pro-handoff pack: /realm/inbox/pro-handoff/ README + 10 missions. Key calibration: Pro sandbox executes pure-Python polylogue (kit proved it) → demand executed proof for Python missions; whole-epic scale per operator ("could be major features, perhaps whole epics").
- Tour first-result timing is load-sensitive: 5.1s idle, 20-34s under codex's live index-v29 rebuild (one failed run at 34.3s vs 30s budget — reran, passed). Watch: budget flakiness under host load could hit CI someday? CI runners are isolated; local-only concern.
- Host state: codex agent live (dialogue [9]-[12]): k6fm merged #2632, kj22 fuzz failures under classification, v6vy in verification, index v29 rebuild running, browser-extension reverse channel proven (dry_run_filled_not_sent), GPT strategy falsification NARROW+30-day gate with 2 rejected recommendations.

## Legibility kit v2 landing (evening)
- PR #2662 merged (529fc4397): demo receipts + public-claims gate + corpus v2 (34 constructs) + tour v2. Escrow .agent/scratch/legibility-kit-v2-2026-07-10/ (reconstructed from loose downloads; MISSING-FROM-DOWNLOAD.txt: incident materials/, parser/, fork prompts, 01/02/09 docs, legibilityctl.py, scorecards).
- Reconciliation lessons: (1) kit tape regressed AGAIN (2nd occurrence; 3tl.17 vindicated); (2) master-side tests added after kit base (test_corpus.py) are the main conflict class; (3) atomic git apply fails on already-applied deletions — exclude regenerable artifact dirs and regenerate locally instead; (4) `demo receipts` refuses to seed when POLYLOGUE_ARCHIVE_ROOT is set — correct live-archive safety, remember --root+--seed for testing.
- render_all is now SERIAL (kit change, kept; quick gate 20.7s unchanged). If renders slow down later, this is where to look.
- Beads: 3tl.18 created (swarmctl/mission-plane adjudication vs s7ae — codex lane); 3tl.16 is a close-candidate after AC re-read; 212.11 got construct 1 delivered.
- CodeRabbit rate-limited on #2662 (no review) — classified as tool capacity; merged on green gates + local verification.

## Fork-capture recovery + capture fidelity (late evening)
- 18 chatgpt captures stranded in codex's TEMP spool (/realm/tmp/polylogue-browser-post/spool) — extension posts to :8876 loopback receiver, not prod :8765 (token-guarded). Durable copy: /realm/inbox/polylogue-browser-spool-2026-07-10/. Ingested via POST /api/ingest on :8766 (stage into archive_root/inbox first; _staged_inbox_source refuses arbitrary paths).
- PR #2666 merged: sandbox:/mnt/data links → sandbox_file attachment rows (unfetched). PR #2668 merged: citation-marker stripping (U+E200-E202) + user/model_editable_context → runtime_context messages (ChatGPT memory + custom instructions now archived).
- New bead polylogue-5k5l: capture-time byte acquisition (sandbox + file-service) — inbound pipeline, NOT ptx (outbound).
- Gotchas burned today: (1) shell cwd drifts back to main checkout between tool calls — cd explicitly in EVERY worktree command chain; test append went to main once (restored). (2) main checkout .venv was broken all day: stale editable pth pointing at deleted worktree /realm/tmp/worktrees/polylogue-demo-shelf-closure (PR #2651's); my uv sync in it accidentally repaired it. (3) `str.replace("", x)` catastrophe: heredoc dropped private-use literal chars → empty-pattern replace interleaved the whole file; recover via git checkout from the just-made commit; always write \\ue escapes, never literals. (4) demo receipts refuses to seed when POLYLOGUE_ARCHIVE_ROOT set — use --root+--seed.

## Asset acquisition + citation deepening (night, PR #2669 merged 8db8d60b3)
- Extension now fetches sandbox/file-service bytes at capture (bridge two-step: interpreter/download|files/download -> signed URL -> bytes; 25/75MB budgets; outcome disclosure). Envelope session.attachments param added to buildEnvelope.
- CRITICAL parser fix: envelope attachments were silently dropped on the native-payload delegation path (every rich capture). _merge_envelope_attachments; byte rows win on id collision, keep parser attachment_kind.
- Citations: ChatGPT cites PARTICULAR LOCATIONS — answer-side char span (start_ix/end_ix) + source-side doc identity (metadata.name/id) + line ranges (often only in the U+E200 marker tokens, metadata line_range usually null). Deepened _construct_from_reference (nested metadata/extra) + inline_citation_marker constructs preserve tokens at original-text offsets (same coordinate system as citation spans — correlatable).
- vitest descriptor test caught real bug: file-service:// scheme matched as file id.
- 5k5l remaining: live capture proof (codex handoff, dialogue [16]): reload extension, repoint at prod receiver, re-capture the 10 forks.
