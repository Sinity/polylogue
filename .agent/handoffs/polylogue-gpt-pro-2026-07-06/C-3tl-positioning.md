## [user 307]
▎ *You have the forked context: the polylogue bundle, the swarm2 daemon/composer/thin-client designs, and our discussion of queries-as-objects, delegation units, and annotation recipes. Ground every claim in the bundle (cite file:line / bead id); tag [evidence] vs [proposal]. Be opinionated — one defended recommendation plus the runner-up you rejected and why. Answer in layers: (1) with today's substrate, (2) with a near-term substrate change, (3) the full direction. End with Open questions for the operator and What's missing (a claim unverified, a source unread, a modality not run). The snapshot is dated; live code may have moved — flag assumptions to verify locally. Don't re-derive swarm2.*

Analytical [A] over the bundle:

[A] Product positioning & the legibility bar (3tl). Sharpen the one-sentence category, the README first-screen, the citable-finding artifact, the stranger-can-run demo. The wedge vs "grep ~/.claude."

## [assistant 319]
bash -lc python3 - <<'PY'
from pathlib import Path
queries = {
'/mnt/data/situation-brief(2).md': ['Polylogue', 'system of record', 'wedge worth leading', 'The bottleneck is legibility', 'The demonstration crux', 'The process story is safest', 'claim-versus-evidence', 'structured tool outcomes', 'stranger'],
'/mnt/data/SWARM_BRIEF.md': ['External legibility', 'Read algebra baseline', 'Powerful, resident', 'composer / live-preview', 'No bare-query', 'Fewer named views'],
'/mnt/data/B8_contract.md': ['PolylogueService', 'six verbs', 'preview does not exist', 'facets', 'Read algebra', 'query', 'act', 'recommend'],
'/mnt/data/C10_composer_ux.md': ['Every commit writes', 'recall entry', 'headline', 'preview', 'complete', 'recalled query'],
'/mnt/data/A2_protocol.md': ['The daemon HTTP API', 'length-prefixed', 'preview', 'complete', 'Conclusion', 'warm resident'],
'/mnt/data/B7_thinning.md': ['cli/archive_query.py', 'The load-bearing finding', 'facade already exists', 'Contract-surface gaps', 'Query schema/parse contract'],
'/mnt/data/Pasted text(35).txt': ['operator-priority-frame', 'Persistent Memories', 'beads-workflow', 'Read tests/unit'],
'/mnt/data/Pasted text(34).txt': ['D. The legibility campaign', 'cfk (uplift re-run', '3tl.4', 'system of record for AI work', 'grep ~/.claude', 'grep'],
'/mnt/data/logs.raw-log(5).md': ['grep ~/.claude', 'rawlogs easy to access', 'Polylogue', 'README still sucks', 'legibility', 'agent does better'],
}
for file, pats in queries.items():
    lines = Path(file).read_text(errors='replace').splitlines()
    print('\n###', Path(file).name, 'lines', len(lines))
    for pat in pats:
        found=[]
        low=pat.lower()
        for i,l in enumerate(lines,1):
            if low in l.lower():
                found.append(i)
                if len(found)>=5: break
        print(pat, found[:5])
PY

## [assistant 322]
bash -lc for f in '/mnt/data/situation-brief(2).md' '/mnt/data/SWARM_BRIEF.md' '/mnt/data/B8_contract.md' '/mnt/data/C10_composer_ux.md' '/mnt/data/A2_protocol.md' '/mnt/data/B7_thinning.md' '/mnt/data/Pasted text(34).txt' '/mnt/data/Pasted text(35).txt' '/mnt/data/logs.raw-log(5).md'; do echo '###' $f; done
nl -ba '/mnt/data/situation-brief(2).md' | sed -n '21,35p;55,66p;73,90p;93,106p;133,141p;153,160p'
echo '--- SWARM'; nl -ba '/mnt/data/SWARM_BRIEF.md' | sed -n '9,42p;48,66p'
echo '--- B8'; nl -ba '/mnt/data/B8_contract.md' | sed -n '66,82p;102,126p;134,160p;266,284p'
echo '--- C10'; nl -ba '/mnt/data/C10_composer_ux.md' | sed -n '1,18p;59,76p;217,229p'
echo '--- A2'; nl -ba '/mnt/data/A2_protocol.md' | sed -n '12,20p;27,63p;76,104p'
echo '--- B7'; nl -ba '/mnt/data/B7_thinning.md' | sed -n '24,42p;150,167p'
echo '--- Pasted34'; nl -ba '/mnt/data/Pasted text(34).txt' | sed -n '1763,1788p;1388,1400p;646,650p;4300,4310p'
echo '--- Pasted35'; nl -ba '/mnt/data/Pasted text(35).txt' | sed -n '33,67p'
echo '--- rawlog'; nl -ba '/mnt/data/logs.raw-log(5).md' | sed -n '624,628p;2144,2150p'

## [assistant 340]
## Analytical [A] — Product positioning and the legibility bar, bead `polylogue-3tl`

Snapshot assumption: this is grounded in the uploaded July 5 bundle and the attached swarm outputs. Live code/beads may have moved. In particular, the checked-out README in the bundle already contains much of the 3tl positioning work, while the post-swarm transcript says some legibility beads remain open or newly created. Verify locally with `bd show polylogue-3tl --json`, `bd show polylogue-3tl.4 --json`, current `README.md`, current `docs/proof-artifacts.md`, and the docs site build before acting.

## Recommendation

[proposal] Lead with **claim-vs-evidence**, not “agent memory,” not “chat archive,” and not “better grep.” The defended public category should be:

**Polylogue is the local system of record for AI work: it turns vendor chat logs and coding-agent traces into evidence you can search, analyze, audit, and use to resume work.**

[evidence] This category is already present in the current README snapshot: “Polylogue is the system of record for AI work,” covering ChatGPT, Claude, Codex, Gemini, Antigravity, Hermes, and coding-agent sessions, with the four verbs search/analyze/audit/remember (`README.md:9-12`). The orientation transcript independently says the legibility epic is explicitly not “chat viewer / observability tool / memory store,” but “the system of record for AI work,” fronted by search/analyze/audit/remember. fileciteturn1file5

[proposal] The first public artifact should be a **claim-vs-evidence finding page** plus a **private-data-free reproduction path**. It should show that Polylogue can detect when coding agents proceed after structured tool failures, because that is the cleanest wedge against both “just grep `~/.claude`” and generic memory products.

[evidence] The Situation Brief identifies this as the defensible wedge: Polylogue captures structured tool outcomes, error flags, and exit codes, allowing comparison between assistant claims and tool results; it argues that the moat is structured-outcome capture + multi-provider corpus + construct-validity discipline, not “chat search” alone (`situation-brief(2).md:57-61`). fileciteturn1file15 The proof page says Polylogue reads tool outcomes from provider structure rather than assistant prose and cites `is_error=1` / non-zero `exit_code` as the failure predicate (`docs/proof-artifacts.md:11-13`, `39-53`). The storage DDL backs this: `blocks` has `tool_result_is_error` and `tool_result_exit_code`, and the `actions` view exposes `is_error` and `exit_code` (`polylogue/storage/sqlite/archive_tiers/index.py:195-196`, `328-337`).

[proposal] The runner-up is **browser-captured agent recovery / context-pack uplift**, but I reject it as the first public wedge. It is strategically important, and probably the higher long-run product story, but the current evidence says the earlier uplift experiment was negative/ambiguous and the re-run is still `polylogue-cfk`, not done. Claim-vs-evidence already has a public-safe aggregate and reproduction method; agent recovery should be second-wave after `cfk` closes.

[evidence] The Situation Brief says the closest “useful” demo is agent recovery, but also says claim-vs-evidence travels on its own and either artifact would be sufficient if finished (`situation-brief(2).md:133-137`). The bead `polylogue-cfk` says the prior two-arm experiment recorded “packs lose” due to stale packs and needs an n≥3 re-run, then n=12–20 for a publishable uplift claim. The README also explicitly says memory-benefit claims must stay capability-phrased until measured uplift proves the effect (`README.md:98-100`). fileciteturn1file15

## Layer 1 — With today’s substrate

[evidence] The first-screen README is already close. It has the category sentence (`README.md:9-12`), a 7-minute skim ladder (`README.md:14-25`), five concrete questions ordinary transcript folders cannot answer (`README.md:27-39`), local-first posture (`README.md:41-45`), and a private-data-free demo entrypoint (`README.md:47-61`). The proof artifacts page already maps claims to artifacts and warns that unproven claims must be phrased as capabilities, not facts (`docs/proof-artifacts.md:1-15`). The release gate also says external release must be installable, demoable without private data, and truthful about shipped behavior (`docs/plans/release-readiness-gate.md:5-8`).

[evidence] The one-command demo is not aspirational in the snapshot. `docs/examples/demo-tour/report.md` says `polylogue demo tour` passed, produced first result in 7.956s and full tour in 13.442s, with 11 sessions and user overlays present (`docs/examples/demo-tour/report.md:1-18`). The pre-release install-path proof says the release command is `uvx polylogue demo tour`, with first result in 4.636s and full tour in 10.180s (`docs/examples/demo-tour/uvx-proof.md:1-23`). The demo corpus datasheet reports 11 sessions, 43 indexed messages, five origins, failed tool results, lineage links, subagent links, compaction events, run rows, observed events, context snapshots, and no declared residual construct gaps (`docs/plans/demo-corpus-construct-audit.md:14-89`).

[evidence] The current claim-vs-evidence proof is stronger than a normal demo. It reports a public-safe aggregate generated on 2026-07-04 against schema v24: 5,000 structured failures sampled from 42,033 structured failures, 1,205 silent-proceed next turns, 24.1% silent lower bound, and marker calibration precision/recall values (`docs/proof-artifacts.md:39-53`). It also gives both private local regeneration commands and private-data-free reproduction commands using a seeded archive (`docs/proof-artifacts.md:55-77`).

[proposal] Therefore, with no substrate change, the legibility deliverable can be:

1. README first screen stays category-first, but gets one **finding card** immediately after the five questions: “In a 5,000-row sample of structured tool failures, at least 24.1% were silent-proceed next turns; reproduce the method on the seeded corpus.” Make the private/archive number clearly public-safe aggregate, not demo-corpus result.

2. Add an explicit “Why not `grep ~/.claude`?” box. The answer should be: grep finds text; Polylogue resolves provider structure, tool outcomes, exit codes, costs, lineage, typed units, and evidence-backed derived claims. Ground it in `README.md:27-39`, `README.md:102-120`, `docs/proof-artifacts.md:39-77`, and `index.py:195-223`.

3. Use `uvx polylogue demo tour` as the stranger command once packaging is real; until release, use `uvx --from <repo> polylogue demo tour` because the uvx proof says that is the pre-release equivalent (`docs/examples/demo-tour/uvx-proof.md:3-8`, `20-23`).

4. Publish a static `docs/findings/claim-vs-evidence/index.md` even before the full 3tl.4 renderer, if necessary. It can be hand-generated from the current proof packet, as long as it contains the reproduction commands and a provenance stanza.

[evidence] There is a docs inconsistency to fix before leaning too hard into the current README: the swarm brief says “no bare-query support” and search should be via explicit verb/composer (`SWARM_BRIEF.md:60-62`), but current `docs/getting-started.md` still teaches `polylogue "css refactor"` and says bare tokens are full-text search terms (`docs/getting-started.md:31-35`), and `docs/search.md` says bare tokens are full-text terms (`docs/search.md:5-6`). The public story should not simultaneously say “explicit verb/composer” and teach bare query. fileciteturn2file7

[proposal] Today’s first-screen copy should be more like this:

> **Polylogue is the local system of record for AI work.** It reads the traces your AI tools already leave behind — ChatGPT, Claude, Codex, Gemini, browser chats, and coding-agent tool calls — and turns them into evidence you can search, analyze, audit, and use to resume work.  
>  
> It is not a prettier transcript folder. Grep can find the word “pytest”; Polylogue can show which tool call failed, whether the agent noticed, what it cost, which session it came from, and what raw evidence supports the claim.

That is sharper than “chat archive with FTS and semantic search,” and it directly answers the skeptical “why not grep?” reader.

## Layer 2 — With one near-term substrate change

[proposal] Implement **3tl.4 findings publishing lane** before adding new product demos. Do not build a new interactive surface for legibility first. The missing external object is not another UI; it is a stable, citable finding URL.

[evidence] Bead `polylogue-3tl` defines terminal acceptance as: a stranger can understand from the README first screen, run the one-command demo, and cite a published finding URL, verified by a cold-reader pass. Bead `polylogue-3tl.4` is already scoped as the findings-publishing lane: `docs/findings/<slug>/finding.yaml`, render to `docs/findings/<slug>/index.md`, stable `/findings/<slug>/` URL, provenance gate, changelog, no private rows, seeded-corpus regeneration, and `devtools render findings --check`. The post-swarm transcript says 3tl.4 was raised to P2 because it produces the published finding URL required by the legibility epic. fileciteturn1file8

[proposal] The near-term substrate should be a **Finding object**, not just a Markdown page. It can start as YAML because 3tl.4 already specifies YAML, but the model should anticipate database-backed query objects later. Minimum fields:

`finding_id`, `slug`, `title`, `claim`, `status`, `archive_cursor`, `sample_frame_query`, `measure_version`, `query_dsl_version`, `code_commit`, `run_date`, `source_query_refs`, `evidence_pack_refs`, `seeded_reproduction_command`, `private_archive_aggregate`, `public_demo_result`, `limitations`, `changelog`.

[proposal] The one near-term design improvement over 3tl.4 is this: the `sample_frame_predicate` should be stored as a **query object reference** when available, and as query text only as a fallback. That lines up with the queries-as-objects direction without blocking 3tl.4. In other words: ship YAML now, but design it as a serialized `Finding`, not a bespoke docs format.

[evidence] C10 already proposes that every committed composer run writes a recall entry to `user.db` with query text, resolved spec, result fingerprint, and timestamp, and that naming a recall entry promotes it to a macro (`C10_composer_ux.md`, §3.5–3.6). fileciteturn2file14 The swarm brief says the read surface is `Query × Projection × Render`, and projection/render are already part of the architectural direction (`SWARM_BRIEF.md:9-29`). fileciteturn2file10

[proposal] The near-term “stranger-can-run” artifact should be:

`uvx polylogue demo tour`  
then  
`uvx polylogue finding reproduce claim-vs-evidence`

The second command may initially be a docs/devtools command, but the public copy should look like a product affordance, not a private devtool. If the real command is still `devtools workspace claim-vs-evidence`, the README should say “developer reproduction” and the finding page should expose that honestly.

[evidence] The proof artifacts page currently reproduces the method with `devtools workspace claim-vs-evidence` after seeding/verifying a demo archive (`docs/proof-artifacts.md:65-77`), so the method exists, but the productized verb does not. That is exactly what 3tl.4 should convert into a stable published lane.

## Layer 3 — Full direction

[proposal] The full direction is **not** “better docs” and not “better demo scripts.” It is that Polylogue should make analyses themselves first-class objects:

`raw session evidence → query runs → result relations/cohorts → evidence packs → annotations/assertions → findings → published pages`

[proposal] In the full version, a public finding is not a static page generated at the end. It is the visible projection of an analysis run. The operator or agent asks a question; Polylogue stores the queries; the result sets become cohorts; an agent annotates rows; those annotations become assertions/judgments; a finding renderer produces a page with claims, caveats, reproduction commands, and evidence packs. The docs page is just one render target.

[evidence] This is consistent with the substrate already in the bundle. The read algebra covers query/projection/render and typed units including assertions, files, runs, observed events, and context snapshots (`SWARM_BRIEF.md:9-29`). The hot-daemon/composer direction makes query composition, preview, and commit histories explicit, with recall entries storing resolved specs and result fingerprints (`C10_composer_ux.md`, §3.5–3.6). fileciteturn2file10 fileciteturn2file14 The current proof discipline already says unsupported claims should be phrased as capabilities or caveats, not facts (`docs/proof-artifacts.md:1-15`).

[proposal] In that full direction, the product positioning becomes stronger:

**Polylogue is the local evidence layer for AI work: it records what agents did, lets you investigate it as structured evidence, and turns the investigation itself into citable, reproducible artifacts.**

That sentence is less cute than “flight recorder,” but it lands the unique category: not just storage, not just memory, not just search, but evidence and investigation.

[proposal] The “grep” comparison should become a table only after the claim-vs-evidence page exists. Until then, use prose. A table without a public finding risks sounding like marketing. Once the finding exists:

| Task | `grep ~/.claude` | Polylogue |
|---|---|---|
| Find a word | Yes | Yes |
| Normalize across Claude/Codex/Gemini/ChatGPT | No | Yes |
| Pair tool calls with tool results | No | Yes |
| Detect structured failure via exit code / `is_error` | No | Yes |
| Ask whether the agent acknowledged the failure | Manual | Query + report |
| Reproduce a public-safe method | No | Seeded demo + finding |

[evidence] The bundle supports every row except “reproduce a public-safe method” as a fully productized finding URL. The structured-failure data path is present (`index.py:195-223`, `328-337`); the private-safe proof packet and seeded method exist (`docs/proof-artifacts.md:39-77`); the missing citable URL is exactly `polylogue-3tl.4`.

## The legibility bar, stated operationally

[proposal] Bead `polylogue-3tl` should be considered closed only when all of these are true:

1. **First screen:** A cold reader can say, in one sentence, what Polylogue is and why grep is insufficient. This is not “knows the architecture.” It is “understands the category.”

2. **One command:** A cold reader can run one public command without private data and get a passing report, transcript, and recording/tape. The current demo-tour artifacts suggest this is close or already true, but the public install path must be verified against the current package.

3. **One finding URL:** A cold reader can cite a stable URL with a specific claim, methodology, caveats, reproduction command, and changelog. This is the missing external artifact.

4. **One caveat discipline:** Any measured claim says whether it comes from private archive aggregate, seeded demo reproduction, or capability-only substrate. No blended rhetoric.

5. **One anti-grep answer:** The README must explicitly say grep is useful for text search but cannot audit structured tool outcomes, costs, lineage, or derived claims.

[evidence] The Situation Brief states the bottleneck is “demonstrated use that an outsider can see,” and every strategic path waits on one externalized proof (`situation-brief(2).md:65-81`). It also warns that substrate/scaffold work can substitute indefinitely for the external artifact (`situation-brief(2).md:103-113`). fileciteturn1file15

## Beads / work items I would sharpen

[evidence] Existing relevant beads: `polylogue-3tl`, `polylogue-3tl.1`, `polylogue-3tl.2`, `polylogue-3tl.4`, `polylogue-3tl.11`, `polylogue-3tl.12`, `polylogue-cfk`, `polylogue-sru`, `polylogue-tf2`, `polylogue-jxe`, `polylogue-212`, `polylogue-4bu`. The post-restructure transcript says external legibility/demos are top-of-frame, with `cfk`, `3tl.11`, `3tl.4`, and `3tl.12` as the critical path. fileciteturn1file11 fileciteturn1file12

[proposal] I would add one child under `3tl` only if it does not already exist locally:

**`3tl.x — Anti-grep positioning proof card`**  
Description: The README/docs site must include a concise “why not grep/transcript folder?” proof card grounded in one claim-vs-evidence example.  
Acceptance: A cold reader can correctly distinguish lexical transcript search from Polylogue’s structured evidence model; the card links to one finding URL and one seeded reproduction command; doc-command verify passes.

[proposal] I would not create a broad “marketing site” bead. That would invite the exact substrate/scaffold substitution the Situation Brief warns against. `3tl.4` and `3tl.12` are enough if executed sharply.

## Open questions for the operator

1. Do you want the first public claim to include the private-archive number “24.1% silent lower bound” as a public-safe aggregate, or should the public page lead only with the seeded reproduction method and keep private-archive numbers behind a caveat?

2. Is the public package path ready enough to promise `uvx polylogue demo tour`, or should the README use `uvx --from git+https://... polylogue demo tour` until release?

3. Should the top-line category be “local system of record for AI work” or “local evidence system for AI work”? I prefer “system of record” for category creation, but “evidence system” explains the grep wedge faster.

4. Should “memory” appear in the first screen at all? I would mention it only as “remember,” with a caveat that memory-benefit claims are capability-phrased until `cfk` proves uplift.

5. Which artifact gets the first stable URL: claim-vs-evidence only, or claim-vs-evidence plus v24 agent-forensics? I recommend claim-vs-evidence first, v24 forensics immediately after.

## What’s missing

I did not run live `bd`, `polylogue demo tour`, `devtools workspace claim-vs-evidence`, `devtools render pages`, or the docs site build. I inspected the uploaded snapshot and reports only. The actual live branch may have already fixed the bare-query docs conflict, advanced `3tl.4`, or changed README copy.

I did not verify that `docs/findings/` exists in the live tree; in the uploaded working tree it does not appear as an implemented publishing lane, while `docs/proof-artifacts.md` and `docs/examples/demo-tour/` do exist.

I did not inspect the actual `.agent/demos/claim-vs-evidence` packet beyond the proof-artifacts summary, so the claim-vs-evidence aggregate is grounded in the documented proof page, not independently re-run.

I did not run a cold-reader pass. The 3tl closure condition explicitly requires one.

I did not inspect current GitHub Pages output or the published docs URL. Stable URL availability remains unverified.

I did not use visual modality for the demo GIF/recording; I only inspected file presence and textual reports.
