## [user 307]
think of many more demo ideas, grounded ones

## [assistant 322]
The right bar for these demos is: **a stranger can inspect the artifact and see a task get easier, truer, or more recoverable because Polylogue exists.** The situation brief is explicit that the current demo shelf mostly proves the substrate is honest, while the missing thing is an agent or operator completing a real task better; it names browser-captured agent recovery and claim-versus-evidence as the closest high-leverage demos. fileciteturn22file4

Below are many grounded demo ideas. I’m deliberately biasing toward things that use real existing traces: Claude Code compaction, Beads, claim/evidence data, browser capture, raw logs, agent swarms, commits, sessions, tool calls, cost data, lineage, and context packs.

## Highest-value demos to ship first

1. **Cut-session recovery pack**

A coding agent is handed a session id and a terse goal: “continue this work.” Baseline: agent gets only the user’s manual summary. Polylogue version: agent gets a recovery pack with prior session summary, relevant messages, tool calls, files touched, commits, open beads, branch state, and unresolved risks. Output: the next agent successfully performs a bounded follow-up without the user re-explaining. This directly matches the situation brief’s closest-to-done demo: browser-captured agent recovery. fileciteturn22file4

2. **Claim-versus-evidence report**

Run a corpus analysis over coding-agent sessions: find places where the assistant claimed success, but tool calls, exit codes, CI logs, failed tests, or verifier output contradicted it. Output: a report with counts by model/provider/tool, a specimen gallery, and exact evidence refs. This is probably the most externally legible “finding travels on its own” demo: “coding agents silently proceed past failed calls N% of the time.” fileciteturn21file11

3. **Claude compaction memory audit**

Use the attached Claude Code excerpt as the hook. After compaction, Claude receives a flat dump of 19 persistent Beads memories, including global doctrine, shell traps, schema rules, daemon gotchas, and cost semantics. Polylogue demo: compile a task-specific context pack from the same memory/assertion material, explain why each memory was included or omitted, and compare token load plus downstream usefulness. fileciteturn22file2

4. **“Fable as Foreman” subagent-rhetoric report**

The tweet hook becomes a general analysis: extract subagent delegations, classify instruction style, output-contract strictness, scope policing, retry behavior, and model-family routing. Output: funny specimen gallery plus serious aggregate tables. This uses the same external-annotation loop discussed earlier: an agent labels delegation events, imports judgments/assertions, then queries them.

5. **Beads swarm autopsy**

Take the huge Beads audit swarm as data. Reconstruct: prompts issued, domains assigned, subagent outputs, contradictions between auditors, critic gate decisions, applied mutations, held items, and final ledger. Output: “How a 40-agent backlog audit actually worked.” This is a process-story demo, and the situation brief says the process story is the safest and most ready inbound story. fileciteturn22file19

6. **Before/after Beads quality audit**

Input: pre-audit Beads snapshot. Polylogue extracts null acceptance criteria, orphaned beads, missing dependency edges, stale GitHub refs, and under-specified frontier work. Output: before/after dashboard: acceptance criteria filled, new beads created, deps added, cycles avoided, held decisions. This turns the Beads restructuring into a concrete “agentic project management substrate” demo.

7. **Stale docs versus code reality check**

Given a stale architecture doc and live code/schema, Polylogue finds contradictions and produces a PR-ready patch plan. The uploaded swarm/chat excerpt contains exactly this pattern: architecture-spine schema-versioning contradicted additive durable migrations and execution-plan was stale. Output: stale-doc report with code/doc/bead refs and recommended patches. fileciteturn22file15

8. **Critical-path selector**

Input: the whole Beads graph. Ask: “What should I work on next?” Polylogue produces a ranked frontier: ready, blocked, stale, high-leverage, proof-campaign relevant, and anti-recommendations. Output: a short decision memo, not a giant list. This is grounded in the audit’s conclusion that the real wave-1 collapsed to a small frontier like `cfk`, `3tl.11`, `3tl.4`, `1xc.*`, `pj8`, and a few 4smp entries. fileciteturn21file15

9. **Notes-sidecar trap detector**

Demo a specific Beads failure mode: many beads looked under-specified because important content lived in the notes sidecar rather than design/acceptance fields. Polylogue finds these, lifts/formalizes them, and prevents agents from inventing duplicate specs. Output: “hidden spec recovery” report plus patch commands.

10. **GitHub external-ref reconciliation**

Beads are authoritative; GitHub issues are external refs, not mirrors. Demo: Polylogue compares Beads and GitHub issue state, identifies done issues still open on GitHub, stale refs, issue scopes folded into larger beads, and non-mirror relationships. Output: operator checklist, no automatic GH mutation. This is grounded in the Beads workflow memory in the compaction excerpt. fileciteturn22file2

## Demos centered on the live query/composer vision

11. **Live set-algebra explainer**

A TUI or recorded terminal demo: compose `A intersect B`, see `|A|`, `|B|`, and `|A∩B|` update before commit, then export the result. The C10 design explicitly frames live set sizes as the payoff of set-algebra preview because users otherwise guess wrong and discover empty results too late. fileciteturn22file0

12. **Projection preview: selection is not enough**

Show the same query rendered as `summary`, `dialogue`, `tool-failures-only`, `commit-card`, and `evidence-pack`. The demo claim: Polylogue is not just finding sessions; it is composing evidence. C10’s commit model already says terminal stages preview their projection, and export reruns the same render spec so preview and artifact match. fileciteturn22file0

13. **“Find touches:ids.py” composer walkthrough**

Use C10’s concrete example. User types `find touches:ids.py |`, stage rail shows the upstream cardinality, completion suggests `group by`, `sort`, `limit`, `intersect`, `union`, `except`, `read`, and preview shows the current relation before the next stage is chosen. Output: a short screen recording showing query literacy emerging from the UI. fileciteturn22file11

14. **Query recall becomes macro**

Run a query, commit it, then reverse-search it with `Ctrl-R`, preview it, name it, and use it as `@macro` in another set expression. This grounds your “query objects” idea in the existing C10 design, which says committed runs write recall entries with query text, resolved spec, result fingerprint, and timestamp, and named recall entries become macros. fileciteturn22file1

15. **Value completion over the real archive**

Type `origin:cod...`, `model:cla...`, `tool:rg...`, `tag:...`, and show values plus counts from the live archive. C10 calls this the honest gap: current completions cover structure, but field-value completion needs live archive-backed values. fileciteturn21file6

16. **Safe destructive action**

A deliberately theatrical demo: query a set, type `| delete` or `| mark`, and Polylogue refuses to do anything until it shows affected-row count and asks for confirmation. This demonstrates the algebra/action model and safety posture at once. C10 explicitly says preview never executes destructive terminals and commit gates them on preview-backed confirmation. fileciteturn22file0

17. **Explain-at-point-of-use**

User types an ugly query. Polylogue shows parse tree, grain, retrieval lane, pipeline stages, projection, exactness/capped count, and why the current preview is stale or sampled. Output: query debugging as a teaching surface. This is a better demo than generic “autocomplete,” because it shows the archive being inspectable.

18. **Daemon-fast CLI versus cold CLI**

Record the same query run cold and then through the daemon-backed thin client. But do not make it a benchmark-only demo. Make the claim: because the warm daemon makes preview/completion possible, the CLI becomes an interactive instrument. A3 says the composer/live-preview UX only exists honestly with the warm daemon, while daemonless remains break-glass/substrate-direct. fileciteturn22file5

## Demos centered on annotations/assertions

19. **External annotation loop**

Prompt an agent: “Label these 200 delegation prompts for directive intensity, output-contract strictness, and scope control. Evidence refs required.” Import JSONL as assertion/judgment candidates. Then query the annotations. Output: report generated from external labels, not manual spreadsheet fiddling.

20. **Assertion provenance demo**

Take one judgment, such as “this assistant ignored a failing test,” and expand it: target message, evidence refs, tool result, confidence, author/model, schema, batch, status. This demonstrates why assertions are better than CLAUDE.md or flat memories. The raw log explicitly notes that assertion entries could be timestamped, carry expiry, have navigable origin refs, versioning, and hierarchical expansion refs. fileciteturn22file8

21. **Human-vs-agent annotation disagreement**

Two models label the same cohort of messages for “claim unsupported,” “scope drift,” or “overconfident closeout.” Polylogue shows agreement, disagreements, and specimen rows. Output: construct-validity artifact, not just labels.

22. **Pathology atlas**

Create an annotation schema for agent failure modes: ignored error, false completion, excessive scope, repeated dead-end, under-specified plan, stale doc reliance, unsafe shell quoting, wrong archive root. Run it over a sample and produce an atlas with examples and counts.

23. **Correction lifecycle**

An agent imports candidate assertions. Human accepts/rejects a few. Polylogue then changes what gets injected into future context packs. Output: one loop from observation → candidate memory → accepted assertion → context behavior.

24. **Finding package**

Select a cohort, import annotations, generate a report, and include every dependency: query run refs, snapshot cohort, annotation batch, evidence pack, render artifact. This is the “analysis itself as object” demo.

## Demos centered on compaction, memory, and context

25. **Flat memory dump versus compiled context**

Use the Claude excerpt exactly. Baseline: 19 global memories dumped after compaction. Polylogue: selects 6 relevant memories for the next task, omits 13 with reasons, includes expansion refs, stale/conflict flags, and task-specific hazards. Output: side-by-side context pack. fileciteturn22file2

26. **Post-compaction failure analysis**

Find sessions where compaction happened, then inspect the next 20 messages/actions. Did the agent ask orientation questions? Re-run commands already known? Miss a memory? Query wrong archive? Output: “what compaction loses” report.

27. **Agent reboot with refs, not pasted context**

A demo inspired by your raw-log idea: instead of huge compaction prose, the agent starts a new session with a compact recap plus refs to expand tool outputs, files, reports, prior messages, and query results. The raw log already sketches this: agents could reset/reboot sessions with prose intact, tool outputs stripped or replaced by refs, and richer context management around session starts. fileciteturn22file6

28. **Context pack diff**

Show what changed between the context given to Agent A and Agent B: new assertions, omitted stale docs, added bead, removed irrelevant memory, branch/daemon warning added. Output: context is no longer invisible magic; it is inspectable state.

29. **Memory utility analytics**

Across sessions, ask: which injected memories were actually used, which were ignored, which warnings prevented mistakes, which are stale noise? Output: delete/supersede recommendations for memories.

30. **CLAUDE.md replacement pressure test**

Use the recent CLAUDE.md compaction session: the file was reduced to ~4K tokens from a ~50K transclusion chain, with AGENTS.md symlinked to it. Demo: Polylogue compiles the truly relevant slices dynamically instead of relying on one static master file. fileciteturn22file3

## Demos centered on work reconstruction and repo evidence

31. **Session-to-commit reconstruction**

Given a commit hash, Polylogue finds the session(s) that produced it, the tool calls that edited files, tests run, verifier output, PR text, and follow-up fixes. Output: “why this commit exists” card.

32. **Commit-to-session archaeology**

Reverse of the above: start from a session, reconstruct which commits/PRs/branches came out of it. The Claude excerpt’s branch/PR cleanup is perfect source material: entangled commit, cherry-pick, force-with-lease, CI verification, final merged commits. fileciteturn22file3

33. **Replay plan, not replay promise**

For a coding-agent session, Polylogue emits a “reproduction plan”: checkout commit X, inspect files Y, rerun commands Z, compare outputs. It does not claim deterministic replay unless evidence supports it. Output: honest work-trace bundle.

34. **Tool-call timeline**

Render a session as: user intent → assistant plan → shell commands → file writes → tests → errors → corrections → commit. This is a much more useful “transcript” for coding work than chat messages.

35. **CI failure root-cause lens**

Select sessions mentioning a failing CI check. Polylogue groups by error class, command, file, responsible commit, and whether the agent fixed or ignored it. Output: top failure modes in agent-built PRs.

36. **“This code path is haunted” report**

Given a file path, Polylogue finds every session, message, tool call, commit, failed test, assertion, and bead involving it. Output: maintenance archaeology for one file.

37. **Branch-local daemon stale-code trap**

Use the compaction memory about the branch-local devloop daemon serving stale code after branch switches. Demo: Polylogue identifies sessions where daemon status/branch state made results suspect, then emits a warning/context pack for the next agent. fileciteturn22file2

38. **Live archive root pitfall detector**

Another concrete memory-derived demo: catch commands that would query `/tmp/polylogue-archive` instead of the live archive. Output: “silent empty archive” prevention. It is small, but highly legible because it prevents a real repeated operator error.

## Demos centered on cost, usage, and provider behavior

39. **Codex cached-token correction**

Before: cost report double-counts Codex cached tokens and inflates costs. After: Polylogue reports disjoint lanes and shows API-equivalent versus subscription-credit views. Output: cost-honesty report with before/after numbers.

40. **Provider/task routing economics**

For multi-agent work, show which model families were assigned which tasks, cost per task type, retries, failures, and output usefulness. This pairs well with the Fable/subagent demo.

41. **Semantic search earned its cost**

Compare lexical, semantic, and hybrid retrieval on a real query set. Show cases where semantic found genuinely useful sessions and cases where it did not. Output: “keep/kill/tune embeddings” decision memo.

42. **Preview budget honesty**

Use the composer or CLI to show exact, estimated, capped, and stale counts. Output: a small demo that Polylogue refuses to silently fabricate certainty. C10’s preview design explicitly labels capped/sampled previews and avoids silent truncation. fileciteturn22file0

43. **Slow demo regeneration autopsy**

Take the claim-vs-evidence regeneration slowdown memory: broad reruns hid an OR/COALESCE predicate and monolithic pairing query; stage timing isolated the bottleneck and led to indexed disjoint failure legs plus chunked pairing. Demo: Polylogue turns a “slow demo” into a structured performance investigation. fileciteturn22file2

44. **Daemon startup and warming honesty**

Record a cold daemon start. Show `ACCEPTING`, `WARMING`, `READY`, stale-available, and composer preview disabled until warm. This demonstrates user trust: no frozen CLI, no silent fallback. A4’s lifecycle design explicitly distinguishes accepting connections from ready/warm reads. fileciteturn22file17

## Demos centered on browser capture and external agents

45. **Browser-captured cloud research delegation**

The raw log suggests browser-capture posting could let a local agent delegate research to cloud web UIs, observe ongoing reasoning, support projects/attachments, and reduce dogfooding friction. Demo version: don’t start with full posting. Start with “local agent sees a cloud research thread as evidence, reconstructs it, and imports the answer with refs.” fileciteturn22file6

46. **WebUI continuation recovery**

Given a browser-captured ChatGPT/Claude/Gemini web thread, Polylogue reconstructs context and emits a prompt for a new web thread to continue it, with source refs and omitted-material index.

47. **Cloud/local boundary report**

Show what a cloud agent can safely certify versus what only the local daemon/operator can certify. This is grounded in the locality doctrine from the Beads audit: cloud agents can patch source/docs/fixtures, but cannot certify deployed local truth. Output: closeout checklist.

48. **Attachment truth demo**

If browser capture says attachments exist, Polylogue shows whether bytes exist, hashes are real/synthetic, whether re-acquisition is possible, and which claims should be avoided. Output: honesty card for an attachment-heavy session.

49. **Multi-instance browser-capture safety**

Use the raw-log concern about multiple browser-capture instances and DOM-degraded coalescing. Demo: simulate duplicate/degraded captures and show Polylogue coalesces or warns rather than double-counting. fileciteturn22file9

## Demos centered on local-first / personal substrate, probably internal first

50. **What was I doing at 23:10?**

Polylogue + adjacent captured sources reconstruct the active AI sessions, terminal commands, browser tabs, raw-log notes, and work item around a timestamp. Output: one “activity reconstruction” page. This touches Sinex/Lynchpin territory, so make it an internal demo unless data is sanitized.

51. **Raw-log to Beads extraction**

Use the raw log as messy intent. Agent extracts candidate work items, deduplicates them against existing Beads, proposes new beads only for genuinely missing items, and cites raw-log timestamps. Output: “thought stream → backlog” conversion. The raw log contains many Polylogue-relevant ideas, including browser-capture posting, agent reboot/compaction, assertions, context injection, alternates, and demo hooks. fileciteturn22file6 fileciteturn22file8 fileciteturn22file9

52. **Tab/provenance capture**

A small demo from the raw-log theme: instead of opening infinite tabs, capture links with provenance, intent, source page, and optional note. Polylogue later queries “why did I save this?” Output: browsing-state recovery.

53. **Terminal/asciinema session summarizer**

Given terminal recordings and AI sessions, Polylogue produces a command diary: what commands were run, what failed, what changed, what should be remembered. Output: self-observability artifact.

54. **Voice/raw note evidence pack**

Internal demo: raw notes or voice transcripts are not “memory”; they are evidence. An agent extracts assertions with confidence, expiry, and evidence refs. This demonstrates the assertion model without making sensitive content public.

55. **Personal test set for models**

Use your own archive as a benchmark: which model is better at reconstructing a messy dev session, finding the decisive tool failure, or extracting useful Beads? Output: model-eval report over your real corpus.

## Demos centered on content variants and render algebra

56. **Translation as content variant, not evidence mutation**

Pick a bilingual/multilingual session. Add translated variants at message/block level. Render original, translation, and side-by-side, clearly indicating variant substitution. The raw log explicitly suggests alternates attached to messages/blocks, with translation as the motivating case but broader uses possible. fileciteturn22file9

57. **Tool-output folding**

Take a huge coding session. Render it three ways: full transcript, authored dialogue only, and evidence pack with tool outputs folded but expandable by ref. Output: same evidence, different projections.

58. **Long-session budget challenge**

Force a report over a session too large for the context window. Polylogue must produce a bounded, honest context pack with omissions and expansion refs, not a fake complete summary. The situation brief says an existing context-pack demo already proves honest truncation; this turns it into a task-completion demo by giving it to an agent. fileciteturn21file11

59. **Specimen-gallery renderer**

Take any annotation batch, such as Fable delegation labels or claim-versus-evidence failures, and render a clean public “gallery” with short examples, counts, caveats, and source refs. This is a generic render layout, not a Fable-specific feature.

60. **Evidence cockpit**

For a report claim, click into a web/workbench page showing query run, cohort, annotations, source messages, tool calls, raw artifacts, and render artifact. This is the “why should I believe this?” demo.

## What these demos imply Polylogue should build

The demos mostly converge on the same missing primitives. Not Fable-specific, not scratchpad-specific.

Polylogue needs **query runs** as first-class objects: every query returns an id, lowered spec, result grain, exactness, degraded state, timing, and result fingerprint. C10 already proposes recall entries with query text, resolved spec, result fingerprint, and timestamp; that should grow into query-run/result-relation objects. fileciteturn22file1

It needs **cohorts**: named dynamic or snapshot sets of sessions/messages/actions/delegations/assertions/artifacts. These make demos reproducible: “the 47 examples in report v1” is a snapshot cohort, not a vague search.

It needs **annotation batches** over assertions: external agents label messages/sessions/delegations/query results, import JSONL, and Polylogue makes those judgments queryable with schema, author, confidence, evidence refs, and status. The raw-log assertion idea already points to timestamped, expiring, ref-backed, versioned memory/assertion entries. fileciteturn22file8

It needs **artifact/work-trace linking**: sessions, actions, files, commits, PRs, reports, evidence packs, and generated artifacts linked by produced/consumed/derived_from edges. This is what makes replay plans, commit archaeology, and agent-swarm autopsies work.

It needs **analysis runs**: not just YAML recipes, not just database rows. A recipe/prompt describes intended procedure; the analysis run records what actually happened: query runs, cohorts, annotation batches, evidence packs, reports, and decisions.

It needs **context compile runs**: every context pack has selected memories, omitted memories, reasons, refs, freshness, and downstream outcome. This directly beats flat `bd prime` memory dumps after compaction.

## My ranking

The first public demo should be either **claim-versus-evidence** or **cut-session recovery**. The first is the best public finding. The second is the best agent-value proof. The third, slightly more playful but still serious, is **Fable/subagent-rhetoric**.

The best internal dogfood demo is **compaction memory replacement**, because the Claude excerpt gives you a perfect before/after: flat persistent memories versus a compiled, cited, task-scoped context pack. The best substrate-shaping demo is **external annotation loop**, because once that works, almost every other analysis becomes scalable instead of fiddly.
