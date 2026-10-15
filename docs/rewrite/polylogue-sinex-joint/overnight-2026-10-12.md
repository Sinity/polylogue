# Overnight, 2026-10-11 to 2026-10-12

What the Polylogue and Sinex sessions did while you slept, in the shared
repository at `/realm/inbox/spec` (git log for every step). `nix flake check
git+file:///realm/inbox/spec` runs every executable part and passes.

## 1. Waiting for you

**Polylogue**
- One open question: the order of work (`polylogue-spec/questions.md`,
  #6 and #7). Default: daily use first, the web reader and workbench later.
- Decided on your behalf, each open to overrule:
  - the whole archive directory is not copy-on-write on btrfs, with checksums
    from Polylogue itself (your question "why COW at all" settled it);
  - every hook line is synced to disk (measured cost: 1.8 ms median);
  - `polylogue sql` goes through the daemon like everything else;
  - agent annotation is constrained by checks and rubric passes rather than
    left free (from your 2026-07-30 journal entry);
  - five new workflows from your journal (see section 3).

**Sinex**
- Nothing blocks the work (`sinex-spec/questions.md`).
- Settled by you: the database lives on `/realm`, where it already is.
- FYI: no Sinex unit has run since the 2026-10-07 boot.
- Lynchpin silently lacks data: Google activity after 2020-01 (`lynchpin-86mu`), Messenger before 2024 (`lynchpin-frb5`), Play Store times (`lynchpin-9ohh`), and counts 21k Spotify plays twice (`lynchpin-n0wq`); its ActivityWatch index writes to the old path (`lynchpin-9iax`).
- Your mpris capture drops records (a 10 s write timeout without retry; `sinnix-3qv3`), and your phone app's own lanes (events, ambient, sensors) stopped on 2026-10-09 while `adb` logcat still arrives (`sinnix-pzxb`).
- Decided on your behalf, each open to overrule:
  - O1 adopted: automata read committed events from Postgres in commit order; NATS stays ingress (`sinex-spec/decisions/event-log.md`);
  - privacy classes `open`, `guarded` and `sealed`; sealed data sits under per-material keys wrapped by a key you hold, and is purged by destroying the key (`sinex-spec/decisions/privacy.md`);
  - one event table with typed per-kind views (`sinex-spec/decisions/event-table.md`).

## 2. Joint decisions made tonight

`polylogue-sinex-joint/decisions.md`, all agreed by both sessions:

| # | Decision |
|---|---|
| 15 | One logical day for both projects: `day.logical_start`, default 04:00, with a shared, tested module |
| 16 | Agent annotations: anchored, checkable claims (holds / fails / unknown), passes under a rubric, blind sampled judging |
| 17 | Every query reads back as a plain sentence (`read_as`) |
| 18 | One session-start brief: Polylogue's hook injects it, and Sinex fills a machine section from a precomputed cache |
| 19, 23, 24 | Sealed blobs in the shared `hashed-blobs` crate; review found and fixed a nonce-reuse break (#23) and names that would confirm content (#24) |
| 20 | Sinex fuses AI sessions by calling Polylogue's merge function, never restating its rules |
| 21 | One visual language: shared type, marks and grounds; Polylogue blue, Sinex teal |
| 22 | AI turns as spans (id, start, end), so Sinex can answer "commands I ran while discussing X" |

Also joint: a backed-mode contract test (`polylogue-sinex-joint/prototype/contract/`,
flake check `joint-contract`). Polylogue's publisher sends Morrow's sessions,
messages and turns; Sinex's admission half admits them whole and suppresses
a second publish. Excision travels as an explicit purge order that Sinex
checks and never infers, including rewriting a shared input.

## 3. What changed

**Polylogue** (`polylogue-spec/`)
- Thirteen audits folded in (eight conceptual, five deep audits of the
  Python code), then a cold review that found 103 contradictions, all fixed.
- From your journal (`evidence/intent-journal.md`):
  - checked claims and rubric passes;
  - workflows for the story of a project from your real archive, session
    reports, sharing annotated excerpts, improving agent instructions from
    history, and studying why models missed problems;
- Prototypes (`prototype/`):
  - a parse framework that cannot drop a record silently;
  - Claude Code into the real `index.db` schema, matching Morrow's slice-1 counts;
    over all 5.8M of your Claude Code records every record is accounted for
    and no author is unknown;
  - append equivalence at 3,234 split points;
  - four search plans returning identical pages;
  - a Codex parser and the shared merge function (your 9,697 rollouts
    exposed a broken authorship rule, now fixed);
  - the daemon path: a warm search through the socket in about 0.1 ms, the
    thin client up in 0.7 ms, writes invalidating the cache;
  - by morning also: live state, checked claims and rubric passes, the
    ChatGPT route merge, all 38 query examples run as a check, and `near:`
    on your 168k real vectors (recall 1.0, no index needed).
- Measured on your real data (`evidence/ingest-benchmark.md`):
  - Claude Code's 20 GB built in 8.3 minutes, Codex's 53 GB in 15.8, inside
    the 30-minute target with little margin;
  - copy-on-write fragmentation slowed cold reads 2.5x; two tuning ideas
    did not help.
- Every local source checked against your real data, counts and key names
  only: about 65 rules corrected (`evidence/*-real-*.md`). Hermes, Grok CLI
  and Gemini CLI parse; Antigravity's steps decode with the schema inside
  its own language server.
- `SET-ASIDE.md` lists what is not built; nothing was dropped.
- The design canvas (Claude Design, private to you): 21 boards, not yet
  checked visually.

**Sinex** (`sinex-spec/`)
- New designs: per-record parsers, one fusion reconciler, a catalog of 135 sources.
- Prototypes (`prototype/`, in the flake):
  - the writer in Rust: invariants held through 30 hard kills;
  - rows that change after capture: today's Sinex truncates them (491 h of your AFK time in 30 days, 2,090 visit durations, 647 long commands); the rewrite re-reads open rows instead;
  - the shared `hashed-blobs` crate: patch-from deltas; batched syncs store media 48 times faster;
  - a query evaluator and an agent using MCP alone (`prototype/query-ref/`);
  - the Sinnix lake: 7.22M records, none unparsed;
  - vector search on real embeddings: HNSW, 98% recall in 1.7 ms;
  - commit stalls of 16-17 s on `/realm` (drive cache), so bulk work paces by commit latency.
- Code audit: 632 of 909 bug findings confirmed (`sinex-spec/audit/SUMMARY.md`).
- Joint: U6 shows both CLIs through a Sinex outage and an excision (`polylogue-sinex-joint/prototype/ux-u6/`); #25 paces both projects' bulk writes on `/realm`.
- On the machine: 9,413 duplicate ActivityWatch rows removed (backed up); Lynchpin reads read-only.
