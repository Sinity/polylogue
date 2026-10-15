# Decisions

Every design decision with a record of its own, who made it, and where the
mechanics live. Smaller decisions are recorded in the spec section they
govern; the questions the spec authors settled are indexed in
[questions.md](../questions.md), and the decisions agreed with the Sinex
session are in `/realm/inbox/spec/polylogue-sinex-joint/decisions.md` (joint decisions, numbered; the file is the current list).

"Spec authors" means the decision was taken from the definitions,
principles, the operator's dated rulings and measured evidence; the operator
may overrule any of them.

| Decision | Status | Made by | Date | Mechanics in |
|---|---|---|---|---|
| [Keeping input bytes](raw-retention.md): every input and every revision, in the cheapest exact form | decided | operator (keep everything); spec authors from measurements (revisions, forms) | 2026-10-11 | `spec/storage/README.md`, `spec/storage/blobs.md` |
| [Database files](storage-files.md): `archive.db`, `index.db`, `embeddings.db` | decided | spec authors, from the operator's request to reduce the tiers | 2026-10-11 | `spec/storage/README.md`, `spec/storage/blobs.md` |
| [How commands talk to the daemon](process.md): every read and write goes through the daemon's socket; warm caches; one writer; no standalone mode | decided | the operator (2026-10-11), after the ruling of 2026-09-14 | 2026-10-11 | `spec/surfaces/README.md` (processes), `spec/behavior.md`, `spec/storage/README.md` |
| [Lineage](lineage.md): full message lists, copied messages marked, three link kinds | decided | spec authors, from measured cost and real data | 2026-10-11 | `spec/model.md`, `spec/sources/claude-code.md`, `spec/sources/codex.md` |
| [One message per model response](one-message-per-response.md) | decided | spec authors, from real data | 2026-10-11 | `spec/model.md` |
| [Merging routes](merging.md): a fixed order of faithfulness, never "latest wins" | decided | operator ("latest wins was sloppy", 2026-07-31) | 2026-07-31 | `spec/sources/README.md`, "The same conversation through several routes" |
| [Schema validation](schema-validation.md): the typed parse is the validation; schemas inferred offline | decided | spec authors, from measurements | 2026-10-11 | `spec/analysis.md` |
| [Query parser](query-parser.md): pest | decided | spec authors | 2026-10-11 | `spec/query.md` |
| [Repository](repository.md): fresh workspace, continuous repository | revised | fresh replacement after preservation | 2026-10-13 | `plan/workspace.md` (branch), `plan/method.md` (PRs) |
| [Polylogue on Sinex](sinex.md): standalone and native backing, qualified before cutover | decided | operator (design, 2026-04 to 2026-10); joint phase (mechanics) | 2026-10-11 | `spec/convergence.md`, `spec/storage/README.md` |
