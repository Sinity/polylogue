# Fork 01 — Polylogue landing narrative and public category

Work directly on the supplied Polylogue repository. Treat Beads—not GitHub Issues or speculative Markdown plans—as roadmap authority. Your mission is to make a technically sophisticated stranger understand and care about Polylogue within one screen, one minute, and ten minutes.

## Product thesis

Use this category consistently:

> Polylogue is the local flight recorder and system of record for AI work.

Its public promise is: search heterogeneous agent histories, read tool activity as work rather than chat, audit claims against structural outcomes, understand lineage and accounting, and resume from reviewed evidence.

Do not collapse the project into “AI memory,” “chat archive,” or “LLM observability.” Those are subordinate capabilities.

## Owned scope

You have exclusive ownership of:

- `README.md`;
- package metadata descriptions in `pyproject.toml`, `flake.nix`, and equivalent top-level metadata;
- landing-page copy and layout templates under `devtools/pages_*`;
- top-level documentation navigation/registry files;
- new public orientation docs, but not product runtime code.

Do not change demo fixture behavior, semantic renderers, daemon behavior, query semantics, or database code.

## Required work

1. Audit every top-level category phrase and remove contradictions such as “Your AI memory” versus “system of record.”
2. Rewrite the README around visible payoff rather than repository process. Use this order:
   - category and promise;
   - a public-safe visual or demo command;
   - five concrete questions the product answers;
   - one structural-evidence chain;
   - the trust model;
   - architecture;
   - current status and honest limitations;
   - installation and documentation.
3. Remove stale meta-persuasion and “how to read this README” copy. Show the product rather than defending it.
4. Ensure the landing page does not render zero-valued archive statistics in CI or a fresh checkout. Prefer capability cards and deterministic demo facts.
5. Add a short “Why Polylogue” page comparing it with grep, transcript viewers, vector memory, agent tracing, and Git without constructing a straw man.
6. Make Beads authority explicit and remove public navigation that implies GitHub Issues are the roadmap.
7. Preserve all required generated-doc markers and repository-specific generation contracts.

## Construct-validity constraints

Every public claim must be one of:

- a deterministic fact from the public demo;
- a bounded private-archive field observation with corpus and caveats;
- a current capability with a direct code/test/proof path;
- an explicitly labeled plan.

Do not claim general reliability uplift, universal provider completeness, provider invoice accuracy, or real-scale latency from the small deterministic corpus.

## Validation

Run the repository’s canonical documentation and page-generation checks. At minimum:

- render the generated docs surface;
- build the static site;
- run doc-command verification;
- run generated-site link checks;
- run `git diff --check`;
- inspect the rendered home page rather than trusting source templates alone.

## Deliverables

Produce:

1. a concrete patch;
2. a one-page before/after explanation;
3. a list of every public claim added or removed;
4. validation output;
5. unresolved legibility defects that require runtime or visual work.

Commit hygiene is secondary to speed. Keep edits inside the owned scope so the conductor can merge the branch wholesale.
