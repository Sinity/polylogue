# Verification Lab Command Surface

Status: accepted for the current proof-kernel generation. Revisit when proof and
evidence runners need a stable end-user distribution surface outside the repo
control plane.

## Decision

The verification lab lives under `devtools` for now, with a narrow, named
surface instead of a new `polylogue-lab` executable, `polylogue lab` namespace,
or more commands under `polylogue audit`.

Selected vertical slices:

| Slice | Command | Role |
| --- | --- | --- |
| Catalog | `devtools render-verification-catalog` | Render and check the proof-obligation catalog generated from subjects, claims, runners, and compiled obligations. |
| Routing | `devtools affected-obligations` | Map changed paths or refs to affected proof obligations and focused verification commands. |
| Evidence | `devtools semantic-axis-evidence` | Produce comparative proof-envelope performance evidence across semantic scale tiers. |
| Corpus | `devtools lab-corpus` | Generate raw synthetic corpus fixtures or seed complete demo archive workspaces for lab runs. |
| Scenarios | `devtools lab-scenario` | Run showcase exercise scenario sets and committed showcase baseline checks outside the product CLI. |

This surface is intentionally a repo operator surface. It works over proof
subjects, generated docs, changed files, evidence envelopes, and local
verification artifacts. Product/archive-facing checks remain in the product CLI
where they already belong, such as `polylogue doctor --proof`, schema proof
rendering, and archive audit/proof checks.

## Catalog Grounding

The decision depends on the proof-obligation catalog from issue #192, not on a
hypothetical future taxonomy. The catalog at
[`docs/verification-catalog.md`](verification-catalog.md) already records:

- command subjects for the product CLI;
- claims for CLI, schema, provider capability, operation-spec, workflow, and
  semantic archive behavior;
- runner bindings and trust metadata;
- compiled obligations that affected-change routing can target.

That means the first lab surface can be explicit without moving command
implementations. `devtools render-verification-catalog` is the catalog slice,
`devtools affected-obligations` is the routing slice, and
`devtools semantic-axis-evidence` is the first comparative evidence slice.
`devtools lab-corpus` and `devtools lab-scenario` carry synthetic/demo and
showcase exercise work that used to overload `polylogue audit`.

## Alternatives Rejected

`polylogue-lab` is premature. A separate executable implies a stable public
distribution and packaging boundary before the proof/evidence runner UX is
settled.

`polylogue lab` would put repo verification and source-control operations into
the product CLI. That blurs the boundary between archive workflows and
repository proof obligations.

`polylogue audit` is already an archive/product QA surface. Adding the proof lab
there would deepen the overload that this issue is meant to resolve.

Exposing only `devtools render-verification-catalog` is now too narrow. The
catalog slice exists, but affected-obligation routing and semantic-axis evidence
also exist and need discoverable operator entry points.

## Surface Rules

New verification-lab commands should implement a real proof, evidence, routing,
or catalog operation. They should not be aliases over older overloaded commands.

Generated docs and `devtools --list-commands --json` must keep naming the
selected surface, so agents and scripts can discover it without scraping prose.

If a future command must operate on user archives rather than repo proof
obligations, it belongs in the product CLI or API first, not in the
verification-lab surface.

## Migration Notes

`polylogue audit generate` was removed immediately instead of retained as an
alias. Use:

```bash
devtools lab-corpus generate
devtools lab-corpus seed --env-only
```

The showcase exercise smoke flow also moved to the lab surface:

```bash
devtools lab-scenario run archive-smoke --tier 0
devtools lab-scenario verify-baselines
```

`polylogue audit` remains the product/archive QA command for schema audit and
artifact proof checks.
