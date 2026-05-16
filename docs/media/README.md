# docs/media

This directory holds **on-demand rendered** documentation media. Only assets
that have an executable render/freshness gate live here.

## Policy

User-facing media (screenshots of the CLI, terminal recordings, VHS tapes,
dashboard captures) are **not committed** to this repository. They drift
faster than the code: terminal sizes, theme colors, fonts, prompt strings,
sample data, and timestamps all change between renders, producing large diffs
that carry no semantic signal and date the documentation badly.

When such media is needed (for a release, a blog post, the website), it is
rendered on demand from the same showcase exercises and seeded environments
that back the test suite:

- `polylogue/showcase/` produces deterministic demo workspaces.
- `polylogue generate` + `polylogue qa --only exercises` reproduce the
  scenarios used in any rendered capture.
- `devtools render-readme-media` re-renders the diagrams committed here
  from their Mermaid sources.

## What lives here

The committed assets in this directory are architecture diagrams rendered
from Mermaid (`.mmd`) source files by
`devtools render-readme-media`. The `.mmd` sources are the authoritative
input; the `.svg` files are generated.

Verify they are up to date:

```bash
devtools render-readme-media --check
```

Regenerate after editing the diagram sources:

```bash
devtools render-readme-media
```

The Mermaid source files are part of the repository; the rendered SVGs are
committed because they are referenced from generated documentation surfaces
and rendering them on every reader's machine is impractical (requires
`mmdc`).

## Why no screenshots or VHS tapes

Both were considered during the verifiability revamp (#1064, #1082) and
explicitly excluded:

- No README content currently references screenshots or terminal recordings.
- A committed asset without a `--check`-style freshness gate is dead weight
  the manifest cannot honestly claim coverage over.
- The showcase + qa workflows already provide reproducible demo output,
  which is the source of truth when capture is actually needed.

If a future README change introduces a screenshot or VHS tape, the asset
must arrive together with:

1. A render script that regenerates it deterministically from a showcase
   exercise or seeded environment.
2. A `--check` mode that fails when the committed asset diverges from the
   freshly rendered output (the pattern used by `render-readme-media`).
3. A new row in `docs/plans/docs-media-coverage.yaml` pointing at that
   render/verify pair.

Until those three pieces exist, screenshots and VHS tapes stay out of the
repository.
