# Live Beads application notes

This package does not assume a specific `bd` mutation syntax beyond the commands visible in the repo guidance. Use `patch_manifest.json` as the authority and translate these operations into your installed Beads CLI version.

Suggested operation order:

1. Import/review acceptance criteria patches for empty fields.
2. Add delivery labels.
3. Add hard `blocks` dependency edges.
4. Append delivery notes; do not replace existing notes.
5. Add the five durable memories.
6. Run `bd ready --json`, backlog lint, and the validator script.

Example pseudo-commands to translate, not a shell script:

```text
bd update polylogue-rsad --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-2qx --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-37t.5 --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-rii.2 --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-fs1.4 --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-scd --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-3tl.10 --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-3tl.8 --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-bfv --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
bd update polylogue-0aj --acceptance '<proposed AC from patches/acceptance_criteria_patches.json>' --json
...
bd dep add polylogue-b5l polylogue-1xc.8 --type blocks  # blue-green rebuild must wait for throughput envelope, rebuild-safety, blob-GC safety, and restore proof
bd dep add polylogue-b5l polylogue-8jg9.4 --type blocks  # blue-green rebuild must wait for throughput envelope, rebuild-safety, blob-GC safety, and restore proof
bd dep add polylogue-b5l polylogue-8jg9.2 --type blocks  # blue-green rebuild must wait for throughput envelope, rebuild-safety, blob-GC safety, and restore proof
bd dep add polylogue-b5l polylogue-4be --type blocks  # blue-green rebuild must wait for throughput envelope, rebuild-safety, blob-GC safety, and restore proof
bd dep add polylogue-83u.5 polylogue-83u.4 --type blocks  # blob compression waits until missing byte debt and acquisition coverage are classified/proven
bd dep add polylogue-83u.5 polylogue-83u.2 --type blocks  # blob compression waits until missing byte debt and acquisition coverage are classified/proven
bd dep add polylogue-83u.5 polylogue-83u.3 --type blocks  # blob compression waits until missing byte debt and acquisition coverage are classified/proven
bd dep add polylogue-83u.5 polylogue-83u.6 --type blocks  # blob compression waits until missing byte debt and acquisition coverage are classified/proven
bd dep add polylogue-bby.15 polylogue-rxdo.1 --type blocks  # evidence basket/export requires object refs, query identity, one read contract, and drift-aware citations
bd dep add polylogue-bby.15 polylogue-rxdo.2 --type blocks  # evidence basket/export requires object refs, query identity, one read contract, and drift-aware citations
...
```
