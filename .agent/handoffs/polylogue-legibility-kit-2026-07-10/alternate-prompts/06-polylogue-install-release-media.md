# Fork 06 — Polylogue install matrix, release channel, and public media

Work directly on the supplied Polylogue repository. Use Beads as roadmap authority, especially `polylogue-3tl.7`, `.5`, `.9`, and `.10`.

## Mission

Make the first-run and first-impression path reproducible from a clean machine. Do not advertise installation channels that are not actually verified.

## Owned scope

Own packaging/install documentation, clean-environment verification scripts or CI matrix, VHS/media tapes and generated public-safe recordings, launch packet assembly, and media drift checks. Avoid changing product semantics or demo fixture definitions.

## Required install matrix

Determine and verify the truth of at least:

- Nix `nix run` path;
- source checkout plus `nix develop` path;
- `uv`/Python source-install path if supported;
- wheel/sdist build and install in a clean virtual environment;
- unsupported or future channels such as PyPI, Homebrew, OCI, or browser stores.

The README must label each channel as supported, experimental, planned, or unavailable. One clean path is better than five aspirational commands.

## Media

Regenerate a slow, comprehensible public recording from deterministic fixtures. It should show:

1. one-command tour;
2. structural failure receipt;
3. semantic aggregation;
4. lineage view;
5. bounded report with “does not prove.”

Avoid tiny text, rapid cuts, full-screen JSON dumps, and private paths. Store the source tape and regeneration command.

## Verification

- run the install paths in isolated roots or containers where available;
- verify the generated artifacts contain no absolute paths, usernames, secrets, private repositories, or volatile timestamps;
- compare generated media/artifact hashes or enforce a drift check;
- record exact environment and duration;
- make failure messages actionable.

## Deliverables

Produce a patch, install matrix report, launch media, release-channel status table, validation transcript, and a concise list of remaining blockers to a first tagged release.
