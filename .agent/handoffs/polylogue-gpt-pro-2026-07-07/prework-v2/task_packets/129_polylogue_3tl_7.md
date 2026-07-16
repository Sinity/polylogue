# 129. polylogue-3tl.7 — Release is a decision: proven install matrix across package managers and OSes

Priority/type/status: **P2 / task / open**. Lane: **12-external-legibility-demos**. Release: **L-legibility-demos**. Readiness: **implementation-ready-after-local-inspection**.

## What the bead says

Release machinery exists (release-please, PyPI + Homebrew tap + GHCR + Nix flake wired in CI per the grok evidence) but 'wired' is not 'proven': nobody continuously verifies that a stranger's install actually works on each lane, so the first real user on each path is the test. The target state the operator named: everything prepared so that shipping is ONLY the decision to merge the release PR — no scramble, no 'does brew even work', no unknown-OS surprises.

## Existing design note

(1) INSTALL-MATRIX CI (scheduled weekly + pre-release, not per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, docker run from GHCR, nix run — on ubuntu + macos runners (arm+x86 where available); each job runs the same smoke: install -> polylogue demo seed -> one find -> one read -> version check. The demo corpus makes this stranger-equivalent. (2) WINDOWS: decide and STATE the story (native is untested; document WSL2 as the supported path honestly in README install section) — an honest 'WSL2 only' beats a broken native promise. (3) ARTIFACT HYGIENE: signed tags already; add sigstore/attestations for wheels + images if cheap; devtools release verify-distribution already checks entrypoints — wire it into the matrix. (4) VERSION SURFACES: polylogue --version correct on every lane (git-hash dev builds vs tagged releases). (5) AUR/extra distros: explicitly OUT until demand exists — the matrix file documents the supported set; adding a lane later is one job. Acceptance: matrix green two consecutive weekly runs; release checklist doc reduced to 'merge the release PR'.

## Acceptance criteria

1. Install-matrix CI workflow (scheduled weekly + pre-release, NOT per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, `docker run` from GHCR, and `nix run`, on ubuntu + macos runners (arm+x86 where available); each job runs the same smoke: install -> `polylogue demo seed` -> one `find` -> one `read` -> `polylogue --version` check. 2. Windows story stated honestly in the README install section (native marked untested; WSL2 documented as the supported path). 3. `devtools release verify-distribution` wired into the matrix; sigstore/attestations added for wheels + images if cheap. 4. `polylogue --version` is correct on every lane (git-hash dev builds vs tagged releases). 5. AUR/extra distros documented as explicitly out-of-scope in the matrix file (adding a lane later is one job). Verify: matrix green two consecutive weekly runs; the release checklist doc is reduced to 'merge the release PR'.

## Static mechanism / likely defect

Issue description localizes the mechanism: Release machinery exists (release-please, PyPI + Homebrew tap + GHCR + Nix flake wired in CI per the grok evidence) but 'wired' is not 'proven': nobody continuously verifies that a stranger's install actually works on each lane, so the first real user on each path is the test. The target state the operator named: everything prepared so that shipping is ONLY the decision to merge the release PR — no scramble, no 'does brew even work', no unknown-OS surprises. Design direction: (1) INSTALL-MATRIX CI (scheduled weekly + pre-release, not per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, docker run from GHCR, nix run — on ubuntu + macos runners (arm+x86 where available); each job runs the same smoke: install -> polylogue demo seed -> one find -> one read -> version check. The demo corpus makes this stranger-equivalent. (2) WINDOWS: decide and STATE the story (native is unteste…

## Source anchors to inspect first

- `README.md` — Public claims should be grounded through the claims ledger.
- `docs/agent-forensics.md` — Existing forensics docs are a pattern for proof artifacts.
- `docs/demo.md` — Demo docs should depend on evidence/citation machinery.
- `polylogue/core/refs.py` — Existing ref model should be extended, not bypassed.
- `polylogue/storage/sqlite/archive_tiers/user_write.py:901` — Findings should reuse assertion/candidate lifecycle.
- `polylogue/surfaces/payloads.py:747` — Surface payloads currently expose branch/message refs; expand consistently.
- `polylogue/mcp/payloads.py:377` — MCP message payloads carry variant/branch metadata today.
- `polylogue/archive/actions/followup.py` — Action/followup classification is a real structural analytics input.
- `polylogue/archive/actions/fields.py` — Action fields determine what can be measured without prose heuristics.
- `polylogue/insights/registry.py:294` — Insight registry should become measure/product registry input.
- `scripts/agent_forensics.py` — Existing forensics script is a proof artifact and candidate product surface.

## Implementation plan

1. (1) INSTALL-MATRIX CI (scheduled weekly + pre-release, not per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, docker run from GHCR, nix run — on ubuntu + macos runners (arm+x86 where available)
2. each job runs the same smoke: install -> polylogue demo seed -> one find -> one read -> version check.
3. The demo corpus makes this stranger-equivalent.
4. (2) WINDOWS: decide and STATE the story (native is untested
5. document WSL2 as the supported path honestly in README install section) — an honest 'WSL2 only' beats a broken native promise.
6. (3) ARTIFACT HYGIENE: signed tags already
7. add sigstore/attestations for wheels + images if cheap

## Tests to add

- Acceptance proof: 1.
- Acceptance proof: Install-matrix CI workflow (scheduled weekly + pre-release, NOT per-PR): fresh-environment jobs for uvx/pipx from PyPI, brew tap, `docker run` from GHCR, and `nix run`, on ubuntu + macos runners (arm+x86 where available)
- Acceptance proof: each job runs the same smoke: install -> `polylogue demo seed` -> one `find` -> one `read` -> `polylogue --version` check.
- Acceptance proof: 2.
- Acceptance proof: Windows story stated honestly in the README install section (native marked untested
- Acceptance proof: WSL2 documented as the supported path).
- Acceptance proof: 3.
- Acceptance proof: `devtools release verify-distribution` wired into the matrix

## Verification commands

- `devtools test <focused tests added for this bead>`
- `devtools verify --docs --demo`

## Pitfalls

- Do not publish a number or success claim until evidence tier, denominator, and caveat rendering exist.

## Expected end state

A coding agent can point to a focused failing test, a small patch at the shared seam, and a verification artifact proving the bead's acceptance criteria without inventing a new product direction.
