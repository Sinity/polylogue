<!-- Turn 19 (mislabeled role=user in capture) is a near-identical streaming duplicate of turn 20 (with a trailing 'Sources' footer); turn 20 is used as canonical since it is properly role=assistant. -->

## Turn 20 (role=assistant)

I inspected the newly attached archives for Sinnix, Lynchpin, and Sinex as fresh current-state packages. This analysis is standalone and assumes the publication question is not “are they polished products,” but “what public shape is feasible, what must change, and which risks are worth accepting.”
My overall read:
Repo	Best publication shape	Feasibility	Main blocker class
Sinnix	Public dotfiles / workstation configuration	High	Exposure hygiene and history/ref cleanup
Lynchpin	Public code framework with private data substrate removed	Medium-high	Personal-derived history and identity/config hardcoding
Sinex	Public pre-alpha dogfood infrastructure repo	Medium	Secrets/agent artifacts, trust posture, open correctness work, overwhelming public surface
The correct move is not to make all three look like normal open-source products. They should publish as different species. Sinnix should look like dotfiles. Lynchpin should look like a local-first parser/evidence framework. Sinex should look like a serious pre-alpha event substrate, dogfooded by its author, not a stable tool for strangers’ life data yet.
GitHub-side baseline: all public repos should at least have a useful README, license, contribution/security posture, and sane issue templates. GitHub’s community profile checklist explicitly checks for recommended community health files such as README, LICENSE, CONTRIBUTING, and CODE_OF_CONDUCT, and GitHub’s README guidance says the README should explain why the project is useful, what people can do with it, and how to use it.
GitHub Docs
+1
 For security, public repos get secret scanning automatically for supported patterns, push protection is designed to block secrets before they are pushed, CodeQL default setup is the low-friction code scanning path, and Dependabot security updates raise PRs for vulnerable dependencies.
GitHub Docs
+3
GitHub Docs
+3
GitHub Docs
+3
The right publication model
These three repos are a stack:
Sinnix is the machine/control-plane layer: NixOS, Home Manager, desktop workflow, local AI/agent tooling, observability, capture service wiring, host policy, secret integration, deployment assumptions.
Lynchpin is the interpretation layer: source adapters over local exports/captures, materialization, evidence graphs, analysis, MCP surfaces, current-state packs.
Sinex is the event substrate: local-first capture, event contracts, schema/runtime machinery, replay/settlement, source catalog, NixOS module, CLI/API/MCP surfaces.
That stack is interesting. But it should not be published as one giant window into your entire operating life. Each repo needs a membrane.
The membrane should be:
public mechanisms
public docs
synthetic examples
dogfood facts you consciously accept
not:
private generated reports
real secrets/ciphertexts
agent scratchpads
raw or derived personal data
private issue archaeology
This is especially important because these repos are about local capture and personal data. The code is the product; the life it parsed is not.
Sinnix
Current state
Sinnix is already the most publishable of the three.
The current snapshot presents itself as a NixOS/dotfiles repo. The README says encrypted secrets live outside the checkout, and the current modules/secrets.nix points to /realm/data/secrets/sinnix/secret, which is materially safer than tracking real ciphertext under the repo. The current working tree has a clear structure: flake/, modules/, hosts/, dots/, scripts/, pkgs/, docs/, assets/, and agent-facing orientation files.
Concrete snapshot signals:
Current working tree scan: about 460 files.
Current clone: 467 commits on master, 631 commits across all refs.
GitHub context: 0 open issues, 0 open PRs, 98 merged PRs in the archive.
Beads context: 57 issues, 27 open, 20 ready, 7 blocked.
Community surface: README.md, LICENSE, issue templates, PR template, and a manual-only CI workflow exist.
Missing public-health files: no root CONTRIBUTING.md, SECURITY.md, or CODE_OF_CONDUCT.md.
The main caveat: the bundle HEAD and the exported working tree are not the same publication surface. The bundle still tracks .agent/ops/2026-05-02-stability.md, .beads, .mcp.json, flake.lock, and uv.lock; the working-tree export omits/deletes those. That strongly suggests the public version should be built from the current intended tree plus a clean history, not by flipping every bundled ref public.
What is publishable
Most of Sinnix should be public if the intent is dotfiles.
Keep:
flake.nix
flake/
modules/
hosts/             # if you accept host/topology disclosure
dots/
scripts/
pkgs/
docs/
assets/
AGENTS.md / CLAUDE.md after reference cleanup
.github issue templates and PR template
This is useful as dotfiles because it shows how the machine is organized: NixOS module taxonomy, Home Manager dotfiles, desktop ergonomics, Hyprland/qutebrowser/Noctalia setup, Sinex integration, capture hooks, resource policy, agent tooling, and local observability.
Do not over-generalize it. A public Sinnix that says “this is my workstation configuration” is more honest than one pretending to be a reusable NixOS framework.
What is risky
The main risk is not plaintext credentials. I did not find obvious high-confidence current plaintext GitHub/OpenAI/Anthropic/AWS/Slack/Google-token shapes in the current working tree scan. The risk is operational map disclosure.
Specific exposure surfaces:
hostnames: sinnix-prime, sinnix-ethereal, sinnix-gw
router/OpenWrt config
LAN/storage maps
agent permission posture
/realm filesystem conventions
capture scripts and browser hooks
secret names in module metadata
This is acceptable for dotfiles if deliberate. Many public dotfiles expose machine layout. The question is whether you are comfortable with this exact layout being indexed.
Higher-risk paths from the bundle/history:
.agent/ops/
.agent/scratch/
.beads/
.mcp.json
historical .claude/.agent includes
historical secret manifests
old config-audit material
.agent/ops/2026-05-02-stability.md is the big current-head wart. It is not a dotfile; it is an operational incident diary. It should not be reachable in public history.
AGENTS/CLAUDE also need one content pass. The current AGENTS.md says deep architecture and machine maps live in .agent/scratch/, but .agent/scratch should not be public. Either remove that reference or move sanitized architecture maps into docs/.
Publication options for Sinnix
Option A: clean public dotfiles repo. This is my recommendation. Keep real structure, clean history, and publish as dotfiles. Do not keep GitHub PR/issue archaeology unless you value it.
Option B: make current private GitHub repo public after history rewrite. Feasible, but only if you are willing to force-push a cleaned branch and lose or confuse old PR references. Since you never intended Sinnix to be PR-flow-driven, this is fine.
Option C: public “kit” split. Still interesting, but probably unnecessary now. The current README already says dotfiles; let it be dotfiles.
Concrete Sinnix cleanup before public
Remove from reachable history:
.agent/ops/
.agent/scratch/
.beads/
.mcp.json
old root secret manifests
old encrypted secret files if any remain in history
old .claude local settings/scratch
key-recovery docs
Keep or consciously accept:
hosts/
router config
/realm paths
Sinex/Lynchpin/Polylogue integration references
agent tooling configs
Patch docs:
README.md: keep as dotfiles, add “not a reusable framework”
AGENTS.md/CLAUDE.md: remove references to non-public .agent/scratch
docs/: add a short “publication boundary” only if useful, not as the README focus
Add:
CONTRIBUTING.md       # short; “personal repo, issues welcome, patches maybe”
SECURITY.md           # report channel, no support guarantee
CODE_OF_CONDUCT.md    # or account-level default community file
.github/dependabot.yml if you want dependency/version updates
Validation before public:
Bash
nix flake check        # or your curated check wrapper
nix fmt -- --fail-on-change
pytest -q pkgs/sinnix-observe/tests
gitleaks detect --source . --verbose
trufflehog git file://$PWD
Sinnix verdict
Publish first, if you want a low-risk public footprint. Use a fresh main branch or new repo, not all current refs. Keep the dotfiles personality. Remove private ops state. Do not waste effort making it look generic.
Lynchpin
Current state
Lynchpin is publishable as code, not as a complete current/private history.
Concrete snapshot signals:
Current working tree: about 805 files.
Bundle: 90 commits on master, 386 commits across all refs.
GitHub context: none, because it is not on GitHub.
Beads: 4 total, all closed.
No root LICENSE, no .github, no workflows, no contribution/security files.
pyproject.toml packages lynchpin version 0.1.0, Python >=3.12, with many heavy/default dependencies.
The snapshot audit says ignored local runtime state under .lynchpin is about 51 GB. That is not in the repo, but it tells you the repo is designed around a large private substrate.
The current working tree already removed some private directories from the top level, but the code and docs still reference thoughtspace, raw_log, health/substance/keylog/personal signals, and private operator identifiers. The history contains much more: deleted .agents/narrativization, checkpoint baselines, docs/personal, docs/analysis, life narratives, raw activity samples, and private reports. That history must not go public.
What is publishable
The code architecture is strong. Keep most parser and framework code:
lynchpin/core/
lynchpin/sources/
lynchpin/ingest/
lynchpin/substrate/
lynchpin/graph/
lynchpin/analysis/        # after privacy review
lynchpin/cli/
lynchpin/mcp/
tests/                    # after fixture audit
docs/reference/           # after removing private/narrative docs
The source-adapter list is not itself a leak if you are comfortable saying “this code can parse these export types.” It is fine to publish adapters for:
ActivityWatch
Atuin
browser history
Gmail Takeout
Google exports
Spotify
Reddit
Wykop
IRC
SVN/git
health/sleep exports
keylog/clipboard sources
local repo telemetry
The important distinction is: publish parsers and synthetic fixtures, not real-derived conclusions.
What is risky
Lynchpin’s risk is not mainly credentials. It is personal-derived inference.
High-risk categories in current code/history:
thoughtspace source and evidence graph
raw_log source
health/substance analysis
keylog analysis
personal-interest fusion
personal signals
Gmail operator-address constants
private SVN/default author names
deleted docs/personal and docs/analysis history
deleted checkpoint/baseline history
deleted .agents/narrativization reports
Some of these can remain as generic modules if renamed or documented carefully. For example, a raw_log parser can become an operator_log parser. A thoughtspace source can become a generic notespace source. Health parsers can remain if fixtures are synthetic and docs do not imply this repo contains your real health data.
The current code still has a real operator-address constant in lynchpin/sources/gmail_takeout.py, plus private author defaults in SVN-related code. Those should become configuration, not source literals.
Public contract problem
Lynchpin currently lacks a public “reader contract.” The README is concise, but it still talks about thoughtspace as a moved external source and assumes /realm. CLAUDE is still internal-agent oriented. There is no license, no public contribution policy, no CI, and no synthetic first-run demo.
That is okay for a private research repo. It is not okay for public.
A public Lynchpin README should say:
Lynchpin is a local-first Python toolkit for parsing local activity captures,
provider exports, repository history, and machine telemetry into typed source
records, DuckDB materializations, evidence graphs, and context packs.
This public repo contains code and synthetic fixtures. Real local data roots,
generated reports, private notes, and operator-specific config live outside it.
It should not say “I removed private stuff.” It should just describe the project.
Publication options for Lynchpin
Option A: fresh public lynchpin repo from cleaned history. Best option. The repo was never public, so there is no reason to preserve private historical blobs. Build a clean history that keeps code evolution but excludes private substrate.
Option B: public lynchpin-core, private sinity-lynchpin. Safest architectural split. Public repo contains code; private repo contains real config, generated context packs, .lynchpin, real source roots, and operator reports.
Option C: publish as research/dogfood code with source names intact. Feasible if you are comfortable exposing source categories. You still must strip actual private-derived docs/history.
I would choose B in practice, even if the public repo is named simply lynchpin.
Concrete Lynchpin cleanup before public
Remove from reachable history:
thoughtspace/                       # actual directory/history
.agent/
.agents/
.claude/
.lynchpin/
checkpoints/
results/
docs/personal/
docs/analysis/
docs/analysis-log.md
private narrative packets
private generated reports
raw exported samples
current-state packs from real data
Decide whether to rename these public concepts:
thoughtspace -> notespace
raw_log -> operator_log
personal_signals -> local_signals
Move to config:
operator email addresses
operator usernames
SVN author defaults
Wykop username
local project roots
/realm data roots
Dependency cleanup:
The default install currently pulls in heavy and live-tooling packages: claude-agent-sdk, mcp, duckdb, hmmlearn, numpy, scipy, tree-sitter, polylogue, etc. Split extras:
TOML
dependencies = [
  "duckdb",
  "polars",
  "pyyaml",
  "typer",
  "rich",
]
[project.optional-dependencies]
dev = ["pytest", "ruff", "mypy", "black"]
agents = ["mcp", "claude-agent-sdk", "tiktoken"]
ml = ["numpy", "scipy", "hmmlearn"]
code = ["tree-sitter", "tree-sitter-python", "tree-sitter-rust"]
personal = ["polylogue"]
Add synthetic fixtures:
tests/fixtures/synthetic_activitywatch/
tests/fixtures/synthetic_gmail_takeout/
tests/fixtures/synthetic_browser_history/
tests/fixtures/synthetic_health/
tests/fixtures/synthetic_git/
examples/synthetic_config.toml
Add public checks:
Bash
python -m compileall lynchpin tests
pytest -m "not slow"
ruff check lynchpin tests
mypy lynchpin/core lynchpin/sources lynchpin/graph
Add community files:
LICENSE
SECURITY.md
CONTRIBUTING.md
CODE_OF_CONDUCT.md
.github/workflows/ci.yml
Lynchpin verdict
Publishable after a clean-history extraction. The code is worth publishing. The old private history is not. Do not flip an existing full repo public; create a clean public repo that contains source adapters, graph/substrate code, synthetic fixtures, and public docs.
Sinex
Current state
Sinex is the most technically impressive and the most publication-sensitive.
Concrete snapshot signals:
Current working tree: about 2,173 files.
Bundle: 1,688 commits on master, 1,788 commits across all refs.
GitHub context: 6 open issues, 0 open PRs, 1,398 merged PRs.
Beads: 311 issues, 236 open, 100 ready, 102 blocked.
Tokei attribution shows enormous test and agent/demo surfaces: test-suite around 218k lines, code buckets for runtime/API/event-engine/db/primitives/CLI/xtask, and an agent-demos bucket of nearly 500k lines, mostly JSON.
Community files: README.md, CONTRIBUTING.md, issue templates, PR template, many workflows.
Missing: root LICENSE, SECURITY.md, CODE_OF_CONDUCT.md, CHANGELOG.md.
Cargo workspace metadata says license MIT and repository https://github.com/sinex/exocortex, while README points to github.com/sinity/sinex. That mismatch should be fixed.
Sinex has real public substance. It also has real public blast radius.
What is publishable
Keep the core project:
crate/
xtask/
schemas/
tests/
docs/
design/
nixos/modules/
nixos/examples/
.github/       # after workflow/permission review
README.md
CONTRIBUTING.md
TESTING.md
The event substrate is the public value: provenance model, typed event contracts, schema convergence, NATS/Postgres/Timescale/pgvector stack, source catalog, runtime settlement/replay, CLI/API, NixOS module, test culture.
The source catalog is a strength. The generated catalog has 66 entries and clear privacy-tier metadata: 50 sensitive, 11 public, 5 secret by contract.privacy_tier. That public fact is acceptable if the project is explicitly local-first and privacy-aware. It tells the truth about what Sinex is: a substrate for personal machine/event history.
What is risky
Sinex currently tracks too much internal and deployment state at HEAD.
High-risk tracked current paths include:
.agent/
.claude/
.beads/
.sinex/
nixos/secret/*.age
nixos/secret/secrets.nix
nixos/secret/sample-admin-token
demo/sinex-recall/sample-output.txt
Cargo.lock / flake.lock if the public working-tree export intentionally omits them
The nixos/secret directory is the clearest “do not publish as-is” path. It contains real encrypted .age files for local/remote DB and NATS credentials, plus an API admin token secret, a real recipient map, and public SSH key labels for your hosts. Encrypted age files are not plaintext, but public ciphertext plus service names plus recipient keys is still a future-decryption and topology risk. Replace it with nixos/secret.example/.
The .agent tree is huge: the agent audit reports about 79.8 MB tracked hidden files/directories, including .agent/demos, .agent/archive/devloop-2026-07, recall artifacts, live baselines, scratch, tools, scripts, and operational logs. That should not be public history. Some of it can become real docs, but it should not be shipped as agent substrate.
The recall demo is valuable but currently too close to live dogfood output. The sample output contains /realm paths and real event-history flavor. Replace it with synthetic data. Keep the demo concept.
The root agent docs are too private-operational. CLAUDE.md/AGENTS.md describes live deployment, private operator assumptions, production doctrine, and agent autonomy. Some of that is useful for you; public readers need a calmer architecture/editing guide. Move private operating doctrine to a private repo or keep a sanitized public version.
Current open issues
The six open GitHub issues are not all blockers, but they matter for public positioning:
#1043 media capture: audio, screen video, OCR streams
#1469 email: staged mailbox parser + live Gmail/IMAP sync
#2182 large historical imports: compression/startup/backlog robustness
#2184 source-material granularity and defragmentation
#2185 historical-import pacing and progress visibility
#2196 schema-apply convergence: duplicate CHECK constraints + required indexes
I would treat them this way:
Public blockers before calling Sinex runnable for others:
#2182 large historical import robustness
#2196 schema convergence correctness/performance
Public pre-alpha blockers unless clearly caveated:
#2184 source-material granularity
#2185 historical-import pacing
Roadmap, not blockers:
#1043 media capture
#1469 live email/Gmail/IMAP
The previous ContentKey/CAS traversal concern appears to have been addressed in the current code: ContentKey now validates local BLAKE3 digests as exactly 64 lowercase hex characters, requires size metadata for the local CAS backend, and tests reject traversal-like digests. Good. Do not carry forward stale security claims in public docs; say what is fixed and what remains open.
Trust posture problem
Sinex’s public trust posture is currently inconsistent.
On one hand, the code has serious security/trust machinery:
privacy tiers
private mode
source contracts
schema validation
replay/settlement logic
DLQ behavior
NixOS module secret resolution
test/fuzz/schema workflows
CodeRabbit/GitGuardian config
On the other hand, the repo currently includes:
real encrypted deployment secrets
agent operational archives
real-ish demo outputs
private deployment references
huge issue/bead/task surfaces
pre-release single-operator assumptions
no root SECURITY.md
no root LICENSE despite Cargo MIT metadata
That mix is the problem. It is a good project with a messy boundary.
CI/workflows
Sinex has many workflows, but most are manual-only. A few have broader triggers, for example verify-closure on issues and verify-perf on pushes to master for specific paths. Manual-heavy CI is understandable for a Nix/Rust/Postgres/NATS project with cost concerns, but public readers need to know what a green state means.
For public release:
Keep expensive Nix/VM/fuzz workflows manual.
Add one cheap automatic PR/push gate.
Make workflow permissions least-privilege and explicit. Several already do this, which is good.
Keep issue-writing workflows only if you are comfortable with public automation comments.
Add CodeQL default setup or an explicit CodeQL workflow.
GitHub Docs
A cheap public gate could be:
Bash
cargo fmt --check
cargo check --workspace --all-targets
cargo test -p sinex-primitives
cargo test -p sinex-db --lib
cargo test -p sinex-schema --lib
Maybe full cargo test --workspace is too costly. That is fine. Say so.
Publication options for Sinex
Option A: public pre-alpha dogfood repo. Best option if you want public technical credibility soon. Clean secrets/agent/demo surfaces, keep code/tests/docs, label known issues, and publish as pre-alpha. No releases yet.
Option B: public core/private ops split. Best long-term shape. Public sinex contains crates, schemas, docs, tests, synthetic demos, NixOS modules/examples. Private sinex-ops contains real NixOS deployment, age ciphertext, .agent, live dogfood reports, source bindings, generated artifacts.
Option C: private code, public architecture/case-study first. Lowest risk, but wastes the code’s credibility. I would only do this if you do not want cleanup work now.
Option D: make the current private GitHub repo public. I do not recommend this. It would expose too much metadata, tracked agent state, encrypted deployment topology, and old refs.
Because Sinex already has 1,398 merged PRs and many issues, there is a real question: keep GitHub history or start clean? I would not preserve GitHub PR history at all costs. It is impressive, but it is also noisy and likely exposes internal context. A clean public repo with curated history is safer. Keep the private repo as the full dogfood mirror.
Concrete Sinex cleanup before public
Remove from reachable history:
.agent/
.claude/
.beads/
.sinex/
nixos/secret/
demo/sinex-recall/sample-output.txt
agent demos/artifacts generated from real dogfood data
old worktree-agent/archive refs
git-annex ref unless intentionally public
era/keep refs unless you want them public
Replace with:
nixos/secret.example/
demo/sinex-recall/synthetic-output.txt
demo/synthetic/
docs/source-catalog.md
docs/threat-model.md
SECURITY.md
LICENSE
CODE_OF_CONDUCT.md
CHANGELOG.md or RELEASES.md
Fix metadata:
Cargo.toml repository = "https://github.com/Sinity/sinex"
root LICENSE file matching Cargo license
README status = pre-alpha dogfood
README quickstart = synthetic-first
AGENTS/CLAUDE = public architecture/editing guide, not live operator doctrine
Triage public issues:
label #2182 as public-blocker/runtime-stability
label #2196 as public-blocker/schema-correctness
label #2184 and #2185 as pre-alpha correctness/UX
label #1043 and #1469 as roadmap/source-packages
Add a synthetic first-run:
Bash
nix develop
xtask demo synthetic
xtask infra start --profile synthetic
sinexctl events recent --profile synthetic -n 20
The exact commands can differ, but public users need one path that does not require your NATS certs, Postgres secrets, browser history, ActivityWatch DB, /realm, or real captures.
Sinex verdict
Publishable, but only as cleaned pre-alpha. Sinex has the most public value and the most public risk. It should probably not be the first repo you flip public unless you are willing to spend real cleanup energy.
Cross-repo publication sequence
I would publish in this order.
1. Sinnix first
Sinnix gives you a public footprint with the least conceptual risk. It is dotfiles; people understand that dotfiles are personal. Do a fresh clean branch, remove .agent/.beads/secrets history, fix AGENTS references, add SECURITY.md and CONTRIBUTING.md, then publish.
This also creates a public home for the machine-layer assumptions that Sinex and Lynchpin reference.
2. Lynchpin second
Lynchpin should come next because it is a coherent Python project once the private substrate is removed. It will benefit from a clean README and synthetic fixtures. It does not need GitHub issue migration because it has no GitHub history.
This is the best repo for demonstrating the “parser/evidence graph” layer without the massive operational weight of Sinex.
3. Sinex third
Sinex should go public once the boundary is clean and the pre-alpha status is crisp. It is the crown jewel and the hedgehog: impressive, spiky, and not something you want rolling into public with secret ciphertext stuck in its quills.
I would not wait for Sinex to be “finished.” I would wait for it to be honestly unfinished.
Concrete release gates
Minimum gate for all three
Before making anything public:
Bash
git log --all --name-only | rg '(^|/)(secret|secrets|\\.agent|\\.agents|\\.claude|thoughtspace|checkpoints|results|docs/personal|docs/analysis)'
gitleaks detect --source . --verbose
trufflehog git file://$PWD
git fsck --no-reflogs
Then enable GitHub-side secret scanning, push protection, CodeQL where relevant, and Dependabot security updates.
GitHub Docs
+3
GitHub Docs
+3
GitHub Docs
+3
Sinnix gate
Bash
nix fmt -- --fail-on-change
nix flake check or curated `check`
pytest -q pkgs/sinnix-observe/tests
Plus manual review:
hosts/sinnix-gw/
docs/incidents/
AGENTS.md
CLAUDE.md
modules/secrets.nix
agent/codex/claude dotfiles
Lynchpin gate
Bash
python -m compileall -q lynchpin tests
pytest -m "not slow"
ruff check lynchpin tests
Plus manual review:
operator emails/usernames
thoughtspace/raw_log naming
health/substance/keylog fixtures
docs/personal and docs/analysis history
source roots in config.py
Sinex gate
Bash
cargo fmt --check
cargo check --workspace --all-targets
cargo test -p sinex-primitives
cargo test -p sinex-db --lib
cargo test -p sinex-schema --lib
xtask schema strict-diff
Plus manual review:
nixos/secret/
.agent/
demo/sinex-recall/
AGENTS.md / CLAUDE.md
Cargo.toml repository/license
open issues #2182 and #2196
workflow permissions
What I would not do
I would not publish all refs for any of them.
I would not preserve Sinnix PR history if it complicates the clean launch. It was not meant as a public PR-flow repo.
I would not publish Lynchpin’s current historical repository, even if HEAD looks cleaner. Its deleted history contains exactly the kind of private-derived material that should not become public archaeology.
I would not publish Sinex’s private GitHub repo as-is. The repo is too large, too operational, and too rich in private agent/deployment material for a simple visibility flip.
I would not remove every personal quirk. The weirdness is part of the value. The goal is not sterile. The goal is intentional.
Final recommendation
Publish all three, but not the same way.
Sinnix: yes, as public dotfiles, after a light-but-real history and exposure cleanup. Highest feasibility.
Lynchpin: yes, as a clean public code repo with synthetic fixtures and private substrate removed. Medium-high feasibility.
Sinex: yes, but as a cleaned pre-alpha dogfood infrastructure repo, not as stable software. Medium feasibility because the project is strong but the publication boundary is currently messy.
The public story should be:
Sinnix: how the machine is declared.
Lynchpin: how local evidence is parsed and analyzed.
Sinex: how local events become a durable substrate.
That is a compelling trilogy. It just needs doors, not glass walls.

---

