[← Back to README](../README.md)

# Generate: Synthetic Data

Polylogue includes a built-in synthetic data generator for exploring features without importing real session data.

For users, the supported entrypoint is `polylogue import --demo`: it generates
the approved deterministic fixture world and schedules it through the same
daemon-backed import path as ordinary source imports. For tests and repository
verification, the same generator is available through `polylogue.scenarios`,
shared fixtures, and `devtools lab smoke`.

## Quick Start

```bash
export POLYLOGUE_DEMO_HOME="$(mktemp -d)"
export POLYLOGUE_ARCHIVE_ROOT="$POLYLOGUE_DEMO_HOME/archive"
export XDG_CONFIG_HOME="$POLYLOGUE_DEMO_HOME/config"

polylogue init
polylogued run
```

In a second terminal with the same environment:

```bash
polylogue import --demo --wait --timeout 30 --with-overlays
polylogue ops status
polylogue analyze
polylogue find "pytest" then read --all --limit 5
polylogue find "pytest" then read --view messages
polylogue find "pytest" then analyze --facets

# Repository verification-lab smokes
devtools lab smoke run archive-smoke --tier 0
devtools test tests/unit/cli/test_demo_command.py tests/unit/demo/test_demo_seed_verify.py tests/visual
```

`polylogue import --demo` is a daemon scheduling command, not an in-process
archive build. It materializes approved fixture sources under the configured
archive root, stages them into the daemon inbox, and reports scheduling only
after the daemon accepts the staged source. Add `--wait` to block until the
daemon-built archive passes the semantic demo verifier before running the
search/read/analyze examples. Add `--with-overlays` with `--wait` to attach the
deterministic user-tier overlays after the target sessions exist.

## How It Works

`SyntheticCorpus` from `polylogue.scenarios` generates realistic session structures for each supported provider:

- **ChatGPT**: JSON documents with UUID-based message graphs (`mapping`)
- **Claude AI**: JSON documents with `chat_messages` arrays
- **Claude Code**: JSONL records with `parentUuid`/`sessionId` markers
- **Codex**: Session-style JSONL exports
- **Gemini**: Structured `chunkedPrompt` format

Synthetic generation includes realistic message patterns: multi-turn dialogues, code blocks, thinking traces, tool use, and attachments metadata.

## Relationship to Test Fixtures

The test suite uses the same `SyntheticCorpus` infrastructure through shared fixtures:

- `seeded_db` — A pre-populated database for integration tests
- `synthetic_source` — A temporary source directory with generated files
- `raw_synthetic_samples` — Raw session data for unit tests

`polylogue import --demo` uses the named `build_demo_corpus_specs()` fixture
world so README examples and release checks can run without private exports.
After that archive has converged, the repository-level
`seed_demo_user_overlays()` fixture attaches deterministic user-tier overlays
to the same sessions for release evidence: a `pytest-triage` tag, a starred
README example mark, a session-scoped blackboard note, a saved query, and
typed assertions in `user.db`.

## README Demo Evidence

The deterministic archive evidence for the current demo fixture world is split
across direct and daemon-backed checks. `tests/unit/scenarios/test_demo_archive_convergence.py`
writes the same `build_demo_corpus_specs()` artifacts, converges them in
process with `parse_sources_archive()`, and verifies:

- three sessions are stored: ChatGPT export, Claude Code session, and Codex
  session;
- nineteen messages are indexed;
- raw source paths remain relative (`chatgpt/demo-00.json`,
  `claude-code/demo-00.jsonl`, `codex/demo-00.jsonl`);
- `polylogue` full-text search for `pytest` is backed by the Claude Code demo
  session;
- deterministic user overlays are stored in `user.db`: the `pytest-triage`
  tag, session mark, blackboard note, saved query, and typed tag/decision
  assertions all target the Claude Code demo session;
- repeating convergence is idempotent for raw sessions, sessions, messages,
  blocks, and user overlay assertions.

`tests/integration/test_demo_daemon_convergence.py` exercises the shipped
daemon-backed surface: `polylogue import --demo --wait --with-overlays` stages
the fixture world, waits for live daemon convergence, seeds overlays only after
the sessions exist, and reports success only after the archive passes semantic
demo verification. The daemon path keeps absolute archive-local raw provenance
internally; the direct `polylogue demo verify` path remains the stricter
no-absolute-raw-path posture for fixture evidence.

---

**See also:** [CLI Reference](cli-reference.md) · [Library API](library-api.md) · [Configuration](configuration.md)
