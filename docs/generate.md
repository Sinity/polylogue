[← Back to README](../README.md)

# Generate: Synthetic Data

Polylogue includes a built-in synthetic data generator for exploring features without importing real session data.

For users, the supported entrypoint is `polylogue import --demo`: it generates
the approved deterministic fixture world and schedules it through the same
daemon-backed import path as ordinary source imports. For tests and repository
verification, the same generator is available through `polylogue.scenarios`,
shared fixtures, and `devtools lab-scenario`.

## Quick Start

```bash
# User-facing demo source import
polylogue import --demo

# Repository verification-lab scenarios
devtools lab-scenario run archive-smoke --tier 0
devtools lab-scenario verify-baselines
```

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

---

**See also:** [CLI Reference](cli-reference.md) · [Library API](library-api.md) · [Configuration](configuration.md)
