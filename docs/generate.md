[← Back to README](../README.md)

# Generate: Synthetic Data

Polylogue includes a built-in synthetic data generator for exploring features without importing real session data.

Synthetic generation is available through the `polylogue.scenarios` module and the test infrastructure. Use `devtools lab-scenario` for demo workspace seeding and verification-lab exercises.

## Quick Start

```bash
# Lab scenario with synthetic data
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

---

**See also:** [CLI Reference](cli-reference.md) · [Library API](library-api.md) · [Configuration](configuration.md)
