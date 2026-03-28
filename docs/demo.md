[← Back to README](../README.md)

# Demo & Synthetic Data

Polylogue includes a built-in demo system for exploring features without importing real conversation data.

## Quick Start

```bash
# Create a full demo environment
eval $(polylogue demo --seed --env-only)

# Now all commands work against synthetic data
polylogue                          # Archive stats
polylogue "error handling"         # Search
polylogue -p claude --latest       # Latest Claude conversation
polylogue dashboard                # Interactive TUI
```

## Two Modes

### `--seed` Mode

Creates a complete demo environment: a temporary database seeded with synthetic conversations, rendered output files, and a search index. Prints environment variables that redirect Polylogue to the demo data.

```bash
# Interactive — prints env vars and instructions
polylogue demo --seed

# Shell integration — eval sets env vars in current shell
eval $(polylogue demo --seed --env-only)

# Custom options
polylogue demo --seed -p chatgpt,claude -n 10
```

The seeded environment uses `run_sources` — the same pipeline as `polylogue run` — so the demo exercises the full ingest → parse → render → index flow.

### `--corpus` Mode

Writes raw provider-format files (JSON, JSONL) to disk without processing them. Useful for:

- Inspecting wire formats from each provider
- Testing parser changes
- Generating fixture data

```bash
# All providers, 3 conversations each
polylogue demo --corpus

# ChatGPT only, 5 conversations, custom output
polylogue demo --corpus -p chatgpt -n 5 -o /tmp/corpus
```

## Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--seed` | | | Create full demo environment |
| `--corpus` | | | Generate raw fixture files |
| `--provider` | `-p` | all | Providers to include (repeatable) |
| `--count` | `-n` | 3 | Conversations per provider |
| `--output-dir` | `-o` | auto | Output directory |
| `--env-only` | | | Print export statements only |

## How It Works

Both modes use `SyntheticCorpus` from `polylogue.schemas.synthetic`, which generates realistic conversation structures for each supported provider:

- **ChatGPT**: `conversations.json` with UUID-based message graphs
- **Claude**: JSONL files with `chat_messages` arrays
- **Claude Code**: JSON arrays with `parentUuid`/`sessionId` markers
- **Codex**: Session-based JSONL exports
- **Gemini**: Structured `chunkedPrompt` format

The synthetic data includes realistic message patterns: multi-turn dialogues, code blocks, thinking traces, tool use, and attachments metadata.

## Relationship to Test Fixtures

The test suite uses the same `SyntheticCorpus` infrastructure through shared fixtures:

- `seeded_db` — A pre-populated database for integration tests
- `synthetic_source` — A temporary source directory with generated files
- `raw_synthetic_samples` — Raw conversation data for unit tests

This means the demo command exercises the exact same code paths as the test suite.

---

**See also:** [CLI Reference](cli-reference.md) · [Library API](library-api.md) · [Configuration](configuration.md)
