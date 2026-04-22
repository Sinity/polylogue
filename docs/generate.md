[← Back to README](../README.md)

# Generate: Synthetic Data

Polylogue includes a built-in synthetic data generator for exploring features without importing real conversation data.

## Quick Start

```bash
# Create a full demo environment
eval "$(devtools lab-corpus seed --env-only)"

# Now all commands work against synthetic data
polylogue                          # Archive stats
polylogue "error handling"         # Search
polylogue -p claude-ai --latest       # Latest Claude conversation
polylogue dashboard                # Interactive TUI
```

## Two Modes

### Default (Corpus) Mode

Writes raw provider-format files (JSON, JSONL) to disk without processing them. Useful for:

- Inspecting wire formats from each provider
- Testing parser changes
- Generating fixture data

```bash
# All providers, 3 conversations each
devtools lab-corpus generate

# ChatGPT only, 5 conversations, custom output
devtools lab-corpus generate -p chatgpt -n 5 -o /tmp/corpus
```

### `seed` Mode

Creates a complete demo environment: a temporary database seeded with synthetic conversations, rendered output files, and a search index. Prints environment variables that redirect Polylogue to the demo data.

```bash
# Interactive — prints env vars and instructions
devtools lab-corpus seed

# Shell integration — eval sets env vars in current shell
eval "$(devtools lab-corpus seed --env-only)"

# Custom options
devtools lab-corpus seed -p chatgpt -p claude-ai -n 10
```

The seeded environment uses the same pipeline as `polylogue run`, so the lab corpus command exercises the full acquire → parse → render → index flow, with validation performed inline during parse.

## Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--provider` | `-p` | all | Providers to include (repeatable) |
| `--count` | `-n` | 3 | Conversations per provider |
| `--output-dir` | `-o` | auto | Output directory |
| `--env-only` | | | Print export statements only for `seed` |

## How It Works

`devtools lab-corpus` uses `SyntheticCorpus` from `polylogue.schemas.synthetic`, which generates realistic conversation structures for each supported provider:

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
- `raw_synthetic_samples` — Raw conversation data for unit tests

This means `devtools lab-corpus` exercises the same schema-driven generation paths as the test suite fixtures.

---

**See also:** [CLI Reference](cli-reference.md) · [Library API](library-api.md) · [Configuration](configuration.md)
