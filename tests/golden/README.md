# Golden Test Files

This directory contains **golden reference files** for snapshot/regression testing.

## Structure

Each test case is a subdirectory containing:
- `input/` - Test conversation data (database state or raw exports)
- `expected/` - Expected rendered output (markdown, HTML, file structure)
- `metadata.yaml` - Test case metadata (description, provider, edge cases tested)

## Test Cases

### chatgpt-simple
Basic ChatGPT conversation with user/assistant turns, no attachments.
Tests: basic markdown formatting, role prefixes, timestamps.

### claude-thinking
Claude conversation with `<thinking>...</thinking>` blocks.
Tests: XML tag preservation, thinking block formatting.

### chatgpt-attachments
ChatGPT conversation with file attachments.
Tests: attachment link generation, file structure.

### multi-provider
Conversations from multiple providers to test consistency.
Tests: provider-specific formatting, metadata handling.

### edge-cases
Conversations with unusual content (empty messages, Unicode, very long text).
Tests: edge case handling, escaping, truncation.

## Updating Golden Files

When intentionally changing output format:

```bash
python tests/scripts/update_golden.py
```

This regenerates all `expected/` files from current rendering code.

## Usage in Tests

```python
def test_basic_markdown_rendering():
    # Load test case
    conv_id = load_golden_conversation("chatgpt-simple")

    # Render
    rendered = render_to_markdown(conv_id)

    # Compare
    expected = read_golden_output("chatgpt-simple", "conversation.md")
    assert rendered == expected
```

## Guidelines

1. **Keep cases small** - Golden files should be human-reviewable
2. **Document edge cases** - Use metadata.yaml to explain what's being tested
3. **Review changes** - When updating golden files, diff them carefully
4. **Commit golden files** - They're part of the test suite, not gitignored
