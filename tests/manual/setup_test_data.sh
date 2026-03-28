#!/usr/bin/env bash
# Setup test data for manual testing

set -euo pipefail

echo "Setting up manual test data..."

# Create test inbox structure
mkdir -p test-inbox/{chatgpt,claude,codex,claude-code}

# Copy small sample files instead of symlinking (safer for testing)
if [ -d "/realm/data/exports/chatlog/raw/chatgpt" ]; then
    find /realm/data/exports/chatlog/raw/chatgpt -name "*.zip" -type f | head -1 | xargs -I {} cp {} test-inbox/chatgpt/
    echo "✓ Copied ChatGPT sample"
fi

if [ -d "/realm/data/exports/chatlog/raw/claude" ]; then
    find /realm/data/exports/chatlog/raw/claude -name "*.zip" -type f | head -1 | xargs -I {} cp {} test-inbox/claude/
    echo "✓ Copied Claude sample"
fi

echo "✓ Test data setup complete in test-inbox/"
