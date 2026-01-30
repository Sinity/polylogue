#!/usr/bin/env bash
# Verify rendered filesystem output structure

set -euo pipefail

RENDER_ROOT="${1:-$HOME/.local/share/polylogue/render}"

if [ ! -d "$RENDER_ROOT" ]; then
    echo "❌ Render directory not found: $RENDER_ROOT"
    echo "   (This is OK if you haven't rendered yet)"
    exit 0
fi

echo "Verifying filesystem: $RENDER_ROOT"
echo ""

# Count providers
for provider in chatgpt claude claude-code codex gemini; do
    if [ -d "$RENDER_ROOT/markdown/$provider" ]; then
        count=$(find "$RENDER_ROOT/markdown/$provider" -name "conversation.md" -type f | wc -l)
        echo "  $provider: $count conversations"
    fi
done

echo ""

# Check for broken symlinks
echo "Checking for broken symlinks..."
broken=$(find "$RENDER_ROOT" -type l ! -exec test -e {} \; -print 2>/dev/null | wc -l)
if [ "$broken" -eq 0 ]; then
    echo "  ✓ No broken symlinks"
else
    echo "  ⚠️  Found $broken broken symlinks"
fi

echo ""
echo "✓ Filesystem verification complete"
