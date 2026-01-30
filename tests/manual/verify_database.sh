#!/usr/bin/env bash
# Verify database integrity and content

set -euo pipefail

DB_PATH="${1:-$HOME/.local/share/polylogue/polylogue.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "❌ Database not found: $DB_PATH"
    exit 1
fi

echo "Verifying database: $DB_PATH"
echo ""

# Run SQL checks
sqlite3 "$DB_PATH" << 'EOF'
.mode column
.headers on

-- Schema version
SELECT 'Schema Version:' as check, * FROM schema_version LIMIT 1;

-- Provider distribution
SELECT 'Provider Distribution:' as check, provider_name, COUNT(*) as count
FROM conversations
GROUP BY provider_name;

-- Orphaned messages check
SELECT 'Orphaned Messages:' as check, COUNT(*) as count FROM messages m
WHERE NOT EXISTS (SELECT 1 FROM conversations c WHERE c.conversation_id = m.conversation_id);

-- Empty conversations check
SELECT 'Empty Conversations:' as check, COUNT(*) as count FROM conversations c
WHERE NOT EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.conversation_id);

-- Sample conversations
SELECT 'Sample Conversations:' as check, conversation_id, provider_name, title
FROM conversations
LIMIT 5;
EOF

echo ""
echo "✓ Database verification complete"
