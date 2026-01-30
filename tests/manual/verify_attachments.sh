#!/usr/bin/env bash
# Verify attachment storage and references

set -euo pipefail

DB_PATH="${1:-$HOME/.local/share/polylogue/polylogue.db}"

if [ ! -f "$DB_PATH" ]; then
    echo "❌ Database not found: $DB_PATH"
    exit 1
fi

echo "Verifying attachments..."
echo ""

sqlite3 "$DB_PATH" << 'EOF'
.mode column
.headers on

-- Attachment statistics
SELECT 'Attachment Count:' as check, COUNT(*) as total FROM attachments;

-- Attachment references
SELECT 'Attachment Refs:' as check, COUNT(*) as total FROM attachment_refs;

-- Orphaned attachment refs (refs without attachment)
SELECT 'Orphaned Refs:' as check, COUNT(*) as count FROM attachment_refs ar
WHERE NOT EXISTS (SELECT 1 FROM attachments a WHERE a.attachment_id = ar.attachment_id);

-- Unreferenced attachments (attachments with no refs)
SELECT 'Unreferenced Attachments:' as check, COUNT(*) as count FROM attachments a
WHERE NOT EXISTS (SELECT 1 FROM attachment_refs ar WHERE ar.attachment_id = a.attachment_id);

-- Sample attachments
SELECT 'Sample Attachments:' as check, attachment_id, mime_type, size_bytes, path
FROM attachments
LIMIT 5;
EOF

echo ""
echo "✓ Attachment verification complete"
