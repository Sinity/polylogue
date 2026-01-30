# Polylogue Manual Testing Protocol

## Goal
Gain **utter confidence** that everything works as intended through systematic, multi-layered testing.

## Test Data Setup

### 1. Symlink Real Chatlogs to Inbox
```bash
# Create test structure
mkdir -p /realm/project/polylogue/test-inbox/{chatgpt,claude,codex,claude-code}

# Symlink actual data
ln -s /realm/data/exports/chatlog/raw/chatgpt/*.json /realm/project/polylogue/test-inbox/chatgpt/
ln -s /realm/data/exports/chatlog/raw/claude/*.json /realm/project/polylogue/test-inbox/claude/
ln -s ~/.codex/sessions/*.jsonl /realm/project/polylogue/test-inbox/codex/
ln -s ~/.claude/projects/*/*.jsonl /realm/project/polylogue/test-inbox/claude-code/
```

### 2. Fresh Database for Testing
```bash
# Backup production DB
cp ~/.local/share/polylogue/polylogue.db ~/.local/share/polylogue/polylogue.db.backup

# Start fresh
rm ~/.local/share/polylogue/polylogue.db
```

## Testing Methodology

### Layer 1: Import & Database Verification

#### Test Each Provider
```bash
# ChatGPT
polylogue import run /realm/project/polylogue/test-inbox/chatgpt --provider chatgpt --out /tmp/test-out-chatgpt

# Claude
polylogue import run /realm/project/polylogue/test-inbox/claude --provider claude --out /tmp/test-out-claude

# Codex
polylogue sync codex --base-dir /realm/project/polylogue/test-inbox/codex --out /tmp/test-out-codex

# Claude Code
polylogue sync claude-code --base-dir /realm/project/polylogue/test-inbox/claude-code --out /tmp/test-out-claude-code
```

#### Verify Database Contents
```bash
# Open DB and inspect
sqlite3 ~/.local/share/polylogue/polylogue.db

# Check schema version
SELECT * FROM schema_version;

# Count conversations per provider
SELECT provider, COUNT(*) FROM conversations GROUP BY provider;

# Check for orphaned messages
SELECT COUNT(*) FROM messages WHERE conversation_id NOT IN (SELECT conversation_id FROM conversations);

# Check for orphaned attachments
SELECT COUNT(*) FROM attachments WHERE conversation_id NOT IN (SELECT conversation_id FROM conversations);

# Sample conversation structure
SELECT * FROM conversations LIMIT 5;
SELECT * FROM messages WHERE conversation_id = (SELECT conversation_id FROM conversations LIMIT 1) LIMIT 10;
SELECT * FROM attachments WHERE conversation_id = (SELECT conversation_id FROM conversations LIMIT 1);

# Check metadata richness
SELECT conversation_id, provider, json_extract(provider_meta, '$.') FROM conversations LIMIT 10;

# Verify timestamps
SELECT conversation_id, created_at, updated_at FROM conversations WHERE created_at IS NULL OR updated_at IS NULL;
```

### Layer 2: Filesystem Layout Verification

#### Check Output Structure
```bash
# Each provider should have clean layout
tree -L 3 /tmp/test-out-chatgpt
tree -L 3 /tmp/test-out-claude
tree -L 3 /tmp/test-out-codex
tree -L 3 /tmp/test-out-claude-code

# Verify naming conventions
ls -la /tmp/test-out-chatgpt/*.md | head -20

# Check for reasonable slugs (no gibberish)
find /tmp/test-out-chatgpt -name "*.md" | grep -E '[0-9a-f]{8}-[0-9a-f]{4}' | head -10

# Should see readable titles in filenames
find /tmp/test-out-chatgpt -name "*.md" | head -10
```

#### Attachment Handling
```bash
# Find all attachment directories
find /tmp/test-out-* -type d -name "*attachments*"

# Check attachment references in markdown
grep -r "!\[" /tmp/test-out-chatgpt/*.md | head -20

# Verify attachments exist
for md in /tmp/test-out-chatgpt/*.md; do
  echo "=== $md ==="
  grep "!\[" "$md" | while read line; do
    # Extract path
    path=$(echo "$line" | sed -E 's/.*\(([^)]+)\).*/\1/')
    # Check if exists (relative to md file)
    if [ -f "$(dirname "$md")/$path" ]; then
      echo "  ✓ $path"
    else
      echo "  ✗ MISSING: $path"
    fi
  done
done | less
```

### Layer 3: Output Content Verification

#### Markdown Quality
```bash
# Pick random conversation, render to HTML, open in browser
RANDOM_MD=$(find /tmp/test-out-chatgpt -name "*.md" | shuf -n 1)
echo "Inspecting: $RANDOM_MD"

# View markdown
bat "$RANDOM_MD"

# View HTML (if exists)
HTML="${RANDOM_MD%.md}.html"
if [ -f "$HTML" ]; then
  firefox "$HTML" &
fi

# Check front matter
head -30 "$RANDOM_MD"

# Verify message structure
grep -n "^### " "$RANDOM_MD" | head -20

# Check for tool use sections
grep -A 5 "Tool Use" "$RANDOM_MD" | head -30
```

#### Content Richness Checks
```bash
# For each provider, sample 5 conversations
for provider in chatgpt claude codex claude-code; do
  echo "=== $provider ==="
  find /tmp/test-out-$provider -name "*.md" | shuf -n 5 | while read md; do
    echo "File: $md"
    echo "Lines: $(wc -l < "$md")"
    echo "Messages: $(grep -c "^### " "$md")"
    echo "Attachments: $(grep -c "!\[" "$md")"
    echo "Tool Use: $(grep -c "Tool Use" "$md")"
    echo "---"
  done
done
```

### Layer 4: Search & Index Verification

#### Build Index
```bash
polylogue maintain index --rebuild
```

#### Test Search
```bash
# Basic search
polylogue search "authentication" --json | jq '.results | length'

# Provider filter
polylogue search "error" --provider chatgpt --json

# Search with attachments
polylogue search "image" --with-attachments --json

# Search in specific conversation
CONV_ID=$(sqlite3 ~/.local/share/polylogue/polylogue.db "SELECT conversation_id FROM conversations LIMIT 1" | tr -d '\n')
polylogue search "test" --conversation-id "$CONV_ID" --json
```

### Layer 5: Semantic API Testing

Create test script:
```bash
cat > /tmp/test_semantic_api.py << 'EOF'
#!/usr/bin/env python3
from polylogue import Polylogue

archive = Polylogue()

# Test 1: Basic conversation retrieval
print("=== Test 1: Retrieve conversations ===")
convs = list(archive.filter().provider("chatgpt").limit(5).list())
print(f"Found {len(convs)} conversations")
for conv in convs:
    print(f"  {conv.conversation_id}: {conv.title}")

# Test 2: Message filtering
print("\n=== Test 2: Filter substantive messages ===")
if convs:
    conv = convs[0]
    messages = list(conv.messages())
    substantive = list(conv.substantive_only())
    print(f"Total messages: {len(messages)}")
    print(f"Substantive messages: {len(substantive)}")
    print(f"Ratio: {len(substantive)/len(messages):.1%}")

# Test 3: Classification
print("\n=== Test 3: Message classification ===")
if convs:
    conv = convs[0]
    for msg in conv.messages()[:10]:
        tags = []
        if msg.is_tool_use: tags.append("TOOL")
        if msg.is_thinking: tags.append("THINK")
        if msg.is_context_dump: tags.append("CONTEXT")
        if msg.is_substantive: tags.append("SUBSTANTIVE")
        print(f"{msg.role:10} {' '.join(f'[{t}]' for t in tags):30} {msg.content[:50]}")

# Test 4: Dialogue pairs
print("\n=== Test 4: Dialogue pairs ===")
if convs:
    conv = convs[0]
    for i, (user_msg, assistant_msg) in enumerate(conv.iter_pairs()):
        if i >= 3: break
        print(f"\nPair {i+1}:")
        print(f"  User: {user_msg.content[:80]}")
        print(f"  Assistant: {assistant_msg.content[:80]}")

# Test 5: Word counts
print("\n=== Test 5: Statistics ===")
if convs:
    conv = convs[0]
    print(f"Message count: {conv.message_count}")
    print(f"Word count: {conv.word_count}")
    print(f"Substantive word count: {conv.substantive_word_count}")
    print(f"Substantive ratio: {conv.substantive_word_count/conv.word_count:.1%}")

# Test 6: Text rendering
print("\n=== Test 6: Text rendering ===")
if convs:
    conv = convs[0]
    dialogue_text = conv.dialogue_only().to_clean_text()
    print(f"Clean dialogue ({len(dialogue_text)} chars):")
    print(dialogue_text[:500])

archive.close()
EOF

chmod +x /tmp/test_semantic_api.py
python /tmp/test_semantic_api.py
```

### Layer 6: Edge Cases

#### Empty/Malformed Inputs
```bash
# Empty file
touch /tmp/empty.json
polylogue import run /tmp/empty.json --provider chatgpt --out /tmp/test-edge 2>&1 | tee /tmp/edge-test-empty.log

# Malformed JSON
echo "not json" > /tmp/bad.json
polylogue import run /tmp/bad.json --provider chatgpt --out /tmp/test-edge 2>&1 | tee /tmp/edge-test-bad.log

# Missing required fields
echo '{"conversations": [{"id": "test"}]}' > /tmp/incomplete.json
polylogue import run /tmp/incomplete.json --provider chatgpt --out /tmp/test-edge 2>&1 | tee /tmp/edge-test-incomplete.log
```

#### Large Files
```bash
# Find largest export
LARGE_EXPORT=$(find /realm/data/exports/chatlog/raw -name "*.json" -o -name "*.jsonl" | xargs du -b | sort -rn | head -1 | cut -f2)
echo "Testing with large file: $LARGE_EXPORT ($(du -h "$LARGE_EXPORT"))"

# Time import
time polylogue import run "$LARGE_EXPORT" --out /tmp/test-large 2>&1 | tee /tmp/large-import.log
```

#### Concurrent Access
```bash
# Import in background
polylogue import run /tmp/test-inbox/chatgpt --out /tmp/test-concurrent-1 &
PID1=$!

# Search while importing
sleep 2
polylogue search "test" --json > /tmp/concurrent-search.json

# Another import
polylogue import run /tmp/test-inbox/claude --out /tmp/test-concurrent-2 &
PID2=$!

wait $PID1 $PID2
```

### Layer 7: Idempotency & Safety

#### Re-import same data
```bash
# Import once
polylogue import run /tmp/test-inbox/chatgpt/export1.json --out /tmp/test-idempotent

# Count conversations
BEFORE=$(sqlite3 ~/.local/share/polylogue/polylogue.db "SELECT COUNT(*) FROM conversations")

# Import again
polylogue import run /tmp/test-inbox/chatgpt/export1.json --out /tmp/test-idempotent --force

# Count again (should be same)
AFTER=$(sqlite3 ~/.local/share/polylogue/polylogue.db "SELECT COUNT(*) FROM conversations")

echo "Before: $BEFORE, After: $AFTER"
[ "$BEFORE" -eq "$AFTER" ] && echo "✓ Idempotent" || echo "✗ Created duplicates!"
```

#### Dirty file protection
```bash
# Import conversation
polylogue import run /tmp/test-inbox/chatgpt/export1.json --out /tmp/test-dirty

# Edit markdown
FIRST_MD=$(find /tmp/test-dirty -name "*.md" | head -1)
echo "\n\n<!-- USER EDIT -->" >> "$FIRST_MD"

# Try to re-import without --allow-dirty (should fail)
if polylogue import run /tmp/test-inbox/chatgpt/export1.json --out /tmp/test-dirty --force 2>&1 | grep -q "local edits"; then
  echo "✓ Correctly rejected dirty file"
else
  echo "✗ Failed to detect dirty file!"
fi

# Re-import with --allow-dirty (should succeed)
polylogue import run /tmp/test-inbox/chatgpt/export1.json --out /tmp/test-dirty --force --allow-dirty
```

## Pass/Fail Criteria

### Must Pass ✓
- [ ] All providers import without errors
- [ ] Database has correct schema version
- [ ] No orphaned messages or attachments in DB
- [ ] All conversations have non-null timestamps
- [ ] Filesystem layout is clean (no temp files, reasonable names)
- [ ] All referenced attachments exist on disk
- [ ] Markdown front matter is valid YAML
- [ ] Search returns results
- [ ] Semantic API can retrieve and filter conversations
- [ ] Idempotent re-imports don't create duplicates
- [ ] Dirty file protection works

### Should Pass ⚠️
- [ ] Filenames are human-readable (not just UUIDs)
- [ ] Attachments are organized sensibly
- [ ] HTML output renders correctly
- [ ] Large files import within reasonable time
- [ ] Concurrent access doesn't corrupt DB
- [ ] Error messages are helpful

### Investigation Required 🔍
- [ ] Are there any rough edges in the output?
- [ ] Is metadata as rich as it could be given input data?
- [ ] Are there missing features in semantic API?
- [ ] What analysis use cases are awkward?

## Next Steps After Testing
1. Document all failures
2. Fix critical issues
3. Polish rough edges
4. Enhance semantic API based on usage
5. Add regression tests for failures
