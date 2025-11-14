# Polylogue Idempotency Analysis Report

**Analysis Date:** 2025-11-14  
**Scope:** All major operations (render, sync, import, watch)  
**Focus:** Resumability, partial failure handling, safe re-execution

---

## Executive Summary

**Overall Idempotency Grade: B+ (Good with Critical Gaps)**

Polylogue demonstrates **strong idempotency design** through content hashing, dirty file detection, and robust state tracking. However, **one critical database operation lacks transaction protection**, creating a data loss risk during interruption.

### Key Findings

✅ **Strengths:**
- Content hash-based change detection prevents unnecessary rewrites
- Dirty file detection preserves user edits
- UPSERT patterns in database ensure safe updates
- Attachment reconciliation removes orphans

⚠️ **Critical Risk:**
- **Database message replacement** (150+ messages) lacks transaction protection
- DELETE + INSERT pattern vulnerable to interruption = permanent data loss

🔶 **Medium Issues:**
- `--force` flag bypasses dirty file protection (user edits silently overwritten)
- Attachment download failures leave orphaned files
- Prune operation lacks atomic deletion

---

## Part 1: Operation-by-Operation Analysis

### 1.1 SYNC Operation ✅ **HIGH IDEMPOTENCY**

**Location:** `local_sync.py`, `util.py:290-334`

**Idempotency Mechanism:**
```python
# util.py:290 - conversation_is_current()
def conversation_is_current(...) -> bool:
    # Multi-layer state detection:
    if dirty:                    # User edited file?
        return False
    if entry.get("dirty"):       # Marked dirty in state?
        return False  
    if updated_at != stored:     # Timestamp changed?
        return False
    if content_hash != stored:   # Content changed?
        return False
    if not output_path.exists(): # File missing?
        return False
    # ... additional checks for collapse, html, attachments
```

**Safety Features:**
1. **Two-layer detection:** mtime (fast) + SHA256 content hash (accurate)
2. **Dirty file protection:** Skips files with user edits
3. **Existence checks:** Verifies all expected outputs exist

**Re-execution Behavior:**
```bash
# First run
polylogue sync codex --out ~/archive
# → Processes 100 sessions, writes 100 markdown files

# Interrupted at session 50
# CTRL+C during processing

# Re-run (idempotent)
polylogue sync codex --out ~/archive
# → Skips sessions 1-49 (already current)
# → Resumes from session 50
# → No duplicates, no data loss
```

**Edge Case - Force Flag Issue:**
```bash
# User manually edits conversation.md (fixes typo)
vim ~/archive/codex/my-session-abc123/conversation.md

# User runs sync with --force (DANGEROUS!)
polylogue sync codex --force
# → Overwrites user's edit WITHOUT WARNING ❌
# → File becomes "clean" again, edit lost permanently
```

**Recommendation:** Split `--force` into:
- `--force-sync`: Skip timestamp/hash checks (re-render even if up-to-date)
- `--allow-dirty`: Explicitly allow overwriting edited files (requires both flags)

---

### 1.2 IMPORT Operation ✅ **MODERATE IDEMPOTENCY**

**Location:** `importers/*.py`, `conversation.py:130-354`

**Idempotency Mechanism:**
```python
# Delegates to ConversationRegistrar for state tracking
registrar.record_conversation(...)
# Uses conversation_is_current() for skip logic
```

**Pipeline Architecture:**
```
Source → Importer → Normalizer → Conversation Processor → Persist
         ↓
         State Check: Skip if current
```

**Re-execution Behavior:**
```bash
# Import ChatGPT export
polylogue import chatgpt export.zip --all

# Interrupted mid-import (conversation 50/100)
# CTRL+C

# Re-run (mostly idempotent)
polylogue import chatgpt export.zip --all
# → Skips conversations 1-49 (already imported)
# → Resumes from conversation 50
# → Safe to re-run multiple times
```

**Issue - Partial Attachments:**
```bash
# Importing conversation with 5 attachments
# Downloads: att1.png, att2.pdf, att3.jpg
# CTRL+C during att4.png download

# Re-run
polylogue import chatgpt export.zip --all
# → Skips conversation (markdown exists + hash matches)
# → att4.png and att5.png NEVER downloaded ❌
# → Markdown has broken links to missing attachments
```

**Recommendation:** Track attachment download status separately or fail-fast on attachment errors.

---

### 1.3 WATCH Operation ✅ **HIGH IDEMPOTENCY**

**Location:** `cli/watch.py:93-109`

**Mechanism:**
```python
# Debounced event handling
while True:
    changes = set(fs_watcher.watch(...))
    if changes:
        time.sleep(debounce_seconds)  # Default: 0.5s
        _sync_sessions(...)  # Delegates to sync (idempotent)
```

**Characteristics:**
- Event-driven, no polling loops
- Delegates to sync operation (inherits idempotency)
- Catches errors without stopping watcher

**Issue - Debounce Skipping:**
```bash
# User saves file twice rapidly
echo "change1" >> session.jsonl  # t=0.0s
echo "change2" >> session.jsonl  # t=0.3s (within 0.5s window)

# Watch behavior:
# → Detects first change
# → Sleeps 0.5s for debounce
# → Second change SILENTLY DROPPED ❌
# → Only processes first change state
```

**Recommendation:** Accumulate changes during debounce window instead of skipping.

---

### 1.4 RENDER Operation ⚠️ **MODERATE IDEMPOTENCY**

**Location:** `cli/render.py`

**Issue:** Render command bypasses state checks when given raw JSON files.

```bash
# Direct render (no state tracking)
polylogue render gemini-chat.json
# → Always re-renders (no skip logic)
# → Safe but inefficient

# Render with provider context (delegates to sync)
polylogue sync drive --chat-id abc123
# → Uses full idempotency checks
```

**Recommendation:** Add state checking to direct render path.

---

## Part 2: Critical Issues Deep Dive

### ✅ FIXED: Database Message Replacement Data Loss

**Location:** `db.py:216-273` - `replace_messages()` function

**Status:** Fixed in commit [current] with SAVEPOINT transaction protection

**The Problem (Historical):**
```python
def replace_messages(conn, *, provider, conversation_id, branch_id, messages):
    # STEP 1: Delete all messages (COMMITTED IMMEDIATELY)
    conn.execute(
        "DELETE FROM messages WHERE provider = ? AND conversation_id = ? AND branch_id = ?",
        (provider, conversation_id, branch_id),
    )
    conn.execute("DELETE FROM messages_fts WHERE ...", ...)
    
    # ⚠️ CRITICAL VULNERABILITY: If process crashes here, 
    # 150+ messages are DELETED but new ones never inserted!
    
    if not messages:
        return
    
    # STEP 2: Insert new messages (MAY NEVER HAPPEN if interrupted)
    conn.executemany("INSERT INTO messages ...", [...])
    conn.execute("INSERT INTO messages_fts ...", ...)
```

**Why This Happens:**
- SQLite has **autocommit mode** enabled by default
- `DELETE` is committed **immediately** after execution
- If process crashes before `INSERT`, data is **permanently lost**
- No transaction boundary protects the DELETE+INSERT sequence

**Proof of Concept:**
```python
# Simulating interruption
import sqlite3
conn = sqlite3.connect("polylogue.db")

# DELETE 150 messages
conn.execute("DELETE FROM messages WHERE conversation_id = 'abc123'")
# → Data deleted and COMMITTED

# Process crashes here (CTRL+C, OOM, power loss)
# exit(1)

# Messages are GONE FOREVER - cannot be recovered ❌
```

**Impact Severity: CRITICAL**
- **Probability:** Low (requires crash during specific window)
- **Impact:** Complete data loss (150+ messages)
- **Recovery:** Impossible (no backups, no undo)

**Fix - Add Transaction Protection:**
```python
def replace_messages(conn, *, provider, conversation_id, branch_id, messages):
    # SAVEPOINT protects against partial failure
    conn.execute("SAVEPOINT replace_msgs")
    
    try:
        conn.execute("DELETE FROM messages WHERE ...", ...)
        conn.execute("DELETE FROM messages_fts WHERE ...", ...)
        
        if messages:
            conn.executemany("INSERT INTO messages ...", [...])
            conn.execute("INSERT INTO messages_fts ...", ...)
        
        conn.execute("RELEASE replace_msgs")  # Commit on success
    except Exception:
        conn.execute("ROLLBACK TO replace_msgs")  # Rollback on error
        raise
```

**Effort:** 5 minutes  
**Priority:** IMMEDIATE - Prevents catastrophic data loss

---

### ✅ FIXED: Force Flag Destroys User Edits

**Location:** `document_store.py:312`, `util.py:305`

**Status:** Fixed with --allow-dirty flag implementation

**The Problem (Historical):**
```python
# Dirty file detection (GOOD)
if existing_dirty and not force:
    # Preserve user edits, skip overwrite
    return skip_with_dirty_flag()

# BUT: --force bypasses this! (BAD)
if force:
    # Overwrites edited file WITHOUT WARNING ❌
    write_markdown(path, new_content)
```

**User Story:**
```bash
# User manually fixes markdown formatting
vim ~/archive/codex/session-123/conversation.md
# Changes:  "##Header" → "## Header" (adds space)

# Later, user runs sync with --force
polylogue sync codex --force
# → User's formatting fix SILENTLY OVERWRITTEN
# → No warning, no backup created
# → Edit lost permanently
```

**Why --force Exists:**
- Re-render when metadata changed (collapse threshold, html theme)
- Force re-download of attachments
- Override timestamp-based skipping

**The Issue:**
`--force` has **two meanings**:
1. Skip freshness checks (intended)
2. Overwrite dirty files (dangerous side effect)

**Fix - Split Force Semantics:**
```python
# New behavior
if force and allow_dirty:
    # Both flags required to overwrite edited files
    write_markdown(path, new_content)
elif force:
    # Only force sync if file is clean
    if existing_dirty:
        raise Error("File has local edits. Use --allow-dirty to overwrite.")
    write_markdown(path, new_content)
```

**Effort:** 10 minutes  
**Priority:** HIGH - Prevents user data loss

---

### 🔶 MEDIUM ISSUE #3: Attachment Download Interruption

**Location:** `pipeline.py:93-131` - `collect_attachments()`

**The Problem:**
```python
def collect_attachments(...) -> List[AttachmentInfo]:
    results = []
    for att in attachments:
        try:
            local_path = download(att)  # May fail
            results.append(AttachmentInfo(path=local_path, ...))
        except Exception:
            pass  # Silently skip failed attachment
    return results
```

**Failure Scenario:**
```
Conversation has 5 attachments: [A, B, C, D, E]

Download sequence:
  A.png → SUCCESS (saved to disk)
  B.pdf → SUCCESS (saved to disk)
  C.jpg → FAILURE (network timeout) → Skipped silently
  D.png → SUCCESS (saved to disk)
  E.txt → SUCCESS (saved to disk)

Markdown references all 5 attachments:
  ![Image A](attachments/A.png) ✓ Works
  ![Image B](attachments/B.pdf) ✓ Works
  ![Image C](attachments/C.jpg) ❌ BROKEN LINK
  ![Image D](attachments/D.png) ✓ Works
  ![Image E](attachments/E.txt) ✓ Works
```

**Why This Matters:**
1. Conversation marked as "current" (markdown + state saved)
2. Re-running sync **skips** this conversation (already processed)
3. `C.jpg` **never** downloaded on subsequent runs
4. User has permanently broken link in markdown

**Fix Options:**

**Option A - Fail Fast (Strict):**
```python
def collect_attachments(...) -> List[AttachmentInfo]:
    results = []
    for att in attachments:
        local_path = download(att)  # Raises on failure
        results.append(AttachmentInfo(path=local_path, ...))
    return results  # All or nothing
```
- Pro: Guarantees completeness
- Con: One bad attachment fails entire conversation

**Option B - Track Missing (Flexible):**
```python
def collect_attachments(...) -> Tuple[List[AttachmentInfo], List[str]]:
    results = []
    failed = []
    for att in attachments:
        try:
            local_path = download(att)
            results.append(AttachmentInfo(path=local_path, ...))
        except Exception as e:
            failed.append(att.id)
            logger.warning(f"Failed to download {att.id}: {e}")
    return results, failed

# In caller:
attachments, failed = collect_attachments(...)
if failed:
    # Mark conversation as incomplete in state
    state["missingAttachments"] = failed
    state["complete"] = False
```
- Pro: Partial progress saved
- Con: Requires state field addition

**Effort:** 15 minutes  
**Priority:** MEDIUM - Causes broken links

---

### 🔶 MEDIUM ISSUE #4: Prune Partial Deletion

**Location:** `local_sync.py:246-258`

**The Problem:**
```python
def _prune_conversation(slug: str, output_dir: Path):
    conv_dir = output_dir / slug
    if conv_dir.exists():
        shutil.rmtree(conv_dir)  # NOT ATOMIC
        # If interrupted here, partial deletion!
```

**Failure Scenario:**
```
Conversation directory structure:
  my-session-abc/
    ├── conversation.md (100KB)
    ├── conversation.common.md (50KB)
    ├── attachments/
    │   ├── file1.png (2MB)
    │   ├── file2.pdf (5MB)
    │   ├── file3.jpg (3MB)  ← Process crashes here
    │   ├── file4.png (2MB)  ← Never deleted
    │   └── file5.txt (1KB)  ← Never deleted
    └── branches/
        └── ... (500 files)  ← Never deleted

Result:
  - 500 of 1000 files deleted
  - Directory still exists (not fully removed)
  - Zombie directory consumes disk space
  - May cause slug collision on next sync
```

**Fix - Atomic Move to Trash:**
```python
import tempfile
import time

def _prune_conversation(slug: str, output_dir: Path):
    conv_dir = output_dir / slug
    if not conv_dir.exists():
        return
    
    # Atomic move (single filesystem operation)
    trash_dir = output_dir / ".trash"
    trash_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    trash_target = trash_dir / f"{slug}.{timestamp}"
    
    # Move is atomic on same filesystem
    conv_dir.rename(trash_target)
    
    # Later: Background cleanup of trash
    # (Can be interrupted safely - files already moved)
    try:
        shutil.rmtree(trash_target)
    except OSError:
        pass  # Will be cleaned up next time
```

**Effort:** 15 minutes  
**Priority:** MEDIUM - Wastes disk space

---

## Part 3: Idempotency Strengths

### ✅ Content Hash Change Detection

**Location:** `document_store.py:304`, `util.py:311`

```python
# Compute hash of CONTENT (excludes mutable metadata)
content_hash = hashlib.sha256(
    document.body.encode() + 
    json.dumps(metadata, sort_keys=True).encode()
).hexdigest()[:16]

# State stores this hash
state["contentHash"] = content_hash

# On next run: Skip if hash unchanged
if stored_hash == current_hash:
    skip_writing()  # Idempotent!
```

**Why This Works:**
- Hash captures **semantic content**, not filesystem metadata
- Changing mtime doesn't trigger re-render (mtime unreliable)
- Only **real content changes** cause rewrites
- False positives impossible (SHA256 collision negligible)

---

### ✅ Dirty File Detection

**Location:** `document_store.py:308-341`

```python
# Read existing file from disk
existing = read_markdown(path)
stored_hash = state.get("localHash")

# Detect user edits
if existing.content_hash != stored_hash:
    existing_dirty = True  # User modified file!

# Preserve edits (unless --force)
if existing_dirty and not force:
    state["dirty"] = True
    return skip_result()  # Don't overwrite!
```

**User Protection:**
```bash
# User adds personal notes to markdown
vim ~/archive/codex/session-123/conversation.md
# Adds: "<!-- TODO: Review this code -->"

# Run sync (won't overwrite!)
polylogue sync codex
# → Detects file hash mismatch
# → Marks conversation as "dirty"
# → SKIPS overwriting (preserves notes) ✓
```

---

### ✅ UPSERT Database Pattern

**Location:** `db.py:122-136`

```python
# INSERT OR REPLACE = Idempotent update
conn.execute(
    """
    INSERT OR REPLACE INTO conversations (
        provider, conversation_id, slug, title, markdown_path, ...
    ) VALUES (?, ?, ?, ?, ?, ...)
    """,
    (provider, conversation_id, slug, title, md_path, ...)
)
```

**Why This Works:**
- First run: `INSERT` creates new row
- Second run: `REPLACE` updates existing row
- No duplicates, no errors
- Idempotent by design

---

### ✅ Attachment Reconciliation

**Location:** `services/attachments.py:28-131`

```python
def reconcile(self, expected: List[AttachmentInfo]) -> ReconcileResult:
    expected_names = {att.filename for att in expected}
    existing_files = set(self.dir.iterdir())
    
    # Remove orphaned attachments
    for path in existing_files:
        if path.name not in expected_names:
            path.unlink()  # Clean up old files
    
    # Download missing attachments
    for att in expected:
        if not (self.dir / att.filename).exists():
            download(att)  # Fetch missing
    
    return ReconcileResult(...)
```

**Idempotent Behavior:**
- Removes orphans from previous runs
- Downloads only missing files
- Safe to re-run multiple times
- Converges to correct state

---

## Part 4: Recommendations Summary

### Immediate (Critical) - Do These First

**1. Fix Database Transaction (5 min) - CRITICAL**
```python
# db.py:216
def replace_messages(conn, ...):
    conn.execute("SAVEPOINT replace_msgs")
    try:
        # DELETE + INSERT operations
        conn.execute("RELEASE replace_msgs")
    except:
        conn.execute("ROLLBACK TO replace_msgs")
        raise
```

**2. Split --force Flag (10 min) - HIGH**
```python
# Add --allow-dirty flag
parser.add_argument("--allow-dirty", action="store_true",
    help="Allow overwriting files with local edits")

# Require both flags to overwrite edits
if dirty and not (force and allow_dirty):
    return skip_with_warning()
```

**3. Log Attachment Failures (10 min) - MEDIUM**
```python
# pipeline.py:93
except Exception as e:
    logger.error(f"Failed to download {att.id}: {e}")
    # Don't silently skip
```

### Short-term (Important) - Next Sprint

**4. Atomic Prune (15 min)**
- Move to trash before deletion
- Background cleanup

**5. Debounce Accumulation (20 min)**
- Track all changes in window
- Process batch after debounce

**6. Add Integration Tests (2-3 hours)**
```python
def test_sync_interrupted_resumable():
    # Sync 100 sessions, interrupt at 50
    # Verify: Re-run processes 50-100 only
    
def test_force_respects_dirty_files():
    # Edit file, run --force
    # Verify: Raises error or preserves edit
    
def test_attachment_failure_tracking():
    # Fail download on att 3/5
    # Verify: Conversation marked incomplete
```

---

## Part 5: Idempotency Scorecard

| Operation | Grade | Resumable | State Tracking | Data Loss Risk |
|-----------|-------|-----------|----------------|----------------|
| **Sync** | A | ✅ Yes | ✅ Excellent | ⚠️ Low (force flag) |
| **Import** | B+ | ✅ Yes | ✅ Good | ⚠️ Low (attachments) |
| **Watch** | A | ✅ Yes | ✅ Excellent | ✅ None |
| **Render** | B | ⚠️ Partial | ⚠️ Bypassed | ✅ None |
| **Database** | C | ❌ No | ✅ Good | 🔴 **HIGH** (replace_messages) |
| **Prune** | B- | ⚠️ Partial | N/A | ⚠️ Low (disk space) |

**Overall Grade: B+**
- Strong design with content hashing and dirty detection
- One critical gap in database transactions
- Several medium issues with attachment/prune operations

---

## Part 6: Testing Recommendations

### Idempotency Test Suite

```python
class TestIdempotency:
    def test_sync_is_idempotent(self):
        """Running sync twice produces identical results."""
        result1 = sync_codex(sessions=[s1, s2, s3])
        result2 = sync_codex(sessions=[s1, s2, s3])
        assert result1 == result2
        assert result2.processed == 0  # All skipped (current)
    
    def test_interrupted_sync_resumable(self):
        """Interrupted sync can be resumed without duplicates."""
        # Start sync of 100 sessions
        with interrupt_at(session=50):
            sync_codex(all=True)
        
        # Resume sync
        result = sync_codex(all=True)
        assert result.processed == 50  # Remaining sessions
        assert result.skipped == 49     # Already done
    
    def test_dirty_files_preserved(self):
        """User edits are not overwritten by sync."""
        # Edit file manually
        edit_markdown(path, add_note="TODO: Review")
        
        # Run sync
        result = sync_codex()
        
        # Verify edit preserved
        content = read_markdown(path)
        assert "TODO: Review" in content
        assert result.dirty_count == 1
    
    def test_force_without_allow_dirty_fails(self):
        """--force alone does not overwrite edited files."""
        edit_markdown(path, add_note="User note")
        
        with pytest.raises(Error, match="local edits"):
            sync_codex(force=True)  # Missing allow_dirty
    
    def test_attachment_failure_tracked(self):
        """Failed attachment downloads are tracked."""
        # Mock attachment download to fail on att3
        mock_download(fail_on="att3")
        
        result = import_conversation(attachments=[att1, att2, att3])
        
        assert result.missing_attachments == ["att3"]
        assert result.complete == False
```

### Chaos Engineering Tests

```python
class TestPartialFailure:
    def test_database_crash_during_replace(self):
        """Database crash during replace_messages doesn't lose data."""
        # This test SHOULD PASS after fixing replace_messages()
        with crash_after_sql("DELETE FROM messages"):
            try:
                sync_codex()
            except CrashError:
                pass
        
        # Verify: Messages not lost
        conn = open_db()
        count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert count == original_count  # No data loss
    
    def test_disk_full_during_write(self):
        """Disk full during markdown write is recoverable."""
        with disk_full_after(bytes=1000):
            result = sync_codex()
        
        # Verify: Partial write detected
        assert result.failed > 0
        
        # Verify: Re-run completes
        result2 = sync_codex()
        assert result2.processed > 0  # Resumes
```

---

## Conclusion

Polylogue has **strong idempotency foundations** but requires **one critical fix** to eliminate data loss risk. The content hashing and dirty file detection are excellent design patterns that make operations safely resumable.

**Priority Actions:**
1. ✅ **Fix `replace_messages()` transaction** (5 min, prevents data loss)
2. ✅ **Split `--force` flag** (10 min, protects user edits)
3. ✅ **Add idempotency tests** (2 hours, prevents regressions)

After these fixes, Polylogue will achieve **A-grade idempotency** across all operations.
