# Polylogue Security, Privacy & Reliability Roadmap

## Overview

Polylogue stores potentially sensitive AI conversation data (personal assistant interactions, code discussions, thinking traces, proprietary information). Current architecture prioritizes data integrity but lacks privacy protections expected for a personal data store. This roadmap addresses encryption at rest, secure import pipeline, audit trails, and disaster recovery.

**Threat Model**:
| Threat | Risk | Current Status | Gap |
|--------|------|---|---|
| Local disk access (theft, forensics) | HIGH | None | No encryption at rest |
| Memory disclosure (core dumps, swap) | MEDIUM | None | Sensitive data unencrypted |
| Malicious import files | MEDIUM | ZIP bomb detection | Limited content validation |
| Data corruption | LOW | WAL, integrity checks | No recovery mechanism |
| Accidental exposure (backup sync) | MEDIUM | None | Unencrypted exports |
| Credential leakage | MEDIUM | Separate token files | OAuth tokens plaintext |
| Audit gaps | LOW | runs table | Limited modification tracking |

---

## Priority 1: Data at Rest Protection

### 1. Database Encryption via SQLCipher

**Files Affected**: `storage/db.py`, `storage/backends/sqlite.py`

**Current State**: SQLite database stored in plaintext at `~/.local/state/polylogue/polylogue.db`

**Solution**: Replace sqlite3 with sqlcipher (AES-256-GCM):

```python
# storage/db.py - Connection creation
import sqlcipher3 as sqlite3

def open_connection_encrypted() -> sqlite3.Connection:
    """Open encrypted SQLite connection."""
    conn = sqlite3.connect(str(db_path))

    # Derive key from passphrase using Argon2id
    from argon2 import PasswordHasher
    from base64 import b64decode

    # Load or create key
    key_path = config_dir / "polylogue.key"
    if not key_path.exists():
        # Generate random key on first run
        key = os.urandom(32)
        key_path.write_bytes(key)
    else:
        key = key_path.read_bytes()

    # Set encryption key (PRAGMA key = 'hex:...')
    hex_key = key.hex()
    conn.execute(f"PRAGMA key = 'hex:{hex_key}'")
    conn.execute("PRAGMA cipher_page_size = 4096")
    conn.execute("PRAGMA cipher_hmac_algorithm = HMAC_SHA512")

    return conn
```

**Configuration**:
```bash
# Option 1: Environment variable
export POLYLOGUE_DB_KEY="hex:a1b2c3d4..."

# Option 2: Interactive passphrase (one-time setup)
polylogue setup --encrypt
# Prompt: Enter passphrase to encrypt database
# Stores derived key in ~/.config/polylogue/db.key (chmod 600)

# Option 3: Keyring integration (Linux/macOS)
polylogue setup --encrypt --keyring
# Uses system keyring (libsecret, Keychain)
```

**Migration for Existing Archives**:
```bash
polylogue migrate --encrypt
# Backs up unencrypted DB to .backup
# Re-encrypts in place
# Deletes unencrypted backup after verification
```

**Performance Impact**: +5-10% latency per query (acceptable for local-first app)

**Success Criteria**:
- Database file unreadable without key
- No performance degradation >15%
- Key stored securely (not hardcoded, file permissions 600)

---

### 2. Attachment Encryption

**Files Affected**: `storage/store.py`, `storage/backends/sqlite.py`

**Current State**: Attachment files stored at plaintext paths, referenced by `attachments.path`

**Solution**: AES-256-GCM encryption for attachment files:

```python
# storage/encryption.py - New module
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

class AttachmentEncryption:
    def __init__(self, master_key: bytes):
        self.master_key = master_key

    def encrypt_file(self, source_path: Path, dest_path: Path) -> tuple[str, str]:
        """Encrypt attachment file, return key handle and nonce."""
        # Generate per-file nonce
        nonce = os.urandom(12)

        # Derive file-specific key from master key + nonce
        kdf = PBKDF2(hashes.SHA256(), nonce, 100000)
        file_key = kdf.derive(self.master_key)

        # Encrypt
        cipher = AESGCM(file_key)
        with open(source_path, 'rb') as f:
            plaintext = f.read()

        ciphertext = cipher.encrypt(nonce, plaintext, None)

        # Write encrypted file
        with open(dest_path, 'wb') as f:
            f.write(nonce)  # Prepend nonce
            f.write(ciphertext)

        return dest_path.name, nonce.hex()

    def decrypt_file(self, encrypted_path: Path) -> bytes:
        """Decrypt attachment file."""
        with open(encrypted_path, 'rb') as f:
            nonce = f.read(12)
            ciphertext = f.read()

        # Derive key using stored nonce
        kdf = PBKDF2(hashes.SHA256(), bytes.fromhex(nonce.hex()), 100000)
        file_key = kdf.derive(self.master_key)

        # Decrypt
        cipher = AESGCM(file_key)
        return cipher.decrypt(nonce, ciphertext, None)
```

**Storage Schema**:
```sql
ALTER TABLE attachments ADD COLUMN is_encrypted BOOLEAN DEFAULT 0;
ALTER TABLE attachments ADD COLUMN encryption_nonce TEXT;  -- Hex-encoded nonce

-- Store encrypted key in DB
CREATE TABLE attachment_keys (
    attachment_id TEXT PRIMARY KEY,
    key_handle TEXT,  -- Path reference
    nonce TEXT,  -- Hex-encoded
    FOREIGN KEY (attachment_id) REFERENCES attachments(attachment_id)
);
```

**Integration**:
- Transparent on read (decrypt on-demand)
- Encrypt on import
- Store nonce for decryption
- Supports multiple key versions (key rotation)

**Success Criteria**:
- Encrypt all attachments during import
- Decrypt transparently for rendering
- No latency impact >5%

---

## Priority 2: Secure Import Pipeline

### 3. Enhanced Input Validation

**Files Affected**: `ingestion/source.py`, new `ingestion/validation.py`

**Current Gaps**:
- No payload size limits on parsed JSON objects
- Limited validation of message counts
- No content sanitization
- No detection of suspicious patterns

**Solution**: Strict validation pipeline:

```python
# ingestion/validation.py
class ImportValidator:
    MAX_PAYLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_MESSAGES_PER_CONVERSATION = 100_000
    MAX_MESSAGE_LENGTH = 1_000_000  # 1MB per message
    MAX_ATTACHMENT_SIZE = 50 * 1024 * 1024  # 50MB per attachment

    def validate_payload(self, payload: Any, source_path: Path) -> ValidationResult:
        """Comprehensive validation before ingestion."""
        errors = []
        warnings = []

        # Size check
        payload_size = self._estimate_size(payload)
        if payload_size > self.MAX_PAYLOAD_SIZE:
            errors.append(f"Payload too large: {payload_size / 1e6:.1f}MB > {self.MAX_PAYLOAD_SIZE / 1e6:.1f}MB")

        # Message count validation
        if isinstance(payload, list):
            if len(payload) > self.MAX_MESSAGES_PER_CONVERSATION:
                errors.append(f"Too many messages: {len(payload)} > {self.MAX_MESSAGES_PER_CONVERSATION}")

        # Content sanitization
        for item in self._iter_messages(payload):
            if len(item.get("content", "")) > self.MAX_MESSAGE_LENGTH:
                warnings.append(f"Message exceeds {self.MAX_MESSAGE_LENGTH} bytes, will truncate")

            # Strip control characters
            item["content"] = self._sanitize_text(item.get("content", ""))

        # Attachment validation
        for attachment in self._iter_attachments(payload):
            if attachment.get("size", 0) > self.MAX_ATTACHMENT_SIZE:
                errors.append(f"Attachment too large: {attachment['size'] / 1e6:.1f}MB")

        return ValidationResult(valid=len(errors) == 0, errors=errors, warnings=warnings)

    def _sanitize_text(self, text: str) -> str:
        """Remove control characters and suspicious patterns."""
        # Remove null bytes
        text = text.replace('\x00', '')

        # Remove other control characters except tab, newline, carriage return
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\t\n\r')

        # Truncate if needed
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[:self.MAX_MESSAGE_LENGTH]

        return text
```

**Integration**:
```python
# In ingestion/source.py
def ingest_from_file(path: Path) -> ParsedConversation | None:
    payload = load_json(path)

    validator = ImportValidator()
    result = validator.validate_payload(payload, path)

    if not result.valid:
        for error in result.errors:
            logger.error(f"Validation failed: {error}")
        raise ValidationError(result.errors)

    for warning in result.warnings:
        logger.warning(f"Import warning: {warning}")

    return detect_provider(payload, path).parse(payload)
```

**Success Criteria**:
- Reject payloads >100MB
- Reject conversations >100K messages
- Sanitize all text content
- Log all validation events

---

### 4. PII Detection and Redaction

**Files Affected**: New `privacy/pii.py`, `privacy/redaction.py`, `cli/commands/export.py`

**Problem**: No protection against accidental exposure of sensitive data in exports or shares.

**Solution**: Optional PII detection layer:

```python
# privacy/pii.py
import re

class PIIDetector:
    PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
        "ssn": r'\b(?!000|111|222|333|444|555|666|777|888|999)(?!12345|98765)(?!.*[a-zA-Z])[0-9]{3}-?[0-9]{2}-?[0-9]{4}\b',
        "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        "api_key": r'(?:api[_-]?key|apikey|api_secret|secret_key|access_token)[\s]*[=:]\s*[\'"]?[a-zA-Z0-9_\-]{20,}[\'"]?',
        "private_key": r'-----BEGIN (RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY-----',
        "password_pattern": r'(?:password|passwd|pwd)[\s]*[=:]\s*[\'"]?([^\s\'"\n]+)[\'"]?',
    }

    def detect_pii(self, text: str) -> list[tuple[str, str, int]]:
        """Detect PII patterns in text. Returns (type, match, position)."""
        matches = []

        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((pii_type, match.group(0), match.start()))

        return sorted(matches, key=lambda x: x[2])

    def redact(self, text: str, mode: Literal["warn", "redact", "skip"] = "warn") -> tuple[str, list[str]]:
        """Redact PII according to mode."""
        pii_items = self.detect_pii(text)

        if mode == "warn":
            return text, [f"Found {len(pii_items)} PII items ({', '.join(set(p[0] for p in pii_items))})"

        elif mode == "redact":
            redacted = text
            for pii_type, match, pos in reversed(pii_items):  # Reverse to maintain positions
                redacted = redacted.replace(match, f"[REDACTED_{pii_type.upper()}]")
            return redacted, [f"Redacted {len(pii_items)} PII items"]

        elif mode == "skip":
            return "", ["Skipped due to PII detection"]

class PIIRedactionLog:
    """Audit trail for redactions."""
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def log_redaction(self, conversation_id: str, message_count: int, pii_count: int, mode: str) -> None:
        """Log redaction event."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            INSERT INTO pii_redaction_log
            (conversation_id, message_count, pii_count, mode, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, message_count, pii_count, mode, datetime.now().isoformat()))
        conn.commit()
```

**Integration with Export**:
```bash
polylogue export --out archive.jsonl --redact warn
# Output: Found 3 credit card patterns, 5 API keys
# Proceed? (y/n)

polylogue export --out archive.jsonl --redact redact
# Exports with [REDACTED_CREDIT_CARD] etc.

polylogue export --out archive.jsonl --redact strict
# Skips any messages/conversations with PII
```

**Success Criteria**:
- Detect common PII patterns (emails, phones, SSNs, API keys, private keys)
- >95% recall on test dataset
- Optional enforcement (warn/redact/skip)

---

## Priority 3: Data Integrity & Recovery

### 5. Content Hash Verification

**Files Affected**: Extend `verify.py` with content integrity checks

**Solution**: Verify stored content hashes match computed hashes:

```python
# verify.py extension
def verify_content_hashes(sample_rate: float = 0.1) -> VerifyResult:
    """Verify stored content_hash matches recomputed hash."""
    repository = StorageRepository()
    result = VerifyResult()

    # Sample conversations
    convs = list(repository.iter_conversations())
    sample = random.sample(convs, max(1, int(len(convs) * sample_rate)))

    for conv in sample:
        for msg in conv.messages:
            # Recompute hash
            computed = compute_content_hash(msg)
            stored = msg.content_hash

            if computed != stored:
                result.errors.append(
                    f"Hash mismatch in message {msg.message_id}: "
                    f"stored={stored}, computed={computed}"
                )
                result.corrupted_messages.append(msg.message_id)

    result.verified_messages = len(sample) * 50  # Avg 50 messages per conv
    return result

class VerifyResult:
    verified_messages: int = 0
    corrupted_messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
```

**CLI**:
```bash
polylogue verify --integrity
# Verifies: orphaned attachments, corrupted content hashes, schema consistency

polylogue verify --sample 0.5  # Check 50% of messages
polylogue verify --full  # Check 100% (slow but thorough)
```

**Success Criteria**:
- Detect content hash mismatches reliably
- Complete full verification in <5 minutes for 10K messages

---

### 6. Automated Backup System

**Files Affected**: New `backup/`, `cli/commands/backup.py`

**Solution**: Built-in encrypted backup and recovery:

```python
# backup/manager.py
class BackupManager:
    def __init__(self, db_path: Path, backup_dir: Path):
        self.db_path = db_path
        self.backup_dir = backup_dir

    def create_backup(self,
                     incremental: bool = False,
                     encrypt: bool = True,
                     compress: bool = True) -> Path:
        """Create backup of archive."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"polylogue_{'incremental' if incremental else 'full'}_{timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.zip"

        with zipfile.ZipFile(backup_path, 'w') as zf:
            if incremental:
                # Only export conversations since last backup
                last_backup = self._get_last_backup_time()
                convs = list(self.repository.iter_conversations(after=last_backup))
            else:
                # Full backup
                convs = list(self.repository.iter_conversations())

            # Create manifest
            manifest = {
                "type": "incremental" if incremental else "full",
                "created_at": datetime.now().isoformat(),
                "conversation_count": len(convs),
                "message_count": sum(len(c.messages) for c in convs),
            }
            zf.writestr("MANIFEST.json", json.dumps(manifest, indent=2))

            # Export conversations
            for conv in convs:
                conv_json = self.repository.export_conversation(conv.conversation_id)
                zf.writestr(f"conversations/{conv.conversation_id}.json", conv_json)

        if encrypt:
            backup_path = self._encrypt_backup(backup_path)

        logger.info(f"Backup created: {backup_path}")
        return backup_path

    def restore_backup(self, backup_path: Path) -> None:
        """Restore from backup."""
        if backup_path.suffix == ".gpg":
            backup_path = self._decrypt_backup(backup_path)

        with zipfile.ZipFile(backup_path, 'r') as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))

            # Import conversations
            for filename in zf.namelist():
                if filename.startswith("conversations/"):
                    conv_json = zf.read(filename)
                    self.repository.import_conversation(json.loads(conv_json))

        logger.info(f"Restored {manifest['conversation_count']} conversations")
```

**CLI**:
```bash
# Full encrypted backup
polylogue backup --full --encrypt --output ~/backups/

# Incremental backup (since last)
polylogue backup --incremental

# List backups
polylogue backup --list

# Restore from backup
polylogue restore ~/backups/polylogue_full_20240615_120000.zip.gpg
# Prompt: Confirm restore will merge/overwrite?
```

**Success Criteria**:
- Full backup: <60 seconds for 1000 conversations
- Incremental backup: <5 seconds (only new conversations)
- Encrypted backups unreadable without key

---

### 7. Crash Recovery Improvements

**Files Affected**: `storage/db.py` migrations, `storage/repository.py`

**Solution**: Safe migration and recovery:

```python
# storage/db.py
def _run_migrations(conn: sqlite3.Connection) -> None:
    """Run pending migrations with safety checks."""
    current_version = _get_schema_version(conn)
    pending = [m for m in MIGRATIONS if m.version > current_version]

    if not pending:
        return

    # Pre-migration backup
    backup_path = db_path.with_suffix('.pre-migration.db')
    logger.info(f"Creating pre-migration backup: {backup_path}")
    import shutil
    shutil.copy(db_path, backup_path)

    # Migration checkpoint
    checkpoint_path = db_path.with_suffix('.migration.checkpoint')

    try:
        for migration in pending:
            logger.info(f"Running migration {migration.version}: {migration.name}")

            # Store checkpoint
            checkpoint_path.write_text(str(migration.version))

            # Execute migration
            migration.up(conn)

            # Verify
            conn.execute("PRAGMA integrity_check")
            conn.commit()

        # Success - clean up checkpoint and backup
        checkpoint_path.unlink(missing_ok=True)
        # Keep backup for 7 days

    except Exception as e:
        logger.error(f"Migration failed: {e}")

        # Rollback to checkpoint
        if checkpoint_path.exists():
            last_version = int(checkpoint_path.read_text())
            logger.info(f"Rolling back to version {last_version}")
            # Load migration code and execute down() for pending

        raise MigrationError(f"Failed to migrate: {e}")
```

**Success Criteria**:
- Safe rollback on migration failure
- 100% uptime during recovery
- Pre-migration backups retained

---

## Priority 4: Audit & Provenance

### 8. Import Audit Trail

**Files Affected**: New `audit/importer.py`, storage schema

**Solution**: Comprehensive import provenance:

```sql
CREATE TABLE import_audit (
    audit_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    source_file_hash TEXT,  -- SHA-256 of source file
    source_file_path TEXT,  -- Original path
    import_timestamp TEXT,
    import_method TEXT,  -- "file", "drive", "api", etc.
    md5_to_hash_mapping JSON,  -- Historical hash mapping
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
);
```

**Captured Metadata**:
- Original export date (from file timestamp or metadata)
- Import date/time
- Source provider
- Source file hash (for deduplication across backups)
- Import method
- User/host that imported

**Success Criteria**:
- Audit trail complete for all imports
- Identify which conversations came from which backups/exports

---

## Priority 5: Credential Security

### 9. OAuth Token Protection

**Files Affected**: `ingestion/drive_client.py`, new `auth/token_store.py`

**Current State**: OAuth tokens stored in plaintext at `~/.config/polylogue/polylogue-token.json`

**Solution**: Encrypt tokens using system keyring or encrypted file:

```python
# auth/token_store.py
class SecureTokenStore:
    def __init__(self, use_keyring: bool = True):
        self.use_keyring = use_keyring
        self.token_path = Path.home() / ".config/polylogue/tokens.json"

    def store_token(self, provider: str, token: dict) -> None:
        """Store token securely."""
        if self.use_keyring:
            try:
                import keyring
                keyring.set_password(
                    "polylogue",
                    f"{provider}.access_token",
                    token["access_token"]
                )
                if "refresh_token" in token:
                    keyring.set_password(
                        "polylogue",
                        f"{provider}.refresh_token",
                        token["refresh_token"]
                    )
            except ImportError:
                logger.warning("keyring not available, falling back to encrypted file")
                self._store_encrypted(provider, token)
        else:
            self._store_encrypted(provider, token)

    def _store_encrypted(self, provider: str, token: dict) -> None:
        """Encrypt and store token in file."""
        from cryptography.fernet import Fernet

        # Derive key from master password
        key = self._get_or_create_key()
        cipher = Fernet(key)

        # Load existing tokens
        tokens = self._load_encrypted_tokens()
        tokens[provider] = token

        # Encrypt and write
        encrypted = cipher.encrypt(json.dumps(tokens).encode())
        self.token_path.write_bytes(encrypted)
        self.token_path.chmod(0o600)
```

**Success Criteria**:
- OAuth tokens never stored in plaintext
- Keyring integration when available
- Fallback to encrypted file storage

---

## Implementation Timeline

| Phase | Priority | Duration | Features |
|-------|----------|----------|----------|
| Phase 1 | Critical | Weeks 1-2 | DB encryption, input validation |
| Phase 2 | High | Weeks 3-4 | Attachment encryption, PII detection |
| Phase 3 | Medium | Weeks 5-6 | Backup system, audit trails |
| Phase 4 | Medium | Weeks 7-8 | Recovery improvements, token protection |

---

## Success Criteria (End State)

- **Encryption**: All data encrypted at rest (AES-256-GCM)
- **Input Validation**: All imports validated, PII detection available
- **Recovery**: Automated backups, safe migrations, crash recovery
- **Audit**: Complete import audit trail, tampering detection
- **Credentials**: No plaintext tokens, keyring integration
- **Compliance**: GDPR-ready (data export, PII controls, audit logs)

**Target**: Enterprise-grade security for personal data store
