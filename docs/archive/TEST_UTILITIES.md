# Test Utilities Guide

**Created**: During aggressive test consolidation (436→468 parametrized cases across 8 files)

**Purpose**: Reduce boilerplate in parametrized tests, improve readability, standardize test data creation

---

## Quick Start

```python
from tests.helpers import (
    ConversationBuilder,
    MessageBuilder,
    make_message,
    assert_messages_ordered,
    assert_contains_all,
)

def test_my_feature(db_path):
    # Build conversation with fluent API
    conv = (ConversationBuilder(db_path, "test-conv")
           .title("My Test")
           .add_message("m1", role="user", text="Hello")
           .add_message("m2", role="assistant", text="Hi there")
           .save())

    # Assert message ordering
    result = render_conversation(conv)
    assert_messages_ordered(result, "Hello", "Hi there")
```

---

## Builders (Fluent API)

### ConversationBuilder

**Best for**: Creating conversations with messages/attachments for database tests

```python
from tests.helpers import ConversationBuilder

# Simple conversation
conv = (ConversationBuilder(db_path, "conv1")
       .title("Test Chat")
       .provider("chatgpt")
       .add_message("m1", role="user", text="Question")
       .add_message("m2", role="assistant", text="Answer")
       .save())

# With attachments
conv = (ConversationBuilder(db_path, "conv2")
       .add_message("m1", text="See attachment")
       .add_attachment("att1", mime_type="image/png", provider_meta={"name": "screenshot.png"})
       .save())

# Custom timestamps
conv = (ConversationBuilder(db_path, "conv3")
       .created_at("2024-01-01T10:00:00Z")
       .updated_at("2024-01-15T15:00:00Z")
       .add_message("m1", text="Old message", timestamp="2024-01-01T10:00:00Z")
       .save())
```

**Methods**:
- `.title(title)` - Set conversation title
- `.provider(name)` - Set provider name
- `.created_at(iso_timestamp)` - Set created timestamp
- `.updated_at(iso_timestamp)` - Set updated timestamp
- `.add_message(message_id, role, text, timestamp, **kwargs)` - Add message
- `.add_attachment(attachment_id, message_id, mime_type, size_bytes, path, provider_meta)` - Add attachment
- `.save()` - Write to database and return ConversationRecord

### MessageBuilder

**Best for**: Building complex messages with metadata

```python
from tests.helpers import MessageBuilder

msg = (MessageBuilder("m1", "conv1")
      .role("assistant")
      .text("Response with metadata")
      .timestamp("2024-01-01T10:00:00Z")
      .meta({"thinking": "Let me analyze...", "cost": 0.005})
      .build())
```

**Methods**:
- `.role(role)` - Set message role
- `.text(text)` - Set message text
- `.timestamp(iso_timestamp | None)` - Set timestamp
- `.meta(dict | None)` - Set provider_meta
- `.build()` - Return MessageRecord

---

## Quick Builders (Simple Cases)

### make_message()

**Best for**: Single message creation without builder

```python
from tests.helpers import make_message

msg = make_message("m1", conversation_id="conv1", role="user", text="Hello")
```

**Signature**:
```python
make_message(
    message_id: str = "m1",
    conversation_id: str = "conv1",
    role: str = "user",
    text: str = "Test message",
    timestamp: str | None = None,
    **kwargs
) -> MessageRecord
```

### make_attachment()

**Best for**: Single attachment creation

```python
from tests.helpers import make_attachment

att = make_attachment("att1", conversation_id="conv1", name="file.pdf")
# Automatically sets provider_meta={"name": "file.pdf"}
```

**Signature**:
```python
make_attachment(
    attachment_id: str = "att1",
    conversation_id: str = "conv1",
    message_id: str | None = None,
    mime_type: str = "application/octet-stream",
    size_bytes: int = 1024,
    name: str | None = None,
    **kwargs
) -> AttachmentRecord
```

---

## Assertion Helpers

### assert_messages_ordered()

**Best for**: Verifying message order in rendered output

```python
from tests.helpers import assert_messages_ordered

result = formatter.format("conv1")
assert_messages_ordered(result.markdown_text, "First", "Second", "Third")
# Raises AssertionError if order is wrong or text not found
```

### assert_contains_all()

**Best for**: Verifying multiple substrings present

```python
from tests.helpers import assert_contains_all

assert_contains_all(markdown, "## user", "## assistant", "Hello", "Response")
```

### assert_not_contains_any()

**Best for**: Verifying substrings absent

```python
from tests.helpers import assert_not_contains_any

assert_not_contains_any(markdown, "ERROR", "```json", "FAIL")
```

---

## Test Data Generators (Importer Tests)

### make_chatgpt_node()

**Best for**: Creating ChatGPT mapping nodes for importer tests

```python
from tests.helpers import make_chatgpt_node

node = make_chatgpt_node(
    msg_id="msg1",
    role="user",
    content_parts=["Hello", "World"],
    children=["msg2", "msg3"],
    timestamp=1704067200,
    metadata={"attachments": [{"file_name": "doc.pdf"}]}
)

# Use in tests
mapping = {"msg1": node}
result = extract_messages_from_mapping(mapping, "conv1")
```

### make_claude_chat_message()

**Best for**: Creating Claude AI chat_messages entries

```python
from tests.helpers import make_claude_chat_message

msg = make_claude_chat_message(
    uuid="u1",
    sender="human",
    text="Question",
    files=[{"file_name": "attachment.pdf"}],
    timestamp="2024-01-01T10:00:00Z"
)

# Use in tests
chat_messages = [msg]
result = extract_messages_from_chat_messages(chat_messages, "conv1")
```

### make_claude_code_message()

**Best for**: Creating Claude Code message entries

```python
from tests.helpers import make_claude_code_message

msg = make_claude_code_message(
    msg_type="tool_use",
    text='{"name": "read", "input": {"path": "file.txt"}}',
    thinking="Let me check the file..."
)

# Use in tests
messages = [msg]
result = parse_code({"session_id": "s1", "messages": messages}, "conv1")
```

---

## Coverage Verification

### parametrized_case_count()

**Best for**: Documenting parametrized test coverage

```python
from tests.helpers import parametrized_case_count

FILTER_CASES = [
    ("user_messages", 2, "user filter"),
    ("assistant_messages", 3, "assistant filter"),
    ("dialogue_only", 5, "dialogue filter"),
]

counts = parametrized_case_count(FILTER_CASES)
# Returns: {"total": 3, "user filter": 1, "assistant filter": 1, "dialogue filter": 1}
```

### verify_coverage()

**Best for**: Verifying consolidation didn't drop tests

```python
from tests.helpers import verify_coverage

old_tests = [
    "test_filter_user_messages",
    "test_filter_assistant_messages",
    "test_filter_dialogue_only"
]

new_cases = [
    ("user_messages", 2, "user filter"),
    ("assistant_messages", 3, "assistant filter"),
    ("dialogue_only", 5, "dialogue filter"),
]

mapping = {
    "test_filter_user_messages": "user filter",
    "test_filter_assistant_messages": "assistant filter",
    "test_filter_dialogue_only": "dialogue filter",
}

result = verify_coverage(old_tests, new_cases, mapping)
# Returns: {
#   "old_count": 3,
#   "new_count": 3,
#   "covered": {"test_filter_user_messages", "test_filter_assistant_messages", "test_filter_dialogue_only"},
#   "missing": [],
#   "extra": [],
#   "coverage_percent": 100.0
# }
```

---

## Fixtures

### db_path

**Usage**: Shortcut for database path setup

```python
def test_something(db_path):
    builder = ConversationBuilder(db_path, "test-conv")
```

### conversation_builder

**Usage**: Factory fixture for ConversationBuilder

```python
def test_something(conversation_builder):
    conv = (conversation_builder("test-conv")
           .add_message("m1", text="Hello")
           .save())
```

### workspace_env

**Usage**: Full workspace environment with paths

```python
def test_something(workspace_env):
    archive_root = workspace_env["archive_root"]
    data_root = workspace_env["data_root"]
```

---

## Before & After Examples

### Before (Original Test - Boilerplate Heavy)

```python
def test_format_basic_conversation(workspace_env):
    db_path = workspace_env["data_root"] / "polylogue" / "polylogue.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conv = ConversationRecord(
        conversation_id="test-conv",
        provider_name="test",
        provider_conversation_id="ext-test-conv",
        title="Test",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        content_hash=uuid4().hex,
    )

    msg = MessageRecord(
        message_id="m1",
        conversation_id="test-conv",
        role="user",
        text="Hello",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash="h1",
    )

    with open_connection(db_path) as conn:
        store_records(conversation=conv, messages=[msg], attachments=[], conn=conn)

    formatter = ConversationFormatter(workspace_env["archive_root"])
    result = formatter.format("test-conv")

    assert "Hello" in result.markdown_text
```

### After (Using Helpers - 70% Less Code)

```python
def test_format_basic_conversation(conversation_builder, workspace_env):
    conv = (conversation_builder("test-conv")
           .add_message("m1", text="Hello")
           .save())

    formatter = ConversationFormatter(workspace_env["archive_root"])
    result = formatter.format("test-conv")

    assert "Hello" in result.markdown_text
```

---

## Impact on Consolidation

**Files using helpers**: 0/8 consolidated files (created after consolidation)

**Expected impact on next 20-30 consolidations**:
- 30-40% less boilerplate per test
- Faster to write parametrized test cases
- More readable test data construction
- Easier to spot missing coverage

---

## When to Use What

| Scenario | Use This |
|----------|----------|
| Building conversation for DB test | `ConversationBuilder(db_path, id).add_message(...).save()` |
| Single message with metadata | `MessageBuilder(id, conv).role(...).meta(...).build()` |
| Quick message for unit test | `make_message(id, role="user", text="Hi")` |
| ChatGPT import test | `make_chatgpt_node(id, role, parts)` |
| Claude AI import test | `make_claude_chat_message(uuid, sender, text)` |
| Claude Code import test | `make_claude_code_message(type, text)` |
| Check message order in output | `assert_messages_ordered(text, "first", "second")` |
| Verify multiple strings present | `assert_contains_all(text, "str1", "str2", "str3")` |
| Verify strings absent | `assert_not_contains_any(text, "ERROR", "FAIL")` |
| Document parametrized coverage | `parametrized_case_count(CASES)` |
| Verify no tests dropped | `verify_coverage(old, new, mapping)` |

---

## Next Steps

1. **Refactor existing consolidated tests** to use helpers (optional)
2. **Use helpers in next 20-30 consolidations** to validate utility
3. **Add more generators** as patterns emerge (Codex, Gemini formats)
4. **Expand assertion helpers** if needed (e.g., `assert_markdown_structure`)

---

## Design Principles

1. **Fluent builders** for complex objects (ConversationBuilder, MessageBuilder)
2. **Quick functions** for simple cases (make_message, make_attachment)
3. **Assertion helpers** for common verification patterns
4. **Generators** for provider-specific test data
5. **Coverage tools** for consolidation verification

**Goal**: Make parametrized tests 30% faster to write, 50% more readable.
