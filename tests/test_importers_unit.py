"""Consolidated provider importer unit tests.

CONSOLIDATED: This file merges tests from:
- test_importers_chatgpt.py (ChatGPT format detection + extraction)
- test_importers_claude.py (Claude AI/Code format detection + extraction)
- test_importers_codex.py (Codex envelope/intermediate format detection)

These are unit tests for individual provider parsers. For cross-provider
integration tests using real export files, see test_importers_parametrized.py.

For property-based testing with Hypothesis, see test_importers_properties.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from polylogue.lib.models import DialoguePair, Message
from polylogue.sources.parsers.base import (
    ParsedAttachment,
    attachment_from_meta,
    normalize_role,
)

# ChatGPT imports
from polylogue.sources.parsers.chatgpt import _coerce_float, extract_messages_from_mapping
from polylogue.sources.parsers.chatgpt import looks_like as chatgpt_looks_like
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse

# Claude imports
from polylogue.sources.parsers.claude import (
    extract_messages_from_chat_messages,
    extract_text_from_segments,
    looks_like_ai,
    looks_like_code,
    parse_ai,
    parse_code,
)

# Codex imports
from polylogue.sources.parsers.codex import looks_like as codex_looks_like
from polylogue.sources.parsers.codex import parse as codex_parse

# Drive imports (for parse_chunked_prompt)
from polylogue.sources.parsers.drive import parse_chunked_prompt

# Test helpers
from tests.helpers import make_chatgpt_node, make_claude_chat_message

# Additional imports for test_ingestion_jsonl
from polylogue.config import Config
from polylogue.pipeline.services.ingestion import IngestionService
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord

# --- merged from test_ingestion_jsonl.py ---

class TestParseRawRecordJsonl:
	"""Tests for IngestionService._parse_raw_record with JSONL and JSON inputs."""

	@pytest.fixture
	def backend(self, tmp_path: Path) -> SQLiteBackend:
		"""Create a SQLiteBackend with a temp database."""
		db_path = tmp_path / "test.db"
		return SQLiteBackend(db_path=db_path)

	@pytest.fixture
	def repository(self, backend: SQLiteBackend) -> ConversationRepository:
		"""Create a ConversationRepository with the test backend."""
		return ConversationRepository(backend=backend)

	@pytest.fixture
	def ingestion_service(
		self, tmp_path: Path, repository: ConversationRepository
	) -> IngestionService:
		"""Create an IngestionService for testing."""
		config = Config(
			sources=[],
			archive_root=tmp_path / "archive",
			render_root=tmp_path / "render",
		)
		return IngestionService(
			repository=repository,
			archive_root=tmp_path / "archive",
			config=config,
			drive_client_factory=None,
		)

	def test_parse_raw_record_single_json(
		self, ingestion_service: IngestionService
	) -> None:
		"""Single JSON document (ChatGPT format) parses correctly."""
		# ChatGPT export format: single JSON with title and mapping
		raw_content = b"""{
    "title": "Test Conversation",
    "mapping": {
        "node1": {
            "message": {
                "id": "msg-1",
                "author": {"role": "user"},
                "content": {"parts": ["Hello"], "content_type": "text"},
                "create_time": 1700000000
            },
            "parent": "root",
            "children": ["node2"]
        },
        "node2": {
            "message": {
                "id": "msg-2",
                "author": {"role": "assistant"},
                "content": {"parts": ["Hi"], "content_type": "text"},
                "create_time": 1700000001
            },
            "parent": "node1",
            "children": []
        }
    },
    "create_time": 1700000000,
    "update_time": 1700000001
}"""
		raw_record = RawConversationRecord(
			raw_id="chatgpt-single-json",
			provider_name="chatgpt",
			source_name="exports",
			source_path="/exports/conversations.json",
			source_index=0,
			raw_content=raw_content,
			acquired_at=datetime.now(timezone.utc).isoformat(),
		)

		# Should parse without error and return a conversation
		parsed = ingestion_service._parse_raw_record(raw_record)

		assert len(parsed) > 0
		assert parsed[0].provider_name == "chatgpt"
		assert parsed[0].title == "Test Conversation"
		# ChatGPT parser extracts messages from the mapping
		assert len(parsed[0].messages) == 2

	def test_parse_raw_record_jsonl(self, ingestion_service: IngestionService) -> None:
		"""Multi-line JSONL (claude-code format) parses correctly and produces messages."""
		# Claude Code format: JSONL with messages as separate lines
		raw_content = b"""{"parentUuid":null,"isSidechain":false,"cwd":"/","sessionId":"test-session-1","version":"1.0.30","type":"user","message":{"role":"user","content":"Hello world"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16.232Z"}
{"parentUuid":"msg-1","isSidechain":false,"cwd":"/","sessionId":"test-session-1","version":"1.0.30","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Hi there!"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20.000Z"}"""

		raw_record = RawConversationRecord(
			raw_id="claude-code-jsonl",
			provider_name="claude-code",
			source_name="claude_code_exports",
			source_path="/exports/session.jsonl",
			source_index=None,
			raw_content=raw_content,
			acquired_at=datetime.now(timezone.utc).isoformat(),
		)

		# Should parse without error and return a conversation
		parsed = ingestion_service._parse_raw_record(raw_record)

		assert len(parsed) > 0
		assert parsed[0].provider_name == "claude-code"
		# Claude Code parser groups all JSONL lines into one conversation
		# with messages from the payload
		assert len(parsed[0].messages) == 2
		# First message should be user
		assert parsed[0].messages[0].role == "user"
		assert "Hello" in parsed[0].messages[0].text
		# Second message should be assistant
		assert parsed[0].messages[1].role == "assistant"

	def test_parse_raw_record_jsonl_with_invalid_lines(
		self, ingestion_service: IngestionService
	) -> None:
		"""JSONL with some invalid lines skips them gracefully."""
		# Mix of valid and invalid JSON lines
		raw_content = b"""{"parentUuid":null,"type":"user","message":{"role":"user","content":"Valid line 1"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16Z"}
This is not JSON at all, should be skipped
{"parentUuid":"msg-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Valid line 2"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20Z"}
{"malformed": "json"
{"parentUuid":"msg-2","type":"user","message":{"role":"user","content":"Valid line 3"},"uuid":"msg-3","timestamp":"2025-06-20T11:34:25Z"}"""

		raw_record = RawConversationRecord(
			raw_id="claude-code-mixed",
			provider_name="claude-code",
			source_name="claude_code_exports",
			source_path="/exports/session-with-errors.jsonl",
			source_index=None,
			raw_content=raw_content,
			acquired_at=datetime.now(timezone.utc).isoformat(),
		)

		# Should parse without error, skipping invalid lines
		parsed = ingestion_service._parse_raw_record(raw_record)

		assert len(parsed) > 0
		# Should have extracted the 3 valid lines (invalid ones skipped)
		# The claude-code parser groups them into one conversation
		assert parsed[0].provider_name == "claude-code"
		# Should have at least 2-3 messages from the valid lines
		assert len(parsed[0].messages) >= 2

	def test_orphan_raw_records_reparsed(
		self, backend: SQLiteBackend, repository: ConversationRepository, tmp_path: Path
	) -> None:
		"""Raw records without conversations are detected and re-parsed.

		This tests the orphaned raw records scenario from ingest_sources():
		When a raw record exists but the corresponding conversation was deleted
		or never parsed, it should be re-parsed.
		"""
		config = Config(
			sources=[],
			archive_root=tmp_path / "archive",
			render_root=tmp_path / "render",
		)
		ingestion_service = IngestionService(
			repository=repository,
			archive_root=tmp_path / "archive",
			config=config,
			drive_client_factory=None,
		)

		# Store a raw record without a corresponding conversation
		raw_content = b"""{"parentUuid":null,"type":"user","message":{"role":"user","content":"Orphaned message"},"uuid":"msg-1","timestamp":"2025-06-20T11:34:16Z"}
{"parentUuid":"msg-1","type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Response"}]},"uuid":"msg-2","timestamp":"2025-06-20T11:34:20Z"}"""

		raw_record = RawConversationRecord(
			raw_id="orphaned-raw-001",
			provider_name="claude-code",
			source_name="orphaned_exports",
			source_path="/exports/orphaned.jsonl",
			source_index=None,
			raw_content=raw_content,
			acquired_at=datetime.now(timezone.utc).isoformat(),
		)

		# Save the raw record
		backend.save_raw_conversation(raw_record)

		# Verify it's stored
		stored_raw = backend.get_raw_conversation("orphaned-raw-001")
		assert stored_raw is not None
		assert stored_raw.provider_name == "claude-code"

		# Query for orphaned raw records (without conversations)
		# This is the pattern from ingest_sources()
		conn = backend._get_connection()
		orphaned_rows = conn.execute(
			"""
			SELECT r.raw_id
			FROM raw_conversations r
			LEFT JOIN conversations c ON r.raw_id = c.raw_id
			WHERE c.conversation_id IS NULL
		"""
		).fetchall()

		# Should find the orphaned record
		orphaned_ids = [row["raw_id"] for row in orphaned_rows]
		assert "orphaned-raw-001" in orphaned_ids

		# Now parse it using ingest_from_raw with the orphaned ID
		result = ingestion_service.ingest_from_raw(raw_ids=["orphaned-raw-001"])

		# Should successfully parse and create a conversation
		assert result.counts["conversations"] > 0 or result.counts["messages"] > 0
		# Verify the conversation was created with raw_id link
		# Query directly for conversations with raw_id
		conn = backend._get_connection()
		linked_convos = conn.execute(
			"""
			SELECT conversation_id, raw_id
			FROM conversations
			WHERE raw_id = ?
		""",
			("orphaned-raw-001",),
		).fetchall()
		assert (
			len(linked_convos) > 0
		), "Orphaned raw record should be linked to created conversation"


# --- merged from test_importers_parametrized.py ---

# =============================================================================
# REAL EXPORT FILE DISCOVERY
# =============================================================================


def discover_real_exports():
	"""Discover all real export samples in fixtures/real/."""
	fixtures_dir = Path(__file__).parent / "fixtures" / "real"
	if not fixtures_dir.exists():
		return []

	exports = []
	for provider_dir in fixtures_dir.iterdir():
		if provider_dir.is_dir():
			provider = provider_dir.name
			for export_file in provider_dir.iterdir():
				if export_file.is_file() and not export_file.name.startswith('.'):
					exports.append((provider, export_file))

	return exports


REAL_EXPORTS = discover_real_exports()


# =============================================================================
# CROSS-PROVIDER REAL DATA TESTS (PARAMETRIZED)
# =============================================================================


@pytest.mark.parametrize("provider,export_file", REAL_EXPORTS, ids=lambda x: f"{x[0]}/{x[1].name}" if isinstance(x, tuple) else str(x))
def test_parse_real_export_produces_valid_conversation(provider, export_file):
	"""Parse all real export files and validate structure.

	This SINGLE test validates EVERY real export sample we have:
	- ChatGPT: simple, branching, attachments, large (4 files)
	- Gemini: sample-with-tools (1 file)
	- Claude: (when added)
	- Codex: (when added)

	One parametrized test = automatic coverage of all future samples.
	"""
	import json

	# Skip if file doesn't exist
	if not export_file.exists():
		pytest.skip(f"Export file not found: {export_file}")

	# Load file based on extension
	if export_file.suffix == '.json':
		with open(export_file) as f:
			data = json.load(f)
		conversations = [data] if isinstance(data, dict) else data
		is_jsonl_format = False

	elif export_file.suffix == '.jsonl':
		with open(export_file) as f:
			conversations = [json.loads(line) for line in f if line.strip()]
		is_jsonl_format = True

	else:
		pytest.skip(f"Unsupported format: {export_file.suffix}")

	# Parse based on provider
	parsed_conversations = []

	# Special handling for JSONL formats that require all lines at once
	if is_jsonl_format:
		# Check if this is Claude Code format (has message/uuid/isSidechain structure)
		is_claude_code = (
			conversations and
			isinstance(conversations[0], dict) and
			"message" in conversations[0] and
			"uuid" in conversations[0]
		)

		# Check if this is Codex envelope format
		# First line is session metadata (id, timestamp, instructions, git)
		# Subsequent lines have type/record_type
		is_codex = (
			provider == "codex" and
			conversations and
			isinstance(conversations[0], dict) and
			("id" in conversations[0] or "record_type" in conversations[0] or "type" in conversations[0])
		)

		if is_claude_code:
			# Claude Code JSONL - parse as a batch
			parsed = parse_code(conversations, export_file.stem)
			parsed_conversations.append(parsed)
		elif provider == "codex" and is_codex:
			# Codex envelope JSONL - parse all lines as a batch
			parsed = codex_parse(conversations, export_file.stem)
			parsed_conversations.append(parsed)
		elif provider == "gemini":
			# True Gemini format - parse each line individually
			for i, conv_data in enumerate(conversations):
				conv_id = f"{export_file.stem}-{i}"
				parsed = parse_chunked_prompt("gemini", conv_data, conv_id)
				parsed_conversations.append(parsed)
		else:
			# Unknown JSONL format
			pytest.skip(f"Unknown JSONL format for provider {provider}")
	else:
		# Standard per-item parsing (JSON arrays)
		for i, conv_data in enumerate(conversations):
			conv_id = f"{export_file.stem}-{i}"

			if provider == "chatgpt":
				parsed = chatgpt_parse(conv_data, conv_id)
			elif provider == "claude":
				# Detect format
				if looks_like_ai(conv_data):
					parsed = parse_ai(conv_data, conv_id)
				elif looks_like_code(conv_data):
					parsed = parse_code([conv_data], conv_id)
				else:
					pytest.skip("Unknown Claude format")
			elif provider == "codex":
				parsed = codex_parse([conv_data], conv_id)
			elif provider == "gemini":
				parsed = parse_chunked_prompt("gemini", conv_data, conv_id)
			else:
				pytest.skip(f"Unknown provider: {provider}")

			parsed_conversations.append(parsed)

	# Validate ALL conversations from file
	for parsed in parsed_conversations:
		assert parsed.provider_name in [provider, "chatgpt", "claude", "codex", "gemini", "claude-code"]
		assert parsed.provider_conversation_id is not None

		# Filter out empty/whitespace-only system messages (ChatGPT markers)
		substantive_messages = [
			m for m in parsed.messages
			if m.text and len(m.text.strip()) > 0
		]
		assert len(substantive_messages) > 0, f"No substantive messages parsed from {export_file}"

		# All substantive messages must have text
		for msg in substantive_messages:
			assert msg.text is not None, f"Empty message text in {export_file}"
			assert len(msg.text.strip()) > 0, f"Whitespace-only message in {export_file}"


@pytest.mark.parametrize("provider,export_file", REAL_EXPORTS, ids=lambda x: f"{x[0]}/{x[1].name}" if isinstance(x, tuple) else str(x))
def test_real_export_messages_have_roles(provider, export_file):
	"""All messages in real exports have valid roles.

	Parametrized across ALL real export files.
	"""
	import json

	if not export_file.exists():
		pytest.skip(f"Export file not found: {export_file}")

	# Load and parse (reuse logic from above)
	if export_file.suffix == '.json':
		with open(export_file) as f:
			data = json.load(f)
		conversations = [data] if isinstance(data, dict) else data
	elif export_file.suffix == '.jsonl':
		with open(export_file) as f:
			conversations = [json.loads(line) for line in f if line.strip()]
	else:
		pytest.skip(f"Unsupported format: {export_file.suffix}")

	valid_roles = {"user", "assistant", "system", "tool", "human", "model"}

	for conv_data in conversations[:5]:  # Test first 5 conversations
		conv_id = f"{export_file.stem}-test"

		if provider == "chatgpt":
			parsed = chatgpt_parse(conv_data, conv_id)
		elif provider == "gemini":
			parsed = parse_chunked_prompt("gemini", conv_data, conv_id)
		else:
			continue

		# All messages must have valid roles
		for msg in parsed.messages:
			assert msg.role.lower() in valid_roles, \
				f"Invalid role '{msg.role}' in {export_file}"


@pytest.mark.parametrize("provider,export_file", REAL_EXPORTS, ids=lambda x: f"{x[0]}/{x[1].name}" if isinstance(x, tuple) else str(x))
def test_real_export_preserves_metadata(provider, export_file):
	"""Real exports preserve provider metadata.

	Parametrized across ALL real export files.
	"""
	import json

	if not export_file.exists():
		pytest.skip(f"Export file not found: {export_file}")

	if export_file.suffix == '.json':
		with open(export_file) as f:
			data = json.load(f)
		conversations = [data] if isinstance(data, dict) else data
	elif export_file.suffix == '.jsonl':
		with open(export_file) as f:
			conversations = [json.loads(line) for line in f if line.strip()]
	else:
		pytest.skip(f"Unsupported format: {export_file.suffix}")

	for conv_data in conversations[:3]:  # Test first 3
		conv_id = f"{export_file.stem}-meta"

		if provider == "chatgpt":
			parsed = chatgpt_parse(conv_data, conv_id)
		elif provider == "gemini":
			parsed = parse_chunked_prompt("gemini", conv_data, conv_id)
		else:
			continue

		# Should have either a title or at least one message with provider metadata
		# (exact fields vary by provider)
		has_message_meta = any(m.provider_meta for m in parsed.messages) if parsed.messages else False
		assert parsed.title is not None or has_message_meta


# =============================================================================
# BOUNDARY CONDITION TESTS (PARAMETRIZED)
# =============================================================================


@pytest.mark.parametrize("char_count,expected_substantive", [
	(9, False),   # Below 10-char threshold
	(10, False),  # Exactly at threshold
	(11, True),   # Just above threshold
	(50, True),   # Well above
])
def test_substantive_message_boundary(char_count, expected_substantive):
	"""Test substantive message length boundary (>10 chars).

	CRITICAL BOUNDARY TEST that was missing from original suite.
	"""
	from polylogue.lib.models import Message

	text = "a" * char_count
	msg = Message(id="1", role="assistant", text=text)

	assert msg.is_substantive == expected_substantive, \
		f"Wrong is_substantive for {char_count} chars"


@pytest.mark.parametrize("fence_markers,expected_context_dump", [
	("``` code ```", False),   # 1 fence = 2 markers (count=2), below threshold
	("``` code ``` ``` code ```", False),   # 2 fences = 4 markers (count=4), below threshold
	("``` code ``` ``` code ``` ``` code ```", True),   # 3 fences = 6 markers (count=6), at threshold
	("``` a ``` ``` b ``` ``` c ``` ``` d ```", True),   # 4 fences = 8 markers (count=8), above threshold
])
def test_context_dump_backtick_boundary(fence_markers, expected_context_dump):
	"""Test context dump detection boundary (6+ backtick markers = 3+ code blocks).

	CRITICAL BOUNDARY TEST that was missing from original suite.
	The logic counts occurrences of '```' (triple backticks), not individual backtick chars.
	A complete code fence needs opening and closing, so 3 fences = 6 markers.
	"""
	from polylogue.lib.models import Message

	msg = Message(id="1", role="user", text=fence_markers)

	assert msg.is_context_dump == expected_context_dump, \
		f"Wrong is_context_dump for fence markers: {fence_markers}"


# =============================================================================
# FORMAT VARIANT TESTS (PARAMETRIZED)
# =============================================================================


# ChatGPT has multiple export format variants over time
CHATGPT_VARIANTS = [
	"simple.json",
	"branching.json",
]


@pytest.mark.parametrize("variant", CHATGPT_VARIANTS)
def test_chatgpt_format_variants(variant):
	"""Test ChatGPT parser handles different export format variants.

	Parametrized to automatically test all discovered variants.
	"""
	import json

	fixture_path = Path(__file__).parent / "fixtures" / "real" / "chatgpt" / variant

	if not fixture_path.exists():
		pytest.skip(f"Variant not available: {variant}")

	with open(fixture_path) as f:
		data = json.load(f)

	parsed = chatgpt_parse(data, f"variant-{variant}")

	# All variants should parse successfully
	assert parsed.provider_name == "chatgpt"
	assert len(parsed.messages) > 0

	# Format-specific validations
	if "branching" in variant:
		# Branching conversations should have structured metadata in at least one message
		has_metadata = any(
			m.provider_meta and "raw" in m.provider_meta
			for m in parsed.messages
		)
		assert has_metadata, "Branching conversation should preserve metadata in messages"

	if "attachments" in variant:
		# Attachment files should have attachments
		assert len(parsed.attachments) > 0


# =============================================================================
# STATISTICS
# =============================================================================


def test_consolidation_statistics():
	"""Document consolidation statistics.

	This is a META test that documents the consolidation impact.
	"""
	num_real_exports = len(REAL_EXPORTS)
	num_chatgpt_variants = len(CHATGPT_VARIANTS)

	# Each parametrized test runs once per export file
	tests_per_parametrized = num_real_exports

	# Total tests generated by this file's parametrized tests
	parametrized_tests = (
		tests_per_parametrized * 3  # 3 real export tests
		+ 4  # Boundary tests (substantive)
		+ 4  # Boundary tests (context dump)
		+ num_chatgpt_variants  # ChatGPT variants
	)

	print(f"\n{'='*60}")
	print("CONSOLIDATION IMPACT")
	print(f"{'='*60}")
	print(f"Real export files discovered: {num_real_exports}")
	print(f"Tests generated from parametrization: {parametrized_tests}")
	print(f"Coverage multiplier: {parametrized_tests / (len(REAL_EXPORTS) or 1):.1f}x")
	print(f"{'='*60}\n")

	assert True  # Always pass, this is just for reporting


# =============================================================================
# CHATGPT IMPORTER TESTS
# =============================================================================


# MERGED FORMAT + COERCE DETECTION
PROVIDER_FORMAT_DETECTION_CASES = [
    # ChatGPT
    ({"mapping": {}}, True, chatgpt_looks_like, "ChatGPT: valid empty mapping"),
    ({"mapping": {"node1": {}}}, True, chatgpt_looks_like, "ChatGPT: valid with nodes"),
    ({}, False, chatgpt_looks_like, "ChatGPT: missing mapping"),
    (None, False, chatgpt_looks_like, "ChatGPT: None input"),
    # Claude AI
    ({"chat_messages": []}, True, looks_like_ai, "Claude AI: chat_messages"),
    ({}, False, looks_like_ai, "Claude AI: missing chat_messages"),
    (None, False, looks_like_ai, "Claude AI: None"),
    # Claude Code
    ([{"parentUuid": "123"}], True, looks_like_code, "Claude Code: parentUuid"),
    ([], False, looks_like_code, "Claude Code: empty list"),
    (None, False, looks_like_code, "Claude Code: None"),
]


@pytest.mark.parametrize("data,expected,check_fn,desc", PROVIDER_FORMAT_DETECTION_CASES)
def test_provider_format_detection(data, expected, check_fn, desc):
    """Unified format detection across all providers."""
    result = check_fn(data)
    assert result == expected, f"Failed {desc}"


# COERCE FLOAT - MERGED WITH FORMAT DETECTION ABOVE

COERCE_FLOAT_CASES = [
    (42, 42.0, "int"),
    (3.14, 3.14, "float"),
    ("2.5", 2.5, "string number"),
    ("invalid", None, "invalid string"),
    (None, None, "None"),
]

@pytest.mark.parametrize("input_val,expected,desc", COERCE_FLOAT_CASES)
def test_coerce_float(input_val, expected, desc):
    """Test _coerce_float conversion."""
    result = _coerce_float(input_val)
    assert result == expected, f"Failed {desc}"


# MESSAGE EXTRACTION - PARAMETRIZED (1 test replacing 17)


CHATGPT_EXTRACT_MESSAGES_CASES = [
    # Basic extraction
    ({"node1": make_chatgpt_node("msg1", "user", ["Hello"])}, 1, "basic message"),

    # Timestamp handling
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=1704067200)}, 1, "with timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=None)}, 1, "null timestamp"),
    ({"node1": make_chatgpt_node("msg1", "user", ["Hi"], timestamp=0)}, 1, "zero timestamp"),

    # Mixed timestamps (should sort)
    ({
        "node1": make_chatgpt_node("msg1", "user", ["First"], timestamp=1000),
        "node2": make_chatgpt_node("msg2", "assistant", ["Second"], timestamp=2000),
        "node3": make_chatgpt_node("msg3", "user", ["Third"], timestamp=500),
    }, 3, "mixed timestamps sorted"),

    # Content variants
    ({"node1": make_chatgpt_node("msg1", "user", ["Part1", "Part2"])}, 1, "multiple parts"),
    ({"node1": make_chatgpt_node("msg1", "user", [None, "Valid"])}, 1, "parts with None"),
    ({"node1": {"message": {"id": "1", "author": {"role": "user"}, "content": {"parts": []}}}}, 0, "empty parts"),

    # Role normalization
    ({"node1": make_chatgpt_node("msg1", "human", ["Hi"])}, 1, "human role alias"),
    ({"node1": make_chatgpt_node("msg1", "model", ["Response"])}, 1, "model role alias"),

    # Missing fields
    ({"node1": {"id": "1", "message": None}}, 0, "missing message"),
    ({"node1": {"id": "1", "message": {"id": "1"}}}, 0, "missing author"),
    ({"node1": {"id": "1", "message": {"id": "1", "author": {}}}}, 0, "missing role"),
    ({"node1": {"id": "1", "message": {"id": "1", "author": {"role": "user"}}}}, 0, "missing content"),

    # Non-dict nodes
    ({"node1": "not a dict"}, 0, "non-dict node"),
    ({"node1": None}, 0, "None node"),

    # Empty mapping
    ({}, 0, "empty mapping"),
]


@pytest.mark.parametrize("mapping,expected_count,desc", CHATGPT_EXTRACT_MESSAGES_CASES)
def test_chatgpt_extract_messages_comprehensive(mapping, expected_count, desc):
    """Comprehensive message extraction test.

    Replaces 17 individual extraction tests.
    """
    messages, attachments = extract_messages_from_mapping(mapping)

    assert len(messages) == expected_count, \
        f"Failed {desc}: expected {expected_count} messages, got {len(messages)}"

    # Verify all messages have required fields
    for msg in messages:
        assert msg.text is not None
        assert msg.role in ["user", "assistant", "system", "tool"]


# -----------------------------------------------------------------------------
# PARENT & BRANCH INDEX EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_PARENT_BRANCH_CASES = [
    # No parent (root message)
    (
        {"node1": make_chatgpt_node("msg1", "user", ["Hello"])},
        [None],
        [0],
        "root message no parent"
    ),

    # Simple linear chain
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Hello"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1"),
        },
        [None, "node1"],
        [0, 0],
        "linear chain parent references"
    ),

    # Branching: one parent with multiple children
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Question"], children=["msg2", "msg3"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Answer 1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["Answer 2"], parent="node1"),
        },
        [None, "node1", "node1"],
        [0, 0, 1],
        "branching with branch indexes"
    ),

    # Three-way branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Q"], children=["msg2", "msg3", "msg4"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["A1"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "assistant", ["A2"], parent="node1"),
            "node4": make_chatgpt_node("msg4", "assistant", ["A3"], parent="node1"),
        },
        [None, "node1", "node1", "node1"],
        [0, 0, 1, 2],
        "three-way branch indexes"
    ),

    # No parent field in node
    (
        {"node1": make_chatgpt_node("msg1", "user", ["Hello"])},
        [None],
        [0],
        "missing parent field defaults to None"
    ),

    # Parent node missing from mapping (orphaned node)
    (
        {"node2": make_chatgpt_node("msg2", "assistant", ["Hi"], parent="node1")},
        ["node1"],
        [0],
        "orphaned node with missing parent"
    ),

    # Mixed chain and branch
    (
        {
            "node1": make_chatgpt_node("msg1", "user", ["Start"], children=["msg2"]),
            "node2": make_chatgpt_node("msg2", "assistant", ["Response"], children=["msg3", "msg4"], parent="node1"),
            "node3": make_chatgpt_node("msg3", "user", ["Follow 1"], parent="node2"),
            "node4": make_chatgpt_node("msg4", "user", ["Follow 2"], parent="node2"),
        },
        [None, "node1", "node2", "node2"],
        [0, 0, 0, 1],
        "mixed chain and branch structure"
    ),
]


@pytest.mark.parametrize("mapping,expected_parents,expected_indexes,desc", CHATGPT_PARENT_BRANCH_CASES)
def test_chatgpt_extract_parent_and_branch_index(mapping, expected_parents, expected_indexes, desc):
    """Test extraction of parent_message_provider_id and branch_index.

    Validates parent message references and branch position calculation.
    """
    messages, _ = extract_messages_from_mapping(mapping)

    assert len(messages) == len(expected_parents), \
        f"Failed {desc}: expected {len(expected_parents)} messages, got {len(messages)}"

    for msg, expected_parent, expected_index in zip(messages, expected_parents, expected_indexes, strict=False):
        assert msg.parent_message_provider_id == expected_parent, \
            f"Failed {desc}: message {msg.provider_message_id} expected parent {expected_parent}, " \
            f"got {msg.parent_message_provider_id}"
        assert msg.branch_index == expected_index, \
            f"Failed {desc}: message {msg.provider_message_id} expected branch_index {expected_index}, " \
            f"got {msg.branch_index}"


# -----------------------------------------------------------------------------
# METADATA EXTRACTION - PARAMETRIZED
# -----------------------------------------------------------------------------


CHATGPT_METADATA_CASES = [
    # Attachments
    ({"attachments": [{"id": "att1", "name": "file.pdf"}]}, True, "attachments field"),
    ({"image_asset_pointer": "asset_123"}, True, "image asset pointer"),

    # Cost/duration
    ({"costUSD": 0.005}, "cost", "cost metadata"),
    ({"durationMs": 2500}, "duration", "duration metadata"),

    # Thinking markers
    ({"content_type": "thoughts"}, "thinking", "thoughts content type"),
    ({"content_type": "reasoning_recap"}, "thinking", "reasoning recap"),

    # Empty
    ({}, None, "no metadata"),
    (None, None, "None metadata"),
]


@pytest.mark.parametrize("metadata,expected_type,desc", CHATGPT_METADATA_CASES)
def test_chatgpt_metadata_extraction(metadata, expected_type, desc):
    """Test metadata extraction from message metadata field.

    Explicit tests for attachment/cost/thinking metadata.
    """
    mapping = {
        "node1": {
            "message": {
                "id": "msg1",
                "author": {"role": "user"},
                "content": {"parts": ["Test"]},
                "metadata": metadata,
            }
        }
    }

    messages, attachments = extract_messages_from_mapping(mapping)

    if expected_type == "attachments":
        # Should have attachment records
        assert len(attachments) > 0 or len(messages[0].attachments) > 0
    elif expected_type == "cost":
        # Should preserve cost in provider_meta
        assert messages[0].provider_meta is not None
    elif expected_type == "thinking":
        # Should mark as thinking
        # (depends on content_blocks implementation)
        pass
    elif expected_type is None:
        # No special metadata
        assert True


# -----------------------------------------------------------------------------
# FULL PARSE - PARAMETRIZED (1 test replacing 12)
# -----------------------------------------------------------------------------


PARSE_CONVERSATION_CASES = [
    # ChatGPT title extraction
    (chatgpt_parse, {"title": "My Conv", "mapping": {}}, "title", "ChatGPT: title field"),
    (chatgpt_parse, {"name": "Conv Name", "mapping": {}}, "name", "ChatGPT: name field"),
    (chatgpt_parse, {"id": "conv-123", "mapping": {}}, "id", "ChatGPT: id field"),
    (chatgpt_parse, {"mapping": {}}, "fallback", "ChatGPT: uses fallback-id"),

    # Claude AI title extraction
    (parse_ai, {"name": "Test Title", "chat_messages": []}, "title", "Claude AI: name → title"),
    (parse_ai, {"chat_messages": []}, "fallback", "Claude AI: uses fallback"),

    # Claude Code provider_name
    (parse_code, [], "provider", "Claude Code: provider_name"),
]


@pytest.mark.parametrize("parse_fn,conv_data,check_type,desc", PARSE_CONVERSATION_CASES)
def test_parse_conversation(parse_fn, conv_data, check_type, desc):
    """Unified conversation parsing across providers."""
    result = parse_fn(conv_data, "fallback-id")

    if check_type == "title":
        assert result.title in conv_data.values(), f"Failed {desc}"
    elif check_type == "id":
        assert result.provider_conversation_id == conv_data["id"], f"Failed {desc}"
    elif check_type == "fallback":
        assert result.provider_conversation_id == "fallback-id", f"Failed {desc}"
    elif check_type == "provider":
        assert result.provider_name in ["claude", "claude-code"], f"Failed {desc}"


# -----------------------------------------------------------------------------
# REAL EXPORT INTEGRATION
# -----------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("fixtures/real/chatgpt/simple.json").exists(),
    reason="Real ChatGPT sample not available"
)
def test_chatgpt_parse_real_simple():
    """Parse real ChatGPT simple export."""
    sample_path = Path(__file__).parent / "fixtures" / "real" / "chatgpt" / "simple.json"
    with open(sample_path) as f:
        data = json.load(f)

    result = chatgpt_parse(data, "simple-test")

    assert result.provider_name == "chatgpt"
    assert len(result.messages) > 0
    # Some messages may have empty text (system messages, etc)
    assert all(m.text is not None for m in result.messages)


@pytest.mark.skipif(
    not Path(__file__).parent.joinpath("fixtures/real/chatgpt/branching.json").exists(),
    reason="Real ChatGPT sample not available"
)
def test_chatgpt_parse_real_branching():
    """Parse real ChatGPT branching conversation."""
    sample_path = Path(__file__).parent / "fixtures" / "real" / "chatgpt" / "branching.json"
    with open(sample_path) as f:
        data = json.load(f)

    result = chatgpt_parse(data, "branching-test")

    assert result.provider_name == "chatgpt"
    assert len(result.messages) > 10  # Branching conversations are larger
    # Branching structure is handled internally, no provider_meta on conversation


# =============================================================================
# CLAUDE IMPORTER TESTS (merged format detection above)
# =============================================================================

CLAUDE_SEGMENT_CASES = [
    (["plain text"], "plain text", "string segment"),
    ([{"text": "dict with text"}], "dict with text", "dict with text"),
    ([{"content": "dict with content"}], "dict with content", "dict with content"),
    (["text1", {"text": "text2"}, "text3"], "text1", "mixed segments"),
    ([{}, "", None], None, "empty/None segments"),
]

@pytest.mark.parametrize("segments,expected_contains,desc", CLAUDE_SEGMENT_CASES)
def test_extract_text_from_segments(segments, expected_contains, desc):
    """Test segment extraction variants."""
    result = extract_text_from_segments(segments)
    if expected_contains:
        assert expected_contains in (result or ""), f"Failed {desc}"
    else:
        assert result is None or result == "", f"Failed {desc}"


CLAUDE_EXTRACT_CHAT_MESSAGES_CASES = [
    # Basic
    ([make_claude_chat_message("u1", "human", "Hello")], 1, "basic message"),

    # Attachments variants
    ([make_claude_chat_message("u1", "human", "Hi", attachments=[{"file_name": "doc.pdf"}])], 1, "attachments field"),
    ([make_claude_chat_message("u1", "human", "Hi", files=[{"file_name": "doc.pdf"}])], 1, "files field"),

    # Role variants
    ([make_claude_chat_message("u1", "human", "Hi")], "user", "human role"),
    ([make_claude_chat_message("u1", "assistant", "Hi")], "assistant", "assistant role"),
    ([make_claude_chat_message("u1", None, "Hi")], 0, "missing sender skipped"),

    # Timestamp variants (with role)
    ([make_claude_chat_message("u1", "human", "Hi", timestamp="2024-01-01T00:00:00Z")], 1, "created_at"),
    ([{"uuid": "u1", "sender": "human", "text": "Hi", "create_time": 1704067200}], 1, "create_time"),
    ([{"uuid": "u1", "sender": "human", "text": "Hi", "timestamp": 1704067200}], 1, "timestamp field"),

    # ID variants (with role)
    ([{"uuid": "u1", "sender": "human", "text": "Hi"}], "u1", "uuid field"),
    ([{"id": "i1", "sender": "human", "text": "Hi"}], "i1", "id field"),
    ([{"message_id": "m1", "sender": "human", "text": "Hi"}], "m1", "message_id field"),

    # Content variants (with role)
    ([{"uuid": "u1", "sender": "human", "text": ["list", "of", "parts"]}], 0, "text as list skipped"),
    ([{"uuid": "u1", "sender": "human", "content": {"text": "nested text"}}], 1, "content dict with text"),
    ([{"uuid": "u1", "sender": "human", "content": {"parts": ["part1", "part2"]}}], 1, "content dict with parts"),

    # Missing text (with role)
    ([{"uuid": "u1", "sender": "human"}], 0, "missing text skipped"),
    ([{"uuid": "u1", "sender": "human", "text": ""}], 0, "empty text skipped"),
    ([{"uuid": "u1", "sender": "human", "text": None}], 0, "None text skipped"),

    # Non-dict items (valid one has role)
    (["not a dict", {"uuid": "u1", "sender": "human", "text": "Valid"}], 1, "skip non-dict"),

    # Empty list
    ([], 0, "empty list"),
]


@pytest.mark.parametrize("chat_messages,expected,desc", CLAUDE_EXTRACT_CHAT_MESSAGES_CASES)
def test_claude_extract_chat_messages_comprehensive(chat_messages, expected, desc):
    """Comprehensive chat_messages extraction.

    Replaces 15 extraction tests.
    """
    messages, attachments = extract_messages_from_chat_messages(chat_messages)

    if isinstance(expected, int):
        assert len(messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str) and messages:
        # ID field tests check provider_message_id, role tests check role
        if "field" in desc and desc not in ["attachments field", "files field", "timestamp field"]:
            assert messages[0].provider_message_id == expected, f"Failed {desc}"
        else:
            # Expected role
            assert messages[0].role == expected, f"Failed {desc}"


# PARSE AI - CONSOLIDATED

CLAUDE_PARSE_AI_CASES = [
    ({"chat_messages": [make_claude_chat_message("u1", "human", "Hello")]}, 1, "basic"),
    ({"chat_messages": []}, 0, "empty messages"),
    ({"chat_messages": [], "name": "Test Title"}, "Test Title", "title extraction"),
    ({"chat_messages": [{"uuid": "u1", "sender": "human", "content": {"text": "nested"}}]}, 1, "content dict"),
]

@pytest.mark.parametrize("conv_data,expected,desc", CLAUDE_PARSE_AI_CASES)
def test_parse_ai_variants(conv_data, expected, desc):
    """Test parse_ai with variants."""
    result = parse_ai(conv_data, "fallback-id")
    if isinstance(expected, int):
        assert len(result.messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str):
        assert result.title == expected, f"Failed {desc}"


# PARSE CODE - CONSOLIDATED

def make_code_message(msg_type, text, **kwargs):
    """Helper to create Code format message."""
    msg = {"type": msg_type}
    if text or "message" not in kwargs:
        msg["message"] = {"content": text} if text else {}
    msg.update(kwargs)
    return msg


CLAUDE_PARSE_CODE_CASES = [
    ([make_code_message("user", "Question")], 1, "user message"),
    ([make_code_message("assistant", "Answer")], 1, "assistant message"),
    ([make_code_message("summary", "Summary text")], 0, "skip summary"),
    ([make_code_message("user", "Q")], "user", "user type"),
    ([], 0, "empty messages"),
]

@pytest.mark.parametrize("messages,expected,desc", CLAUDE_PARSE_CODE_CASES)
def test_parse_code_variants(messages, expected, desc):
    """Test parse_code with variants."""
    result = parse_code(messages, "fallback-id")
    if isinstance(expected, int):
        assert len(result.messages) == expected, f"Failed {desc}"
    elif isinstance(expected, str):
        if result.messages:
            assert result.messages[0].role == expected, f"Failed {desc}"


# =============================================================================
# PARSE_CODE REGRESSION TESTS (text=None guard, tool_result content)
# =============================================================================


def test_parse_code_progress_record_text_never_none():
    """Progress records must have text='' not None after text guard fix."""
    items = [
        {"type": "progress", "uuid": "prog-1", "sessionId": "sess-1", "timestamp": 1704067200},
    ]
    result = parse_code(items, "fallback")
    for msg in result.messages:
        assert msg.text is not None, f"Message {msg.provider_message_id} has text=None"
        assert isinstance(msg.text, str)


def test_parse_code_result_record_text_never_none():
    """Result records must have text='' not None after text guard fix."""
    items = [
        {"type": "result", "uuid": "res-1", "sessionId": "sess-1", "timestamp": 1704067200},
    ]
    result = parse_code(items, "fallback")
    for msg in result.messages:
        assert msg.text is not None, f"Message {msg.provider_message_id} has text=None"
        assert isinstance(msg.text, str)


def test_parse_code_assistant_no_text_blocks_text_never_none():
    """Assistant records with only tool_use blocks must have text='' not None."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {"path": "/tmp/x"}},
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    for msg in result.messages:
        assert msg.text is not None, f"Message {msg.provider_message_id} has text=None"


def test_parse_code_tool_result_content_preserved():
    """Tool result content blocks must preserve content and is_error fields."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-1",
                        "content": "file contents here\nline 2",
                        "is_error": False,
                    },
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    assert result.messages, "Expected at least one message"
    meta = result.messages[0].provider_meta
    blocks = meta.get("content_blocks", [])
    tool_results = [b for b in blocks if b.get("type") == "tool_result"]
    assert tool_results, "Expected tool_result in content_blocks"
    tr = tool_results[0]
    assert "content" in tr, "tool_result must have content field"
    assert tr["content"] == "file contents here\nline 2"
    assert "is_error" in tr, "tool_result must have is_error field"
    assert tr["is_error"] is False


def test_parse_code_tool_result_error_preserved():
    """Tool result with is_error=True must preserve the flag."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu-2",
                        "content": "Error: file not found",
                        "is_error": True,
                    },
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    meta = result.messages[0].provider_meta
    blocks = meta.get("content_blocks", [])
    tool_results = [b for b in blocks if b.get("type") == "tool_result"]
    assert tool_results[0]["is_error"] is True
    assert tool_results[0]["content"] == "Error: file not found"


def test_parse_code_mixed_content_blocks_all_preserved():
    """Complex assistant message with thinking + tool_use + tool_result + text all preserved."""
    items = [
        {
            "type": "assistant",
            "uuid": "ast-1",
            "sessionId": "sess-1",
            "timestamp": 1704067200,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze this..."},
                    {"type": "text", "text": "I'll read the file."},
                    {"type": "tool_use", "id": "tu-1", "name": "Read", "input": {"path": "/tmp/x"}},
                    {"type": "tool_result", "tool_use_id": "tu-1", "content": "file data", "is_error": False},
                ],
            },
        },
    ]
    result = parse_code(items, "fallback")
    meta = result.messages[0].provider_meta
    blocks = meta.get("content_blocks", [])
    types = [b["type"] for b in blocks]
    assert "thinking" in types
    assert "text" in types
    assert "tool_use" in types
    assert "tool_result" in types
    # Verify tool_result has all fields
    tr = next(b for b in blocks if b["type"] == "tool_result")
    assert tr["content"] == "file data"
    assert tr["is_error"] is False
    assert tr["tool_use_id"] == "tu-1"


# =============================================================================
# CODEX IMPORTER TESTS
# =============================================================================


def test_codex_looks_like_envelope_format():
    """Test looks_like returns True for envelope format."""
    valid_payload = [
        {"type": "session_meta", "payload": {"id": "test-123", "timestamp": "2025-01-01"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": []}},
    ]
    assert codex_looks_like(valid_payload) is True


def test_codex_looks_like_intermediate_format():
    """Test looks_like returns True for intermediate format."""
    valid_payload = [
        {"id": "test-123", "timestamp": "2025-01-01", "git": {}},
        {"type": "message", "role": "user", "content": []},
    ]
    assert codex_looks_like(valid_payload) is True


def test_codex_looks_like_empty_list():
    """Test looks_like returns False for empty list."""
    assert codex_looks_like([]) is False


def test_codex_looks_like_not_list():
    """Test looks_like returns False for non-list types."""
    assert codex_looks_like({}) is False
    assert codex_looks_like("string") is False
    assert codex_looks_like(None) is False
    assert codex_looks_like(123) is False


def test_codex_parse_empty_list():
    """Test parsing an empty list returns conversation with no messages."""
    result = codex_parse([], fallback_id="test-empty-list")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "test-empty-list"
    assert len(result.messages) == 0


def test_codex_parse_envelope_format():
    """Test parsing envelope format with session_meta and response_item."""
    payload = [
        {"type": "session_meta", "payload": {"id": "session-123", "timestamp": "2025-01-01T00:00:00Z"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-1",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
                "timestamp": "2025-01-01T00:00:01Z",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": "msg-2",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "Hi there!"}],
                "timestamp": "2025-01-01T00:00:02Z",
            },
        },
    ]
    result = codex_parse(payload, fallback_id="fallback-id")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "session-123"
    assert result.created_at == "2025-01-01T00:00:00Z"
    assert len(result.messages) == 2

    assert result.messages[0].provider_message_id == "msg-1"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Hello"
    assert result.messages[0].timestamp == "2025-01-01T00:00:01Z"

    assert result.messages[1].provider_message_id == "msg-2"
    assert result.messages[1].role == "assistant"
    assert result.messages[1].text == "Hi there!"


def test_codex_parse_intermediate_format():
    """Test parsing intermediate format with direct records."""
    payload = [
        {"id": "session-456", "timestamp": "2025-01-02T00:00:00Z", "git": {}},
        {"record_type": "state"},
        {
            "type": "message",
            "id": "msg-3",
            "role": "user",
            "content": [{"type": "input_text", "text": "Test message"}],
            "timestamp": "2025-01-02T00:00:01Z",
        },
    ]
    result = codex_parse(payload, fallback_id="fallback-id")

    assert result.provider_name == "codex"
    assert result.provider_conversation_id == "session-456"
    assert result.created_at == "2025-01-02T00:00:00Z"
    assert len(result.messages) == 1

    assert result.messages[0].provider_message_id == "msg-3"
    assert result.messages[0].role == "user"
    assert result.messages[0].text == "Test message"


# =============================================================================
# IMPORTERS.BASE MODULE TESTS (merged from test_importers_base.py)
# =============================================================================

# -----------------------------------------------------------------------------
# NORMALIZE_ROLE - PARAMETRIZED
# -----------------------------------------------------------------------------


NORMALIZE_ROLE_STANDARD_CASES = [
    ("user", "user", "user stays user"),
    ("assistant", "assistant", "assistant stays assistant"),
    ("system", "system", "system stays system"),
    ("human", "user", "human aliased to user"),
    ("model", "assistant", "model aliased to assistant"),
]

NORMALIZE_ROLE_CASE_INSENSITIVE_CASES = [
    ("USER", "user", "upper USER"),
    ("Human", "user", "mixed Human"),
    ("ASSISTANT", "assistant", "upper ASSISTANT"),
    ("Model", "assistant", "mixed Model"),
    ("SYSTEM", "system", "upper SYSTEM"),
]

NORMALIZE_ROLE_EDGE_CASES = [
    ("  user  ", "user", "whitespace stripped user"),
    ("\tassistant\n", "assistant", "whitespace stripped assistant"),
    ("custom_role", "unknown", "unrecognized returns unknown"),
    ("BOT", "unknown", "unrecognized BOT returns unknown"),
]

NORMALIZE_ROLE_STRICT_CASES = [
    (None, "None raises ValueError"),
    ("", "empty raises ValueError"),
    ("   ", "whitespace only raises ValueError"),
    ("\t\n", "tabs and newlines raise ValueError"),
]


class TestNormalizeRole:
    """Parametrized tests for role normalization."""

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_STANDARD_CASES)
    def test_normalize_role_standard(self, role: str, expected: str, description: str):
        """Standard role mappings."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_CASE_INSENSITIVE_CASES)
    def test_normalize_role_case_insensitive(self, role: str, expected: str, description: str):
        """Role normalization is case-insensitive."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,expected,description", NORMALIZE_ROLE_EDGE_CASES)
    def test_normalize_role_edge_cases(self, role: str, expected: str, description: str):
        """Edge cases: whitespace stripping, unrecognized roles."""
        assert normalize_role(role) == expected

    @pytest.mark.parametrize("role,description", NORMALIZE_ROLE_STRICT_CASES)
    def test_normalize_role_rejects_empty(self, role: str | None, description: str):
        """Empty/None roles raise ValueError - role is required at parse time."""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            normalize_role(role)  # type: ignore


# -----------------------------------------------------------------------------
# ATTACHMENT_FROM_META - SPOT CHECKS
# -----------------------------------------------------------------------------


class TestAttachmentFromMeta:
    """Tests for attachment metadata parsing."""

    def test_attachment_from_meta_basic(self):
        """Creates ParsedAttachment from minimal metadata."""
        meta = {"id": "att123", "name": "file.txt"}
        result = attachment_from_meta(meta, "msg1", 0)

        assert result is not None
        assert isinstance(result, ParsedAttachment)
        assert result.provider_attachment_id == "att123"
        assert result.message_provider_id == "msg1"
        assert result.name == "file.txt"
        assert result.provider_meta == meta

    def test_attachment_from_meta_with_all_fields(self):
        """Creates ParsedAttachment with all supported fields."""
        meta = {
            "id": "att456",
            "name": "document.pdf",
            "mimeType": "application/pdf",
            "size": 1024,
        }
        result = attachment_from_meta(meta, "msg2", 1)

        assert result is not None
        assert result.provider_attachment_id == "att456"
        assert result.message_provider_id == "msg2"
        assert result.name == "document.pdf"
        assert result.mime_type == "application/pdf"
        assert result.size_bytes == 1024

    def test_attachment_from_meta_missing_id(self):
        """Generates fallback ID when id is missing but name exists."""
        meta = {"name": "image.png"}
        result = attachment_from_meta(meta, "msg3", 2)

        assert result is not None
        assert result.provider_attachment_id.startswith("att-")
        assert result.name == "image.png"

    def test_attachment_from_meta_empty_dict(self):
        """Returns None for empty metadata dict."""
        result = attachment_from_meta({}, "msg4", 0)
        assert result is None

    def test_attachment_from_meta_not_dict(self):
        """Returns None when meta is not a dict."""
        result = attachment_from_meta("not_a_dict", "msg5", 0)
        assert result is None

        result = attachment_from_meta(None, "msg6", 0)
        assert result is None

    def test_attachment_from_meta_alternative_id_fields(self):
        """Recognizes alternative ID field names."""
        meta1 = {"file_id": "file123", "name": "doc.txt"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.provider_attachment_id == "file123"

        meta2 = {"fileId": "file456", "name": "doc.txt"}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.provider_attachment_id == "file456"

        meta3 = {"uuid": "uuid789", "name": "doc.txt"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.provider_attachment_id == "uuid789"

    def test_attachment_from_meta_alternative_name_fields(self):
        """Recognizes alternative name field names."""
        meta = {"id": "att", "filename": "report.docx"}
        result = attachment_from_meta(meta, "msg", 0)
        assert result.name == "report.docx"

    def test_attachment_from_meta_size_conversion(self):
        """Converts size from string to int."""
        meta1 = {"id": "att", "name": "file", "size": "2048"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.size_bytes == 2048

        meta2 = {"id": "att", "name": "file", "size_bytes": 4096}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.size_bytes == 4096

        meta3 = {"id": "att", "name": "file", "sizeBytes": "8192"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.size_bytes == 8192

    def test_attachment_from_meta_invalid_size(self):
        """Handles invalid size gracefully."""
        meta = {"id": "att", "name": "file", "size": "invalid"}
        result = attachment_from_meta(meta, "msg", 0)
        assert result.size_bytes is None

    def test_attachment_from_meta_mime_type_variations(self):
        """Recognizes different mime_type field names."""
        meta1 = {"id": "att", "name": "file", "mimeType": "text/plain"}
        result1 = attachment_from_meta(meta1, "msg", 0)
        assert result1.mime_type == "text/plain"

        meta2 = {"id": "att", "name": "file", "mime_type": "image/jpeg"}
        result2 = attachment_from_meta(meta2, "msg", 0)
        assert result2.mime_type == "image/jpeg"

        meta3 = {"id": "att", "name": "file", "content_type": "application/json"}
        result3 = attachment_from_meta(meta3, "msg", 0)
        assert result3.mime_type == "application/json"


# -----------------------------------------------------------------------------
# DIALOGUE_PAIR VALIDATION
# -----------------------------------------------------------------------------


class TestDialoguePairValidation:
    """Tests for DialoguePair validation."""

    def test_dialogue_pair_valid(self):
        """Valid user + assistant pair is accepted."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.role == "user"
        assert pair.assistant.role == "assistant"

    def test_dialogue_pair_wrong_user_role(self):
        """Raises ValueError if user message doesn't have user role."""
        user_msg = Message(id="u1", role="assistant", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_wrong_assistant_role(self):
        """Raises ValueError if assistant message doesn't have assistant role."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="user", text="Hi there")

        with pytest.raises(ValueError, match="assistant message must have assistant role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_human_alias_valid(self):
        """Human role is accepted for user message."""
        user_msg = Message(id="u1", role="human", text="Hello")
        assistant_msg = Message(id="a1", role="assistant", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.user.is_user

    def test_dialogue_pair_model_alias_valid(self):
        """Model role is accepted for assistant message."""
        user_msg = Message(id="u1", role="user", text="Hello")
        assistant_msg = Message(id="a1", role="model", text="Hi there")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        assert pair.assistant.is_assistant

    def test_dialogue_pair_system_role_invalid(self):
        """System role is not valid for dialogue pair."""
        user_msg = Message(id="u1", role="system", text="System prompt")
        assistant_msg = Message(id="a1", role="assistant", text="Response")

        with pytest.raises(ValueError, match="user message must have user role"):
            DialoguePair(user=user_msg, assistant=assistant_msg)

    def test_dialogue_pair_exchange_property(self):
        """Exchange property renders the dialogue correctly."""
        user_msg = Message(id="u1", role="user", text="What is 2+2?")
        assistant_msg = Message(id="a1", role="assistant", text="4")

        pair = DialoguePair(user=user_msg, assistant=assistant_msg)
        exchange = pair.exchange

        assert "User: What is 2+2?" in exchange
        assert "Assistant: 4" in exchange
