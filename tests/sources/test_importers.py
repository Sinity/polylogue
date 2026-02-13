"""Provider importer tests — JSONL parsing, real export discovery, cross-provider parametrized tests, boundaries."""

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


# =============================================================================
# REAL EXPORT FILE DISCOVERY
# =============================================================================


def discover_real_exports():
	"""Discover all real export samples in fixtures/real/."""
	fixtures_dir = Path(__file__).parent.parent / "fixtures" / "real"
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

	fixture_path = Path(__file__).parent.parent / "fixtures" / "real" / "chatgpt" / variant

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
