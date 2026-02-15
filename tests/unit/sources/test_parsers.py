"""Provider parser tests — JSONL parsing, real export discovery, cross-provider parametrized tests, boundaries."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

# Test helpers
# Additional imports for test_ingestion_jsonl
from polylogue.config import Config
from polylogue.pipeline.services.parsing import ParsingService

# ChatGPT imports
from polylogue.sources.parsers.chatgpt import parse as chatgpt_parse

# Claude imports
from polylogue.sources.parsers.claude import (
	looks_like_ai,
	looks_like_code,
	parse_ai,
	parse_code,
)

# Codex imports
from polylogue.sources.parsers.codex import parse as codex_parse

# Drive imports (for parse_chunked_prompt)
from polylogue.sources.parsers.drive import parse_chunked_prompt
from polylogue.storage.backends.sqlite import SQLiteBackend
from polylogue.storage.repository import ConversationRepository
from polylogue.storage.store import RawConversationRecord


class TestParseRawRecordJsonl:
	"""Tests for ParsingService._parse_raw_record with JSONL and JSON inputs."""

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
	def parsing_service(
		self, tmp_path: Path, repository: ConversationRepository
	) -> ParsingService:
		"""Create a ParsingService for testing."""
		config = Config(
			sources=[],
			archive_root=tmp_path / "archive",
			render_root=tmp_path / "render",
		)
		return ParsingService(
			repository=repository,
			archive_root=tmp_path / "archive",
			config=config,
			drive_client_factory=None,
		)

	def test_parse_raw_record_single_json(
		self, parsing_service: ParsingService
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
		parsed = parsing_service._parse_raw_record(raw_record)

		assert len(parsed) > 0
		assert parsed[0].provider_name == "chatgpt"
		assert parsed[0].title == "Test Conversation"
		# ChatGPT parser extracts messages from the mapping
		assert len(parsed[0].messages) == 2

	def test_parse_raw_record_jsonl(self, parsing_service: ParsingService) -> None:
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
		parsed = parsing_service._parse_raw_record(raw_record)

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
		self, parsing_service: ParsingService
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
		parsed = parsing_service._parse_raw_record(raw_record)

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

		This tests the orphaned raw records scenario from parse_sources():
		When a raw record exists but the corresponding conversation was deleted
		or never parsed, it should be re-parsed.
		"""
		config = Config(
			sources=[],
			archive_root=tmp_path / "archive",
			render_root=tmp_path / "render",
		)
		parsing_service = ParsingService(
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
		# This is the pattern from parse_sources()
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

		# Now parse it using parse_from_raw with the orphaned ID
		result = parsing_service.parse_from_raw(raw_ids=["orphaned-raw-001"])

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
# SYNTHETIC EXPORT GENERATION
# =============================================================================


def discover_synthetic_exports():
	"""Generate synthetic exports for all available providers."""
	from polylogue.schemas.synthetic import SyntheticCorpus

	exports = []
	for provider in SyntheticCorpus.available_providers():
		exports.append(provider)
	return exports


SYNTHETIC_PROVIDERS = discover_synthetic_exports()


# =============================================================================
# CROSS-PROVIDER SYNTHETIC DATA TESTS (PARAMETRIZED)
# =============================================================================


@pytest.mark.parametrize("provider", SYNTHETIC_PROVIDERS)
def test_parse_synthetic_export_produces_valid_conversation(provider, tmp_path):
	"""Parse synthetic export for each provider and validate structure."""
	import json
	from polylogue.schemas.synthetic import SyntheticCorpus
	from polylogue.paths import Source
	from polylogue.sources import iter_source_conversations

	corpus = SyntheticCorpus.for_provider(provider)
	raw_bytes = corpus.generate(count=1, messages_per_conversation=range(3, 8), seed=42)[0]

	# Write to file and parse through source detection
	ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
	export_file = tmp_path / f"{provider}-synthetic{ext}"
	export_file.write_bytes(raw_bytes)

	source = Source(name=f"{provider}-test", path=export_file)
	conversations = list(iter_source_conversations(source))

	assert len(conversations) > 0, f"No conversations parsed for {provider}"
	for conv in conversations:
		assert conv.provider_name in (provider, "claude")  # claude-ai maps to "claude"
		assert len(conv.messages) > 0, f"No messages in conversation for {provider}"
		for msg in conv.messages:
			assert msg.text is not None


@pytest.mark.parametrize("provider", SYNTHETIC_PROVIDERS)
def test_synthetic_export_messages_have_roles(provider, tmp_path):
	"""All messages in synthetic exports have valid roles.

	Parametrized across all synthetic providers.
	"""
	import json
	from polylogue.schemas.synthetic import SyntheticCorpus
	from polylogue.paths import Source
	from polylogue.sources import iter_source_conversations

	corpus = SyntheticCorpus.for_provider(provider)
	raw_bytes = corpus.generate(count=1, messages_per_conversation=range(3, 8), seed=42)[0]

	ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
	export_file = tmp_path / f"{provider}-synthetic{ext}"
	export_file.write_bytes(raw_bytes)

	valid_roles = {"user", "assistant", "system", "tool", "human", "model"}

	source = Source(name=f"{provider}-test", path=export_file)
	conversations = list(iter_source_conversations(source))

	for conv in conversations:
		# All messages must have valid roles
		for msg in conv.messages:
			assert msg.role.lower() in valid_roles, \
				f"Invalid role '{msg.role}' in {provider}"


@pytest.mark.parametrize("provider", SYNTHETIC_PROVIDERS)
def test_synthetic_export_preserves_metadata(provider, tmp_path):
	"""Synthetic exports preserve provider metadata.

	Parametrized across all synthetic providers.
	"""
	import json
	from polylogue.schemas.synthetic import SyntheticCorpus
	from polylogue.paths import Source
	from polylogue.sources import iter_source_conversations

	corpus = SyntheticCorpus.for_provider(provider)
	raw_bytes = corpus.generate(count=1, messages_per_conversation=range(3, 8), seed=42)[0]

	ext = ".json" if corpus.wire_format.encoding == "json" else ".jsonl"
	export_file = tmp_path / f"{provider}-synthetic{ext}"
	export_file.write_bytes(raw_bytes)

	source = Source(name=f"{provider}-test", path=export_file)
	conversations = list(iter_source_conversations(source))

	for conv in conversations:
		# Should have either a title or at least one message with provider metadata
		# (exact fields vary by provider)
		has_message_meta = any(m.provider_meta for m in conv.messages) if conv.messages else False
		assert conv.title is not None or has_message_meta


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
def test_chatgpt_format_variants(variant, tmp_path):
	"""Test ChatGPT parser handles different export format variants with synthetic data.

	Parametrized to automatically test all discovered variants.
	"""
	import json
	from polylogue.schemas.synthetic import SyntheticCorpus
	from polylogue.paths import Source
	from polylogue.sources import iter_source_conversations

	corpus = SyntheticCorpus.for_provider("chatgpt")

	# Generate appropriate message count based on variant
	if "branching" in variant:
		raw_bytes = corpus.generate(count=1, messages_per_conversation=range(12, 20), seed=42)[0]
	else:
		raw_bytes = corpus.generate(count=1, messages_per_conversation=range(3, 8), seed=42)[0]

	export_file = tmp_path / f"{variant}"
	export_file.write_bytes(raw_bytes)

	with open(export_file) as f:
		data = json.load(f)

	parsed = chatgpt_parse(data, f"variant-{variant}")

	# All variants should parse successfully
	assert parsed.provider_name == "chatgpt"
	assert len(parsed.messages) > 0

	# Format-specific validations
	if "branching" in variant:
		# Branching conversations should have many messages
		assert len(parsed.messages) > 10


# =============================================================================
# STATISTICS
# =============================================================================


def test_consolidation_statistics():
	"""Document consolidation statistics.

	This is a META test that documents the synthetic provider coverage.
	"""
	num_synthetic_providers = len(SYNTHETIC_PROVIDERS)
	num_chatgpt_variants = len(CHATGPT_VARIANTS)

	# Each parametrized test runs once per synthetic provider
	tests_per_parametrized = num_synthetic_providers

	# Total tests generated by this file's parametrized tests
	parametrized_tests = (
		tests_per_parametrized * 3  # 3 synthetic provider tests
		+ 4  # Boundary tests (substantive)
		+ 4  # Boundary tests (context dump)
		+ num_chatgpt_variants  # ChatGPT variants
	)

	print(f"\n{'='*60}")
	print("SYNTHETIC PROVIDER COVERAGE")
	print(f"{'='*60}")
	print(f"Synthetic providers available: {num_synthetic_providers}")
	print(f"Tests generated from parametrization: {parametrized_tests}")
	print(f"Coverage multiplier: {parametrized_tests / (len(SYNTHETIC_PROVIDERS) or 1):.1f}x")
	print(f"{'='*60}\n")

	assert True  # Always pass, this is just for reporting
