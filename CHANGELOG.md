# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-06-01

### Added

- Multi-provider conversation archive: ChatGPT, Claude, Claude Code, Codex, and Gemini (via Google Drive)
- Unified conversation model with semantic projections (`substantive_only()`, `iter_pairs()`, `to_text()`)
- SQLite storage backend with FTS5 full-text search and sqlite-vec vector search
- Async SQLite backend for concurrent read operations
- Pipeline architecture: acquire → parse → render → index stages
- HTML and Markdown rendering with branch-aware tree visualization
- Google Drive integration for Gemini AI Studio conversations
- Model Context Protocol (MCP) server for AI assistant integration
- Fluent `ConversationFilter` API for programmatic queries
- Interactive TUI dashboard (Textual-based) and plain CLI mode
- `Polylogue` facade class for library usage
- Comprehensive test suite with Hypothesis property-based tests
- Provider-specific JSON schema inference from real data
- Attachment tracking with ref-counted deduplication
- Event bus for decoupled CLI/pipeline communication

### Infrastructure

- Zero-config XDG-compliant path resolution
- Nix flake for reproducible development and packaging
- CI with multi-Python matrix (3.10, 3.12, 3.13), type checking, and linting
- Pre-commit hooks with ruff
