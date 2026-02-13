# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public issue
2. Email the maintainers at security@polylogue.dev (or open a private security advisory on GitHub)
3. Include a description of the vulnerability, steps to reproduce, and potential impact
4. Allow up to 72 hours for an initial response

## Scope

Polylogue processes user-provided conversation exports and connects to Google Drive via OAuth.
Security-relevant areas include:

- **File parsing**: JSON/JSONL deserialization of untrusted exports
- **OAuth credentials**: Token storage under `$XDG_CONFIG_HOME/polylogue/`
- **SQLite operations**: Database queries constructed from user input
- **MCP server**: Model Context Protocol endpoints exposed to AI assistants

## Credential Handling

- OAuth tokens are stored locally under XDG config paths
- Tokens are never logged or included in error messages
- The `--preview` flag shows resolved paths without executing operations
