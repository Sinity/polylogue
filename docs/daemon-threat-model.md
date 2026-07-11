# Daemon Threat Model

The Polylogue daemon (`polylogued`) is a local-first HTTP server that
serves the archive read API and ingests from the local filesystem.

## Trust Boundary

The daemon binds to `127.0.0.1` by default. Non-loopback bind requires
`--insecure-allow-remote` *and* `--api-auth-token`; the daemon refuses
to start otherwise (`daemon/cli.py:309-319`).

Authentication: optional bearer token (`--api-auth-token`). When set,
machine clients present the bearer; the first-party shell uses a separate,
short-lived, scoped HttpOnly cookie minted only for the exact daemon origin.
That cookie is limited to reads, events, and user-overlay writes; reset, ingest,
and maintenance control routes accept only the machine bearer when configured.
When unset, the API is open on loopback; the loopback bind is the access
boundary in that mode.

Cross-origin POSTs (CSRF) are refused by exact `Origin`-to-`Host` authority
matching. Another loopback port is not trusted. First-party credentials are
also origin-bound and expose explicit missing/invalid/expired/revoked/wrong-origin
states without putting secret bytes in URLs or response bodies.

For the full security policy and explicit decisions on raw-artifact
redaction, `/api/sources` paths, and `OPTIONS` handling, see
[`docs/security.md`](security.md).

## Assets

| Asset | Sensitivity | Exposure |
|-------|------------|----------|
| Session content (messages, titles, timestamps) | High — personal AI chat history | Read API |
| User metadata (tags, summaries, notes) | Medium — user-curated | Read + Write API |
| Session identity (provider, dates, durations) | Low — operational metadata | Read API |
| Raw artifacts (JSONL payloads) | High — contains full session data | Filesystem only |

## Threats

### Local process reading the API
- **Risk**: Any process on the machine can `curl http://127.0.0.1:8765/api/...`
- **Mitigation**: Loopback binding limits exposure to the local machine. This is the same trust model as `localhost` databases, dev servers, and CLI tools.
- **Residual**: Processes running as the same user can read the SQLite archive directly anyway.

### Extension posting forged captures
- **Risk**: A malicious browser extension or page could POST to the receiver
- **Mitigation**: Receiver validates envelope shape and source origin. Only supported provider DOM adapters produce valid captures.
- **Residual**: Local browser extensions have the same trust as the user's browser profile.

### Daemon process compromise
- **Risk**: If the daemon binary or its dependencies are compromised
- **Mitigation**: Nix-supplied dependencies with known hashes. No dynamic code loading.
- **Residual**: Supply-chain risk is inherited from the Nix/NixOS package closure.

### Archive file tampering
- **Risk**: Another process modifies the SQLite database or blob store
- **Mitigation**: SQLite WAL mode provides crash safety, not access control. File permissions are the boundary.
- **Residual**: Same-user processes can modify the archive. This is inherent to local-first tools.

## Non-Threats

These are explicitly out of scope for the daemon threat model:

- **Multi-user access**: Polylogue is a single-user tool. There is no user isolation.
- **Network exposure**: The daemon does not bind to `0.0.0.0` or non-loopback interfaces.
- **Encryption at rest**: Archive content is stored as plaintext SQLite. Disk encryption is the OS's responsibility.
- **Multi-user authentication**: No user accounts, RBAC, or per-user tokens. The configured machine bearer and its scoped first-party browser adapter are same-user access controls, not user identities.

## API Roles

The daemon HTTP API exposes archive reads plus explicit user-overlay writes.
Those overlay writes require either the machine bearer or the scoped
first-party cookie and an exact-origin request. Archive reset, ingest, and
maintenance controls are separate machine-bearer capabilities when auth is
configured. MCP write operations remain gated by the server's `--role write`
flag.

The MCP server has three roles:

| Role | Capabilities |
|------|-------------|
| `read` | Query, search, list, get, stats, insights — all safe operations |
| `write` | Tag management, metadata mutations, session deletion |
| `admin` | Maintenance operations, index rebuilds, insight refresh |

## Future Considerations

- **Unix socket**: Could replace loopback TCP for stronger access control (file permissions on the socket).
- **Read-only mode**: Could open the SQLite database in read-only mode for the HTTP API, with a separate write connection for ingest.
- **Secrets in sessions**: API keys and tokens that appear in session text are stored as-is. A future redaction layer could strip these.
