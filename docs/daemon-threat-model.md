# Daemon Threat Model

The Polylogue daemon (`polylogued`) is a local-first HTTP server that
serves the archive read API and ingests from the local filesystem.

## Trust Boundary

The daemon binds to `127.0.0.1` by default. Non-loopback bind requires
`--insecure-allow-remote` *and* `--api-auth-token`; the daemon refuses
to start otherwise (`daemon/cli.py:309-319`).

Authentication: a bearer token is required by default (`--api-auth-token`,
or `daemon.api.auth_token` / `POLYLOGUE_API_AUTH_TOKEN`). If none is
explicitly configured, the daemon auto-mints one on first start and persists
it to a `0600` file under the config root (`polylogue.paths.api_auth_token_path`,
polylogue-rzve); every first-party CLI/MCP client that talks to the daemon
resolves the same explicit-or-auto-minted token
(`polylogue.daemon.api_auth.resolve_api_auth_token`) so this is transparent
for the common single-user case. Machine clients present the bearer; the
first-party shell uses a separate, short-lived, scoped HttpOnly cookie minted
only for the exact daemon origin. That cookie is limited to reads, events,
and user-overlay writes; reset, ingest, and maintenance control routes accept
only the machine bearer when configured. The fully-open posture (no token at
all) now requires an explicit, loudly-logged opt-out
(`--api-allow-no-auth` / `POLYLOGUE_API_ALLOW_NO_AUTH=1`); it is no longer
the unconfigured default.

**Cross-uid boundary (polylogue-n6pz, closed):** loopback TCP has no
peer-uid check, so before auto-minting existed, "the API is open on
loopback" meant any local uid -- not just the archive's owner -- could read
full session content over `127.0.0.1:8766` even though the archive's SQLite
files themselves are `0700` and deny every other uid. The auto-minted,
`0600`, owner-verified token (`polylogue.daemon.api_auth._is_trusted_token_file`
checks uid + permission bits before trusting an existing token file, refusing
to follow a symlink or trust a file it does not exclusively own) restores
that boundary: a different uid can still reach the loopback port, but cannot
present a valid bearer, so it is rejected before any route runs. Residual
risk is unchanged from before: any process running as the *same* uid as the
daemon can always read the token file (and the archive's SQLite files
directly), which this design does not attempt to prevent -- same-uid
co-tenancy is out of scope, same as for any other localhost dev server or
database that trusts its own uid.

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
- **Risk**: Any process on the machine can `curl http://127.0.0.1:8766/api/...`.
- **Mitigation**: Loopback binding limits exposure to the local machine, and the auto-minted, `0600`, owner-verified bearer token (see Authentication above) means a process running as a *different* uid gets a `401` with no valid credential available to it -- it cannot read the token file the daemon itself trusts. This is stronger than "the same trust model as `localhost` databases": most dev-server defaults do not restore the OS's own cross-uid file-permission boundary the way this token does.
- **Residual**: Processes running as the *same* user can always obtain the token (it is a plain file they have read access to) and read the SQLite archive directly anyway. Same-uid co-tenancy is not a boundary this design defends.

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
configured. MCP write operations remain gated by the server's
`mcp_write_enabled` config opt-in (`polylogue.toml` `[mcp]` or
`POLYLOGUE_MCP_WRITE_ENABLED`).

The MCP server has no role ladder (polylogue-800m); each privileged dispatcher
is its own independent config opt-in, off by default:

| Capability | Enables |
|------------|---------|
| _(none; default)_ | Query, read, get, explain, context, status — all safe read operations |
| `mcp_write_enabled` | `write`/`run`: tag management, metadata mutations, session deletion, saved-query/recipe execution |
| `mcp_judge_enabled` | `judge`: assertion-candidate judgment |
| `mcp_maintenance_enabled` | `maintenance`: maintenance operations, index rebuilds, insight refresh |

## Future Considerations

- **Unix socket**: Could replace loopback TCP for stronger access control (file permissions on the socket).
- **Read-only mode**: Could open the SQLite database in read-only mode for the HTTP API, with a separate write connection for ingest.
- **Secrets in sessions**: API keys and tokens that appear in session text are stored as-is. A future redaction layer could strip these.
