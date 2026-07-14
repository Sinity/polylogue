# Security Policy

Polylogue is a **local-first, single-user archive**. It assumes a trusted
single-user local host: the daemon binds to loopback (`127.0.0.1`) by
default, protected routes require a bearer token or a same-origin web
credential, browser capture uses a separate token, and mutating
browser-accessible routes enforce exact `Origin` matching. It is not designed
for multi-user isolation, and encryption at rest is the operating system's
responsibility (raw archives can contain source code, secrets, personal
conversations, file paths, and tool output — use host disk encryption).

The full threat model — trust boundaries, actors, assets, per-endpoint attack
surface, mitigations, explicit non-threats, and the reasoning behind each — is
documented at [`docs/security.md`](docs/security.md) and
[`docs/daemon-threat-model.md`](docs/daemon-threat-model.md). Read those
before assuming a given behavior is a bug rather than a documented boundary.

## Supported Versions

Polylogue is pre-1.0 and moves quickly. Security fixes land on `master` and
the latest published release; there is no separate long-term-support branch.

## Reporting a Vulnerability

Please use GitHub's private
[vulnerability reporting](https://github.com/Sinity/polylogue/security/advisories/new)
for the repository rather than a public issue, so a fix can land before the
report is public. If that is not available to you, open a minimal-detail
public issue asking for a private contact channel rather than describing the
vulnerability inline.

Please include:

- the affected component (CLI, daemon HTTP API, browser-capture receiver,
  MCP server, a specific parser);
- reproduction steps or a minimal example;
- the trust boundary crossed (see the actor/asset tables in
  `docs/security.md`) — this is usually the fastest way to tell a real
  security issue apart from expected same-user-trusted behavior.

There is no bug-bounty program. Reports are read and triaged by the
maintainer; expect an acknowledgment before a fix timeline.
