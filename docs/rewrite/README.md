# Implementation notes

Notes on how parts of the Rust Polylogue work, beyond what the README covers.
The full design is in the spec repository (`/realm/inbox/spec/polylogue-spec/`).

| Note | Covers |
|---|---|
| [observed-source.md](observed-source.md) | collecting one configured Claude Code location |
| [codex-configured-discovery.md](codex-configured-discovery.md) | finding and following a configured Codex home |
| [codex-native-representations.md](codex-native-representations.md) | importing Codex rollouts and thread-history databases |
| [python-port.md](python-port.md) | Claude and Codex conversation features ported from the Python version |
| [provider-installation.md](provider-installation.md) | how parsed provider state is written in batches |
| [message-preparation.md](message-preparation.md) | building messages from parsed records |
| [read-connections.md](read-connections.md) | read connections and prepared statements |
| [prose-search-scratch.md](prose-search-scratch.md) | per-query full-text search tables |

The SQL schema is in `crates/store/schema/`; `provenance.json` records which
spec revision each file was taken from.
