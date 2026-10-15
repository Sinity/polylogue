`cksumvfs.c` is upstream public-domain SQLite source from
https://github.com/sqlite/sqlite/blob/version-3.53.2/ext/misc/cksumvfs.c .
The sole local change calls `polylogue_checksum_fault` at an actual checksum
comparison failure, preserving the upstream log and return code. The Rust store
copies the filename and aligned page offset into thread-local, operation-scoped
evidence. Only a matching `SQLITE_IOERR_DATA` becomes a typed `CorruptPage`;
other errors never receive an inferred page number. This wraps main-database
page reads; it does not add WAL checksum verification or recovery.

It matches libsqlite3-sys 0.38.2 (rusqlite 0.40.2), SQLite 3.53.2.
`bridge.c` checks that version at compile time and reserves eight bytes before
new durable databases receive their first write. Registration occurs once,
before store opens any SQLite connection. Durable connections use `cksmvfs`;
the recomputable index explicitly uses `unix` without checksum wrapping.
