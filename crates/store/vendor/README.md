`cksumvfs.c` is unchanged upstream public-domain SQLite source from
https://github.com/sqlite/sqlite/blob/version-3.53.2/ext/misc/cksumvfs.c .
It matches libsqlite3-sys 0.38.2 (rusqlite 0.40.2), SQLite 3.53.2.
`bridge.c` checks that version at compile time and reserves eight bytes before
new durable databases receive their first write. Registration occurs once,
before store opens any SQLite connection. Durable connections use `cksmvfs`;
the recomputable index explicitly uses `unix` without checksum wrapping.
