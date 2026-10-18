# Streaming credential masking

`crates/credential-output` masks the 22 credential families of the shared `credential-output-v1` catalog in streamed UTF-8 text, through `Masker::new`, `push`, `block_boundary` and `finish`. It changes display bytes only; stored content is untouched.

A credential becomes `[credential: KIND, masked]`, with no preview. Labels such as `--secret=` or `Authorization: Basic` stay visible. URL masking keeps the scheme, the last `@` before the host, the host and the path. PEM masking covers any uppercase label ending in `PRIVATE KEY`, through the matching END line or the end of the string. Overlapping matches merge into one marker; adjacent matches keep separate markers.

`push` pieces never end a match. `block_boundary` ends token matches and writes unfinished prefixes as plain text, but an open PEM block stays masked. `finish` ends the string. Masking never refuses input; the only errors come from the sink.

Text that may still become a credential is held back, at most 512 UTF-8 bytes. When an undecided match would hold more, it is forced: what it has read is masked (from its secret part if that has begun, otherwise from its start), and it keeps masking each character it reads until it ends. A credential longer than the window, such as a JWT with a large payload or a URL with a long password, is masked whole; a forced URL also masks its host, and a forced match that turns out not to be a credential stays masked. A PEM label longer than 512 bytes masks to the end of the string, since its END line is not kept. Memory stays bounded: forced matches that a longer forced match already covers are dropped.

Left word boundaries use a Unicode 16.0.0 table of Python `str.isalnum` plus underscore (`src/word_ranges.rs`). Whitespace includes U+001C to U+001F, as in Python. Update the table and the boundary tests together.

A sink error is returned unchanged and closes the stream; later calls return `BrokenPipe` without writing. `finish` also closes the stream. The masker never calls `flush`.

The tests run all 132 catalog cases at every byte split and at chunk widths 1 to 8, plus overlap, block boundary, Unicode boundary, long-credential and sink-error cases, and randomized checks that output does not depend on chunking and that embedded credentials never appear in it. The catalog and its provenance are in `crates/credential-output/tests/fixtures/credential-output-v1`.
