# Streaming credential masking

`crates/credential-output` masks the 22 credential families of the shared `credential-output-v1` catalog in streamed UTF-8 text, through `Masker::new`, `push`, `block_boundary` and `finish`. It changes display bytes only; stored content is untouched.

A credential becomes `[credential: KIND, masked]`, with no preview. Labels such as `--secret=` or `Authorization: Basic` stay visible. URL masking keeps the scheme, the last `@` before the host, the host and the path. PEM masking covers any uppercase label ending in `PRIVATE KEY`, through the matching END line or the end of the string. Overlapping matches merge into one marker; adjacent matches keep separate markers.

`push` pieces never end a match. `block_boundary` ends token matches and writes unfinished prefixes as plain text, but an open PEM block stays masked. `finish` ends the string. On any error the caller must discard the response.

Limits: at most 512 undecided UTF-8 bytes, 2,048 open token matches, 512 open URL matches and 512 open PEM headers of at most 512 bytes each. Input that would exceed a limit returns `io::ErrorKind::InvalidData` before the held text is written. Long secrets and PEM bodies do not count against the limits.

Left word boundaries use a Unicode 16.0.0 table of Python `str.isalnum` plus underscore (`src/word_ranges.rs`). Whitespace includes U+001C to U+001F, as in Python. Update the table and the boundary tests together.

A sink error is returned unchanged and closes the stream; later calls return `BrokenPipe` without writing. `finish` also closes the stream. The masker never calls `flush`.

The tests run all 132 catalog cases at every byte split and at chunk widths 1 to 8, plus overlap, block boundary, Unicode boundary, limit and sink-error cases. The catalog and its provenance are in `crates/credential-output/tests/fixtures/credential-output-v1`.
