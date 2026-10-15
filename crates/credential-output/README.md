# credential-output

The pure `Masker` implements the fixed `credential-output-v1` catalog using bounded streaming state. Feed UTF-8 fragments with `push`, settle logical block boundaries with `block_boundary`, and end a logical string with `finish`. An error fuses the stream; callers must discard the response rather than return partial success. The caller owns response budgets, structured keys and opaque identity refusal. Independent strings must use separate maskers.

The implementation and Unicode table were extracted unchanged from Polylogue. The independent 132-case corpus and its provenance live in `tests/fixtures/credential-output-v1`. No storage, transport or admission policy belongs here.

The owner follows [the existing shared release procedure](../../docs/shared-file-layer-release.md). This crate requires its own reviewed immutable `credential-output-v0.1.0` tag; `hashed-blobs-v0.1.0` is unrelated and must never move. Consumer qualification binds an exact owner commit until that release is published.
