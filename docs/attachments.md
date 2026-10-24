# Retained export attachments

Import a ChatGPT export directory or ZIP, then inspect an attachment by its
full message ID and zero-based stored block index:

```sh
polylogue import /path/to/export.zip
polylogue attachment chatgpt:MESSAGE-ID 0
```

For attachments within a tool result, add `--attachment N` (default `0`). The
response describes the recorded attachment and its retained candidates. It
keeps the source fields separate from resolved filenames and byte hashes.
Read pages describe only attachments whose labels appear on that page.

`available` means one current retained candidate, `ambiguous` means several,
`unavailable` means the observed import scope has no current retained candidate
for that reference, and `unknown` means its identity or acquisition scope is
missing. These states describe retention, not whether a provider still has a
file. Conflicting assets remain separate candidates even when conversation bytes match.

Ordinary CLI, MCP and HTTP reads return masked descriptions. Exact bytes can
contain credentials. To write them explicitly, run the interactive CLI with
terminal stdout outside an agent harness:

```sh
polylogue attachment chatgpt:MESSAGE-ID 0 --reveal --out attachment.bin
```

For an ambiguous result, add `--asset INPUT-ID` from its candidate list.
`--length BYTES` changes the retrieval range size (default 65536). These
retrieval options are CLI-only. Output goes to a private temporary file and
replaces the destination only after all ranges, length and BLAKE3 agree.
Changed source or association state refuses continuation; a failure before installation
leaves an existing destination untouched.

Import results separate byte retention (`imported` / `already_stored`) from
interpretation (`unrecognized`), so those lists can overlap. Each new acquisition
records its selected format beside its captured path. Rebuild reuses that pair
without changing the original Input format. For legacy acquisitions without
format hints, current recognition uses retained bytes and recorded paths;
`legacy_imports` reports its current candidates. Unknown or competing origins
remain uninterpreted, and this result does not become a historical source claim.
A later opaque acquisition keeps earlier recognized context. Switching between
competing parsed origins for one immutable input is unsupported. Text excision
still refuses Inputs originally captured as opaque, including ones interpreted
later; it does not infer a safe decoder from a later format hint.

Parsing and reading use retained bytes, without reopening provider files or
fetching URLs. Whole-input excision removes an asset's retained Input and
blocks reimport. Text excision of an opaque companion refuses until its format
has a decoder. Malformed naming metadata is reported at import and raw sidecars remain
retained. Native file-ID matching can still resolve bytes without a filename.

Current ZIP import retains exact uncompressed member bytes and acquisition /
member observations, but not the original compressed ZIP container or its
metadata. Original-container custody and coordinated container/member excision
remain open in `plnext-185.12`. This workflow covers ChatGPT native file-ID
matching and naming/library sidecars; other provider and filename-only matching
rules remain outside this delivery.
