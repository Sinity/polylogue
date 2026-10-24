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
fetching URLs. Whole-input excision removes an independent asset Input and blocks reimport. Text excision of an opaque companion refuses until its format
has a decoder. Malformed naming metadata is reported at import and raw sidecars remain
retained. Native file-ID matching can still resolve bytes without a filename.

ZIP import retains the exact original container, including its metadata, and
separate uncompressed file-member Inputs. Member observations record their
original container and native member index. Equal bytes share storage while
keeping every physical observation. Archive10 stores this relationship; older
writers and row decoders refuse it. Older member-only captures do not acquire
missing original bytes during rebuild.

A ZIP's file members are decoded, hashed and staged privately before any bytes
are published. A refused member refuses the linked acquisition. Corrupt,
encrypted or unreadable member populations refuse before public installation;
the original does not fall back to opaque retention. After reservation, an I/O
failure can leave accountable unfinished bytes, reported as unavailable until
verified or supplied by an exact retry.

Removing a member, or rewriting its records for session/document removal,
retires every containing original ZIP representation, including nested containers.
Unrelated decoded member Inputs survive. Removing only an original ZIP preserves
its independently retained members. `containers_retired` identifies originals
removed (or that would be removed in a dry run); their raw metadata and exact
container bytes become unavailable. Polylogue never substitutes a regenerated
ZIP for the original. Acquisition observations remain; retired container links
are removed through the existing excision publication.

If another retained raw Input, decoded association, attachment or delta depends
on the original bytes outside the removal set, the operation refuses before
mutation. Direct attachment or delta-base references to original bytes currently refuse
even when their owners are also selected. Selective ZIP regeneration and text
removal from opaque member formats remain unsupported. A selected immutable member Input is shared by content across
its acquisitions, so every containing original for that Input is retired; this
does not extend removal to unrelated location-owned Inputs. Standing refusal
blocks recapturing the selected member and retired originals.

The original container is material-only data, so its path can appear in both
`imported` and `unrecognized`.

Staging holds metadata for all entries and private compressed stages for all
member bodies. The complete acquisition uses a synchronous Writer turn; this
does not promise constant memory, fixed disk use or fair scheduling. Parsing
continues from retained member bytes afterward. This workflow covers ChatGPT
native file-ID matching and naming/library sidecars; other provider and
filename-only matching remain open.
