# Export attachments

Import a ChatGPT export directory or ZIP, then describe an attachment by its
full message ID and zero-based block index:

```sh
polylogue import /path/to/export.zip
polylogue attachment chatgpt:MESSAGE-ID 0
```

For an attachment inside a tool result, add `--attachment N` (default `0`).
The answer describes the attachment as the conversation recorded it and lists
the stored export files that match it in `matches`, each with its size, names,
blob and `source_version`. Recorded fields stay apart from matched filenames
and hashes. Read pages describe only the attachments whose labels appear on
that page.

`state` is `available` for one match, `ambiguous` for several, `unavailable`
when the import holds no matching file, and `unknown` when the attachment's
identity or the import it came from is not recorded. It says what the archive
holds, not whether the provider still has the file. Different files stay
separate matches even when the conversations are identical.

## Writing the bytes

CLI, MCP and HTTP answers are masked. The exact bytes can contain
credentials, so only the CLI writes them, with terminal stdout and outside an
agent harness:

```sh
polylogue attachment chatgpt:MESSAGE-ID 0 --reveal --out attachment.bin
```

For an `ambiguous` attachment, add `--asset INPUT-ID` from `matches`.
`--length BYTES` sets how many bytes each read takes (default 65536). The
bytes go to a private temporary file, which replaces FILE only once every
range, the length and the BLAKE3 hash agree. If the stored bytes change
meanwhile, or anything fails, FILE is left as it was.

## How matches are found

An import lists paths as `imported`, `already_stored` and `unrecognized`. A
file is stored even when no parser recognizes it, so a path can be in both
`imported` and `unrecognized`; a ZIP's own path is, since only the files in
it are parsed. Each import records the format chosen for the
file beside its path, and a rebuild uses that pair. Parsing the same stored
file as another origin later refuses.

Parsing and reading use only stored bytes; provider files are not reopened
and URLs are not fetched. ChatGPT matches use the native file ID and the
export's naming and library sidecars; a file ID can match without a name.
Malformed naming metadata is reported at import and the sidecar stays stored.
Matching by filename alone, and other providers, are not built.

A ZIP import stores the ZIP exactly as given as one input and each file in it
as its own input. Each file's observation records its ZIP and position in
`observation_container`. Identical bytes are stored once with every
observation kept. The files are decoded, hashed and compressed to private
scratch files before the writer stores the whole ZIP in one turn; parsing
then reads the stored files. A corrupt, encrypted or unreadable file refuses
the whole ZIP before anything is stored. A stop after the rows are written
lists the bytes in `pending_inputs` until the same bytes are imported again.

## Excision

Excising a whole input removes it and refuses it on later imports. Nothing
inside an export ZIP can be excised: the ZIP, a file read from it, and a
session or document parsed from such a file all refuse before changing
anything, because the kept ZIP also holds unrelated conversations
([excision.md](excision.md)).
