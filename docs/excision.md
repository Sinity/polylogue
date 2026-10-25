# Excision

`polylogue excise TARGET` removes content on purpose: a session, a document,
an input, or every stored occurrence of a text (a leaked secret). It is a rare
operator action. `--dry-run` reports what would go and changes nothing.

## What each scope removes

- **Session:** its rows from every `index.db` table, its texts from search
  and `embeddings.db` when no other row holds them, and its bytes. An input
  that held only this session is removed whole. In an input shared with other
  sessions, the byte ranges of its records are overwritten with spaces of the
  same length, line breaks kept, so every other record keeps its offsets. In a
  JSON export the removed records' array separators go too. In a whole
  document with no record ranges, the session's own texts are blanked where
  they stand (`inputs_kept` and `texts_left` report what could not be).
- **Document:** its rows, the byte ranges that held it, and notes about it.
- **Input:** the input whole. Sessions that also came from other inputs are
  rebuilt by parsing those inputs again.
- **Text:** every stored occurrence, in every revision chain, as written and
  as JSON escapes it, overwritten with spaces of the same length. Notes,
  variants, kept results, deliveries and excision reasons holding it are
  redacted in place. The rows parsed from the changed inputs go, and the
  inputs are parsed again from their overwritten bytes (`inputs_reparsed`),
  so no column list decides what is cleaned.

Notes about a removed session go, and delivered briefs that included its
texts are blanked. The `excision` row keeps the removed session and message
IDs, locations and hashes (never the content); the writer refuses those rows
and the loop stops collecting those locations from then on.

Bytes read from a ZIP were also stored as the original ZIP. Removing them
removes that ZIP input and its blob (`zips_removed`); the ZIP's other members
stay as their own inputs.

## What refuses

The design overwrites bytes in place and parses again. Anything it cannot do
that way refuses with `unsupported_excision`, naming the input and format,
before anything changes:

- a text inside a ZIP container, SQLite page image, PDF or gzip stream, a
  non-UTF-8 representation, a delta-encoded blob, or an input stored as an
  opaque file;
- a text matching a JSON object key;
- a session whose records cut through scalar values or share bytes with an
  input outside the excision, or overlap a surviving session's records;
- a ZIP whose bytes another kept input or a shown attachment also uses.

Provider files are never edited. `on_disk` lists the locations whose stored
history held a text match; clean those files by hand, or a later full read
collects the text again.

## The order of the work

An excision runs in the writer twice. The first run is rolled back; it only
writes each new blob file the excision needs (the rewritten bytes) to
`blobs/tmp/`. Their blob rows are then committed as `deleting` and the files
moved under their content names, so a stop from here on leaves only files
that their rows mark for deletion. The second run does the work, one
transaction per database, and marks the new blobs it uses `present`:

1. **Embeddings first.** Vectors and provider failures of the removed texts
   are deleted and committed. If anything later fails, nothing else changed
   and the cost is buying those vectors again.
2. **The archive.** The `excision` row, the rewritten or removed bytes, the
   redactions and `excised` observations commit together. Blobs nothing
   refers to any more are marked `deleting`. From here the excision is in
   force.
3. **The index.** Its rows go or are cleared for parsing again, and its meta
   records how many excisions it holds.
4. **Files.** Blob files marked `deleting` are deleted, then their rows.

A stop between steps needs no recovery code. Reopening the archive clears
`blobs/tmp/` and deletes the files of blobs still marked `deleting`, and an
index that holds fewer excisions than the archive is deleted and rebuilt from
the already excised stored inputs. While an index is behind, the embeddings
writer refuses to store vectors, and it never stores a vector for a text the
index no longer holds. A failed commit stops the writer until the archive is
reopened.

The text scan keeps its matches, frames and the decoded blobs it reads in the
operation's memory and private scratch files, released when it ends. The
largest JSON record, the selected IDs and the reports must fit in memory.
