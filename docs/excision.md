# Excision

`polylogue excise TARGET` removes stored files on purpose, with everything
parsed from them. It is a rare operator action. Without `--yes` it reports
what would go and changes nothing.

An excision removes whole stored inputs and never rewrites stored bytes.
Anything it could only remove by rewriting a file that also holds other
content refuses with `unsupported_excision` before anything changes.

## What can be excised

- **`session:<id>`:** every input the session came from, when those inputs
  hold nothing but this session: their stored bytes, the session's rows in
  every `index.db` table, its texts in search and `embeddings.db` when no
  other row holds them, and the notes about it. A note left with no target
  goes whole, with its request and the withdrawals of it.
- **`document:<id>`:** every input the document came from, when those inputs
  hold nothing but this document, with its rows and notes.
- **`input:<id>`:** the input, with the sessions and documents it holds, when
  none of them also comes from another input.

The `excised` rows keep what was removed by identity (session, message and
document IDs, inputs, blob hashes and locations), never the content. The
writer never stores those rows again, refuses those inputs and bytes on later
reads, and the loop stops collecting those locations. Provider files are
never edited.

## What cannot be excised

Each refuses, naming the input or record that stops it:

- anything inside an export ZIP: the ZIP is kept as imported and holds
  unrelated conversations (the message names the ZIP file);
- a session or document whose input also holds another session or document,
  such as a JSON lines file with records of two sessions or a
  multi-conversation export; excise the whole input instead;
- an input holding a session or document that also comes from another
  input; excise that session instead;
- a session a delivered brief names, since the brief holds its text;
- a session whose note would go while a revision of that note stays, since
  the revision's request could no longer be replayed; withdraw or excise the
  revision first;
- a text inside messages (`text:`).

## The order of the work

The writer runs an excision in one pass. It plans, reading only, and refuses
there. Then one transaction per database commits embeddings, then the
archive, then the index: the vectors of removed texts go first (a later
failure only costs buying them again); once the archive commits the
excision is in force. Blobs nothing refers to are marked `deleting`, and the
index's meta records how many excisions it holds. With `secure_delete` on,
freed pages are zeroed; afterwards the search indexes are merged and both
write-ahead logs are checkpointed, so no removed text stays readable in the
database files, and the files of blobs marked `deleting` are deleted.

A stop needs no recovery code. Reopening the archive deletes the files of
blobs still marked `deleting`, and an index that holds fewer excisions than
the archive is rebuilt from the stored inputs that remain. While an index is
behind, the embeddings writer refuses to store vectors. A failed commit
stops the writer until the archive is reopened. Running the same request
again reports what the recorded excision removed and finishes what is left.
A dry run plans and rolls the transactions back.
