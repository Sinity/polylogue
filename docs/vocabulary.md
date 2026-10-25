# Vocabulary

One name per concept. Code, SQL, CLI help, JSON keys, errors, comments, tests
and docs use these words and no others for the same thing. A new concept gets
a plain name here before it gets code.

## Stored data

- **archive**: `archive.db` and `blobs/`, what cannot be recomputed. The
  connection to it is the archive connection.
- **index**: `index.db`, everything parsed and derived. Rebuilt from the
  archive when deleted or when `INDEX_VERSION` changes.
- **embeddings**: `embeddings.db`, purchased vectors, kept apart from the
  index because they cost money to make.
- **blob**: bytes stored by their BLAKE3 hash, inline or as a file under
  `blobs/`. Its row's state is `writing`, `present` or `deleting`; the file is
  written only after the row exists.
- **input**: one logical file: one location's history, or one imported file.
- **revision**: one distinct content of an input.
- **chain**: revisions of an input where each one appends to the one before.
- **snapshot revision**: a revision that holds a whole copy, such as a
  tool database's rows at one moment, instead of appending to a chain.
- **extent**: a range of a blob placed at an offset in a chain.
- **observation**: one time the collector found something at a path.
- **machine**: a computer, by `/etc/machine-id`; its host name is a label.
- **export ZIP** and **member**: an imported ZIP, stored whole, and one file
  inside it.
- **extracted blob**: bytes decoded from inside a record, such as inline
  media, stored with the record's byte range.
- **write counter** (`commit_seq`): incremented once per archive transaction;
  orders the change feed and row batches.
- **stored**: kept in the archive. Prefer no word at all: an input is stored
  by definition.

## Reading sources

- **origin**: the tool or export a file comes from (`claude-code`,
  `chatgpt`, ...). Each has one **origin declaration** (`OriginSpec`) holding
  everything Polylogue knows about it.
- **format**: which parser reads an input.
- **parser**: turns an input's bytes into sessions, messages, documents and
  events. Its output is **parsed rows**.
- **record**: one entry in an input: a JSON line, an export item, a database
  row. Every record has a **byte range** in its chain.
- **tool database**: a tool's own SQLite file. Polylogue copies its rows up
  to the **synced prefix** named by the tool's checkpoint.
- **capture**: one conversation saved by the browser extension, appended to
  the provider's **capture log**. The
  extension's **backfill** fetches older conversations as **jobs**; a worker
  holds a job by a **claim** that expires.
- **diagnostic**: a note from a parser about a record it could not fully
  read, with a `reason`.

## What the index holds

- **session**, **message**, **block**, **tool call**, **event**, **link**,
  **document**, **attachment**, **run**, **process**, **work graph**: the
  parsed things. A message seen in several inputs is stored once, with one
  **shown version**.
- **duplicate**: a message another route of its session holds under a
  different ID, matched by exact evidence; the session shows the other one.
- **record position**: a message's place among the messages its record
  holds; it orders messages that share an input and byte offset.
- **derivation**: a product computed from parsed rows, with a rule version
  (cost, the work graph, session facts, measures).

## What people and agents add

- **note** (with **tag**, **title** and the other note kinds), **kept
  result**, **variant**, **setting**, **delivery** (exactly what was put into
  a model's context), **client request** (one write request, for retries).
- **excision**: stored inputs removed on purpose, with everything parsed from
  them. The request is recorded once, with the identities it removed
  (**excised**); a stop part way needs no recovery.

## The daemon

- **daemon**: owns the archive lock and serves clients over a socket.
- **loop**: compares what exists with what should exist and does the
  difference: collect, store, parse, derive.
- **writer**: the one thread that writes the databases.
- **read pool**, **reader**, **reader slot**: readers see one SQLite read
  transaction each; a slot bounds how many run at once.
- **write lock**: what writers take so readers start from a consistent pair
  of databases.
- **command table**: every command, once, for the CLI, MCP and HTTP.
- **mode**: `standalone`, or `primary` with Sinex holding the archive.
- **row batch**: the rows one archive write added, sent to Sinex in primary
  mode under the name `rows:<archive>:<position>`.

## Queries and output

- **query**, **unit** (what a query lists: sessions, messages, ...),
  **field**, **condition**, **measure**.
- **page** and **cursor**: a page ends with a cursor holding the last key
  shown; the next page reads the rows after it.
- **view**: how a session is shown: `full`, `dialogue`, `prompts`,
  `actions`.
- **budget**: the size a page for a model may take.
- **masking**: hiding credentials in output; stored bytes never change.
- **change feed** and **position**: `changes --since POSITION`.

## Words not used

| Not | Use |
|---|---|
| retain, retained, retention | stored, kept, or nothing |
| custody, reservation, reserve (bytes) | pending blob row |
| publication, publish | commit, make visible, send to Sinex, write lock |
| admission, admit | insert, reader slot |
| retire, retirement | delete, replace |
| canonical | archive (connection), sorted (JSON), exact |
| candidate | match, possible |
| interpretation | format, parsed rows |
| backing | mode |
| unfinished | pending |
| legacy, Archive*N*, old*N* | older (for an outside format), or delete |
| basis | source bytes |
| control, proof (tests) | test |
| qualify, qualification | test, or nothing |
| lease | claim |
| witness, authority, closure, frozen, donor | say what it is |
| snapshot (for reads or captures) | read transaction, capture |

Excision, epoch, sealed and receipt (a delivery record) are kept.
