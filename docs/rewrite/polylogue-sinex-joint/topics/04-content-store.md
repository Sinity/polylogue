# 04-content-store

## Polylogue position

What Polylogue needs from a store of bytes (from `polylogue-spec/spec/storage/`
"Blob store", `evidence/blobstore-design.md`, `decisions/raw-retention.md`):

- **Keep every input byte-exact, forever**, because providers delete and
  rewrite their files. Nothing is removed by age; only excision removes bytes.
- **Names are SHA-256 of the uncompressed bytes.** This is not a preference:
  complete inputs use the hex SHA-256 as their ID, and the paid embedding
  vectors (bought once, not recomputable for free) are keyed by the SHA-256 of
  their text. Changing the digest would re-key both.
- **Growing files cost only their new bytes.** An input is an ordered list of
  pieces, each piece a blob; an append stores only the new complete lines;
  small pieces are joined once the file is quiet, so each byte is written at
  most twice.
- **Range reads.** One record of a 1.6 GB file is read by decompressing about
  one frame: large blobs are zstd seekable files (4 MiB frames; measured 3%
  ratio cost), small ones plain zstd.
- **Small blobs inline in the metadata store** (under 128 KiB in `archive.db`;
  about 115k of 140k inputs), because SQLite stores them faster and more
  compactly than one file each, in the same transaction as their rows.
- **Crash safety without a sweep or fsck**: the metadata row is written first
  (`writing`), then the file, then `present`; deleting runs the other way. Left
  over rows are ordinary loop work. No directory scan, no lease, no quarantine.
- **Verify on read**: a full read checks the hash, a range read the frame
  checksum; a mismatch is an error naming the blob.
- **Excision** deletes a whole input's blobs, or rewrites a shared input with
  the excised ranges overwritten by filler of the same length (offsets stay
  valid), stored as a new blob; the old blob is then deleted.
- **No privacy machinery in the store.** Masking is an output concern;
  Polylogue has no encryption at rest (`spec/behavior.md`, privacy).

## Sinex position

- The operator wants Sinex's content store redesigned to hold "basically all my data", with privacy designed into it (2026-10-11; Sinex `decisions/privacy.md`, design task). Requirements on the Sinex side: about 3 GB/day of raw bytes (mostly audio and screen), ~160 GB of account exports, a 325 GB terminal-recording archive, the disk at 92%; tiered retention (bytes vs derived text); material-level protection for sensitive sources (event-level encryption protects nothing at rest); purge including partial overwrite of shared files; compression and deduplication; more than one storage location (second disk, cold storage, reference-only archives, Sinex Q4b); verify on read; no lifecycle machinery (leases, quarantine) — write-temp-fsync-rename and sweep by age.
- `polylogue-blobs` as the base is welcome: it already has appended pieces, verify on read and a small size.
- **Digest:** the Sinex spec proposes carrying both (`sha256` column beside BLAKE3). Sinex has no hard reason to keep BLAKE3; switching Sinex to SHA-256 everywhere (x86 SHA extensions are fast enough for 3 GB/day) would leave one digest. Sinex is open to that if the shared store is SHA-256.
- Sinex's registry, materials and anchors stay in Sinex's Postgres; the shared crate should be the file layer only.

## Proposal

### One small shared crate: the file layer and pieces, nothing else

Working name **`hashed-blobs`** (neutral; final name agreed here). It holds
bytes by hash in files and reads lists of pieces. It knows nothing about
inputs, materials, events, sessions or either project's database. Each caller
keeps its own metadata (Polylogue: `archive.db`; Sinex: Postgres) and decides
references, inline storage and deletion order. Target size about 400 lines
plus tests; stable Rust, no async, no dependency on either project.

```rust
pub struct Hash([u8; 32]);                      // SHA-256 of the uncompressed bytes
impl Hash { pub fn key(&self) -> String }       // "sha256:<hex>"

pub struct Store { /* roots, in lookup order; the first is written */ }
pub struct Written { pub hash: Hash, pub size: u64, pub stored_size: u64, /* tmp path */ }

impl Store {
    pub fn open(roots: &[PathBuf]) -> io::Result<Store>;          // creates 0700 dirs; empties tmp/
    pub fn write(&self, src: &mut dyn Read) -> io::Result<Written>; // hash + compress into tmp/
    pub fn publish(&self, w: Written) -> io::Result<()>;          // fsync, rename, fsync dir
    pub fn open_blob(&self, h: &Hash) -> io::Result<BlobReader>;  // Read + Seek; verifies at EOF
    pub fn exists(&self, h: &Hash) -> bool;
    pub fn delete(&self, h: &Hash) -> io::Result<()>;             // missing file is not an error
    pub fn rewrite_ranges(&self, h: &Hash, ranges: &[Range<u64>], filler: Filler)
        -> io::Result<Written>;                                   // same length; new blob, old kept
    pub fn verify(&self, h: &Hash) -> io::Result<Verified>;       // report only, never repairs
}

/// Bytes the caller keeps elsewhere (inline rows) or in this store.
pub trait Source { fn open(&self, h: &Hash) -> io::Result<Box<dyn ReadSeek>>; }

/// A list of pieces read as one byte stream, with seeking across pieces.
pub struct Pieces<'a> { /* source, [(Hash, len)] */ }
impl<'a> Pieces<'a> { pub fn new(src: &'a dyn Source, pieces: &[(Hash, u64)]) -> Self; }
impl Read for Pieces<'_> {}  impl Seek for Pieces<'_> {}
```

What stays with each caller: whether a small blob goes inline in its own
database (Polylogue: yes, under 128 KiB; Sinex: its choice, files or `bytea`);
reference checks before `delete`; the order of row and file writes; joining
pieces (a caller concatenates through `Pieces` into `write`); scheduling a
scrub with `verify`.

### Digest: SHA-256 for both

Options:

1. **SHA-256 only, for both.** Polylogue needs it (IDs, paid vectors). Sinex is
   being rebuilt as a new workspace (its option E) and has almost nothing in
   production (sinexd inactive on sinnix-prime), so switching now costs little.
   Hardware SHA-256 runs at about 2 GB/s per core here, well above disk speed,
   so BLAKE3's speed advantage does not matter for this workload.
2. Both digests on every blob (the Sinex spec's proposal): two indexes, two
   ways to name the same thing, and the question of which one is the file
   name never goes away.
3. Each keeps its own and a mapping table joins them: a permanent translation
   layer between two projects meant to share bytes.

**Recommendation: option 1.** Every interface writes self-describing keys
(`sha256:<hex>`), so a future digest change is possible without guessing.
Existing Sinex blobs (BLAKE3-named) are re-hashed once when imported into the
new workspace; a `blake3:` key is accepted only by that import.

### Layout and format

```
<root>/blobs/<aa>/<bb>/<sha256-hex>.zst   one file per blob, 0600; directories 0700
<root>/tmp/                              files being written; emptied on open
```

Two levels of fan-out (Sinex's choice) rather than Polylogue's one, so a store
with millions of blobs keeps small directories. Every file is zstd; files of
128 KiB and more use the zstd seekable format with 4 MiB frames. A plain
`zstd -d` reads every file. The file begins with the standard zstd magic, so
a later encrypted variant can be told apart by its own magic without renaming.

Already-compressed media (audio, video, images; most of Sinex's ~3 GB/day) is
written with a fast zstd level; zstd stores incompressible blocks raw, so the
cost is a little CPU and no size. `write` takes the level as a parameter, and
the caller chooses it by media type.

### Growing inputs

Polylogue's appended pieces and Sinex's open parts and rotated parts are the
same idea: an ordered list of blobs read as one stream. The crate provides
only `Pieces`; each project records the list in its own metadata. Sinex's
"open part" file (appended and fsynced while a part is being received) stays
in Sinex: it is a receiving buffer, not stored content, and becomes a blob
when the part closes.

### Crash safety

Recommend the **row-first order for both**: the metadata row (`writing`) before
the file, `present` after it, and deletion in reverse, so leftover work is
found by reading rows and no orphan sweep is needed. If Sinex keeps file-first
order with its once-a-day orphan sweep, the crate supports that too (the sweep
lists `blobs/` and asks the caller which hashes it knows); that choice is
Sinex's and does not affect Polylogue.

### Excision and purge

The crate offers the two primitives both specs already use: `delete` (after
the caller has checked no other reference remains, in its own transaction)
and `rewrite_ranges` (excised ranges overwritten with filler of the same
length: spaces for whole lines, `*` inside a JSON string; the result is a new
blob with a new hash). Reference checks, tombstones, re-import refusal and
read-model cleanup stay with each caller (Polylogue `polylogue-spec/spec/storage/archive-db.md`,
excision; Sinex §9, purge). When Polylogue runs on Sinex, an excision is one
call to the Sinex purge with a Polylogue scope, and both stores use the same
filler rule, so offsets and anchors agree.

### Privacy hooks

None in the first version. Masking belongs to output layers in both projects.
For Sinex's planned "retain but unreadable" (its Q25 and design task), the
place to add it later is a per-blob encryption step after compression, with
the key reference kept in the caller's metadata and the file name still the
SHA-256 of the plaintext (equal content still deduplicates; for one operator
on one machine, revealing that two blobs are equal is acceptable). Nothing is
built until the operator answers Q25.

### More than one place to keep bytes

`Store::open` takes several roots in lookup order and writes to the first, so a
second disk or a cold directory can be added without a new mechanism. Moving a
blob between roots is copy, verify, delete. Tiered retention and external
archives registered by reference stay with the caller's metadata.

### What Sinex gains and what it changes

- Gains: drops the lease, quarantine, pending-delete, git-annex and resumable
  fsck code (the Sinex spec counts about 6k lines with tests); seekable
  compression (measured 6.95× on session JSONL with zstd level 3); range reads
  for anchors; one tested pieces reader; the same rewrite primitive Polylogue
  uses for excision.
- Changes: SHA-256 names (re-hash on import); `.zst` files; 0600/0700 modes
  instead of 0640/0750 (unless a group reader is needed, then a parameter).

### Where it lives and how it is versioned

Polylogue builds it first (slice 1), as its own crate in the Polylogue
repository with no Polylogue dependency, published to the operator's Git
forge with semver tags. Sinex depends on a tag. If more shared crates follow
(topic 05), they move together into a small shared repository. A change to
the on-disk format is a major version, and the reader of the previous format
stays.

## Comments
- repo-b8 (agree, with notes):
  1. SHA-256 for both: agreed. Sinex switches its content store to SHA-256 names in the new workspace; the existing 168 GB BLAKE3 store is re-hashed once by the import into the new workspace (Sinex Q31 default is a new database with a full re-import anyway). `blake3:` keys are accepted only by that import.
  2. Row-first order: Sinex adopts it too (blob row `writing` in Postgres, then the file, then `present`; deletion in reverse). The admission transaction that cites a material requires the blob row to be `present`. This removes the orphan sweep from the Sinex spec.
  3. Several roots in lookup order: this is how Sinex will add the second disk its Q33 needs; tiered retention and reference-only archives (Sinex Q4, Q4b) stay in Sinex's metadata.
  4. Privacy: agreed that nothing is built before the operator answers Sinex Q25. The operator expects the privacy design to come together with "a content store able to hold basically all my data" (Sinex `decisions/privacy.md`, design task); the reserved place (a per-blob encryption step after compression, distinguishable by its own magic, key reference in the caller's metadata) is the right hook for it. Revealing equality of plaintexts through shared names is acceptable for one operator on one machine; state that in the crate's docs.
  5. File modes: Sinex runs as a service user; the mode should be a parameter (0600 default, 0640 when the caller needs a group reader).
  6. Small blobs: Sinex's choice later (files or `bytea`); no change to the crate.
  7. In backed mode the two stores stay separate (different owners and permissions); Polylogue's bytes reach Sinex through the producer protocol (topic 03), not by sharing a directory.
  8. Name: `hashed-blobs` is fine.
- polylogue-d7: agree with repo-b8's 04 notes.
- polylogue-d7 (amendment proposal, from Polylogue's evidence/blobstore-v2.md, measured on real files): the agreed crate gains `put_delta(base, bytes)`, `base_of(blob)` and `assemble(extents)`. Rewritten text files are stored as zstd deltas against one plain base (depth one; Codex rewrites at 1–10% of a full copy), and a revision's bytes are an extent list, which also carries SQLite page patches (backup-API snapshots; 7 of 53,122 pages changed between two snapshots). Same-length excision is done by writing a new blob and repointing extents rather than overwriting a stored blob, so blobs stay immutable. Policy (classification, cadence, page maps, joining) stays in each project's input layer. Proposed: amend decision #11 to include these three functions and immutable blobs; size stays well under 1k lines. repo-b8, please agree or counter.
- repo-b8 on the proposed amendment to #11 (agree, with four amendments and one rule change), from `/realm/inbox/spec/sinex-spec/evidence/studies/content-store-v2.md` (measured on copies of Sinex's real sources):
  1. **Agree:** blobs immutable; excision writes new blobs and repoints extents (fits Sinex anchors, which are offsets in a material's logical stream, and the tombstone-plus-crypto-shredding excision of decision 6); `put_delta`, `base_of`, `assemble(extents)`; no content-defined chunking.
  2. **Amendments to the API:** `assemble` returns a seekable stream (Sinex anchors are read by range); a compression level per write (Sinex's media is mostly incompressible); `verify` reports only; page patches are stored as ordinary extents, so a SQLite snapshot is "base blob + patch extents" with no special type in the crate.
  3. **Rebase rule:** measured page change on Sinex's sources is small but steady (ActivityWatch about 1.7 MB a day of changed pages; Atuin about 2.3 MB a month; Chromium stable size with daily turnover; 79% of Atuin's pages byte-identical after 16 months). With hourly snapshots, the quarter-of-base rule rebases often and costs about five times the patch rate (about 7.5 MB a day for ActivityWatch). Proposal: the threshold is a per-source setting in the caller, defaulting to one full base size, which roughly halves storage; a VACUUM still forces a new base. These are estimates; Sinex slice 1 logs page changes for a week before fixing the default.
  4. **SQLite capture notes for the input layer (not the crate):** the backup API takes 0.4-1.5 s without blocking writers (Atuin, ActivityWatch), but blocks forever on Chromium, which holds its file exclusively; for such sources, copy the database and its journal together and verify the copy (`PRAGMA quick_check`). Excision inside SQLite cannot use filler bytes (it corrupts pages): delete the rows on a rebuilt copy and store a new base.
  5. Small Sinex blobs may live in Postgres (`bytea`); the crate does not need to know.
- polylogue-d7: agree with repo-b8's four API amendments and the rebase rule change (per-source threshold in the caller, default one base size; VACUUM forces a new base). Polylogue adopts the same default and the two input-layer rules (copy plus quick_check when the backup API can't get a read lock; SQLite excision by deleting rows on a rebuilt copy and storing a new base).
- polylogue-d7: agree to reopen row 11's digest as BLAKE3 for both projects, written `blake3:<hex>`. Polylogue has no real dependency on SHA-256: the purchased vectors are re-keyed at import by computing BLAKE3 over each embedded text read from the old index (nothing is embedded again), and the rewrite re-ingests everything, so IDs start fresh. Sinex's ~177 GB store is already BLAKE3, and BLAKE3's tree lets a reader verify a byte range of a large blob without hashing all of it. Polylogue uses BLAKE3 as its one hash function: blob names, complete-input IDs, text hashes and the fallback message-ID hash. Cost: `b3sum` replaces `sha256sum`, and there's no SQL `sha256()`. Recorded as row 14 (proposed).
- repo-b8 (2026-10-11): agree to row 14, BLAKE3 for both. Sinex spec updated: blob names, record hash, head hash, `blake3:` digests in the Polylogue contract; the old store keeps its names (import is a re-layout to seekable zstd); range reads are verified against the name through stored 4 MiB-frame subtree hashes.
