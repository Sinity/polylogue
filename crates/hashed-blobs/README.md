# hashed-blobs

Polylogue owns this standalone Unix file-layer crate. The prepared next owner release is `hashed-blobs-v0.1.1`; remote publication remains pending. The consumed `hashed-blobs-v0.1.0` tag is immutable. It builds on stable Rust 1.98.1 or newer with a C compiler for zstd. Product backends and privacy policy remain in the caller. See `docs/shared-file-layer-release.md` in the owner repository for the release and consumer-replacement procedure.

The shared content-store crate (joint #11, #12, #14, #19, #23, #24), built to Polylogue's API (`polylogue-spec/spec/storage/blobs.md`, "The file layer is the hashed-blobs crate") and Sinex's sealed design (`sinex-spec/decisions/privacy.md`). Flake check `joint-hashed-blobs`.

It holds no policy. Inline-versus-file size, when to use a delta, snapshots, page maps, joining, the excision rewrite, privacy classes and key management stay with the caller. The crate gives the caller the primitives those need.

| Layout | Name | Reads |
|---|---|---|
| plain | BLAKE3 of the content | zstd frames (4 MiB, each with its content checksum) plus a seek table in the zstd seekable format; stock `zstd -d` decompresses the file; stateless ranges validate all frames; scoped Reader first verifies the whole name, then reuses its index and bounded selected-frame buffer |
| delta | BLAKE3 of the content | a skippable header naming the base, then one zstd frame made as `zstd --patch-from` makes one: the base as prefix, a window over base and content, long-distance matching; base always plain (at most two blobs per read); not seekable |
| sealed | BLAKE3 keyed hash under HKDF(data key, "hashed-blobs name v1") (#24) | frames under ChaCha20-Poly1305 with a fresh per-encoding frame key HKDF-SHA256(data key, salt = random 32-byte encoding salt, info = "hashed-blobs frame v2" ‖ name) and nonce = frame index; AAD binds the v2 encoding context and current frame entry (#23); a footer carries the frame table and a BLAKE3 of the ciphertext, so `verify` works without the key (#19b) |

## Integrity, custody and bounded verification

The writer encoding is unchanged in 0.1.1. Supported plain input is the existing writer's known-size, checksummed, dictionary-free zstd frames with a no-table-checksum seek layout. General zstd input without these fields is refused as corruption/unsupported representation. The descriptor's two unused low bits remain ignored as specified by the seekable format. Actual decoding checks every logical length and exact physical extent: matching two corrupt length declarations is insufficient.

The stateless `Store::read_range` conservatively validates all plain frames with the existing zstd streaming decoder before returning selected bytes. It checks structure/checksums but retains the weaker whole-name boundary: another valid object substituted under a name can return its bytes. This method now performs whole-object decode work and is unsuitable for repeated small reads. Sealed range behavior is unchanged.

`Reader` owns exactly one opened plain representation and its expected identity, index and descriptor generation. Its first plain range checks the whole expected name; `verify_plain` performs that same validation with a fallible callback and optional caller consumption of provisional chunks. It warms at most one decoded frame of up to 4 MiB. Subsequent ranges use the held file/index and reuse that bounded frame. Switching objects evicts the slot; there is no static/global registry or unbounded map. The retained slot holds at most one frame; each concurrent in-flight call can also own a representation/frame, so the caller's admitted concurrency bounds the aggregate. The state mutex is released before IO and callbacks. Mutation of the opened descriptor refuses and invalidates the slot; pathname replacement never redirects a held proof to the new file (it can conservatively refuse a generation change caused by unlink). Failed/cancelled validation publishes no reusable state. Reader remains Send + Sync.

Callbacks poll before metadata/frame work, per indexed header, and between at most 64 KiB decoder outputs, allowing actual caller cancellation and deadlines. Requested output, the seek table, one bounded frame buffer, fixed IO buffers and the zstd window determine memory. Maximum window log is 31, preserving legitimate larger writer frames; frames larger than 4 MiB remain readable through bounded streaming, but repeated ranges over those frames decode the selected frame again before returning checked bytes. This declared cost avoids an arbitrary giant allocation or unchecked partial-frame publication. `read` remains eager. Statistics count decoded/compressed bytes and format metadata reads, excluding filesystem-stat syscall cost; no wall-clock or RSS speedup is claimed.

Polylogue ReadHandle owns this one-slot Reader; each fork starts a new lifetime. Existing retained-source range callers keep fresh SQL source/blob admission while reusing the actual opened representation. `assemble` also owns a scoped Reader across its component reads/seeks. Input presentation and public operation reads have an operation-scoped handle; verification, counting and output do not inherit proof from a pathname string. Sequential recovery and capture use the same streaming verifier. A failed batch cannot advance source publication before its required captured dependency succeeds.

Successful put reuses an existing object only after full expected-name verification (with the supplied key for sealed data). An invalid/unavailable destination is refused, left untouched, and the completed new candidate remains in `blobs/tmp/` for owner-scoped recovery. No silent repair occurs. New names use no-clobber hard-link installation on the same filesystem; concurrent immutable writers cannot overwrite a winner. Existing-object validation adds work to dedup and does not certify future out-of-band mutations. Sealed/delta full verification retains its documented eager memory limits.

A batch groups synchronization, not atomic visibility. Before commit, new candidates are temporary. Error may leave installed names and temporary files; it promises neither zero effects nor prefix durability. Success establishes durability of every returned object, including revalidated reused objects. Retrying the same puts is safe; callers publish domain references only once all dependencies succeed. Never roll back an installed prefix blindly, since another reference can own it. `contains` reports presence only.

## Sealed encoding version 2

Every encoding attempt draws a fresh 32-byte salt from the OS CSPRNG before encrypting its first frame. This includes identical repeated puts, different compression levels or frame sizes, full-versus-delta representations, and attempts later discarded by dedup or a lost install. The salt is public encoding metadata, not a new data key. Naming still uses the unchanged keyed-name derivation, so dedup keeps the existing installed representation. Dedup happens after encryption and cannot prevent key/nonce reuse. Entropy failure refuses the encoding; no deterministic fallback exists.

The frame key is HKDF-SHA256 with the data key as input, the encoding salt as HKDF salt, and `hashed-blobs frame v2` followed by the 32-byte keyed name as info. A nonce is eight zero bytes followed by the frame index as a four-byte big-endian integer. ChaCha20-Poly1305 remains the AEAD. A fresh HKDF salt fixes the encoding boundary without adding a different AEAD dependency; AES-GCM-SIV would tolerate nonce misuse but needs its own implementation and vectors.

The wire file is ciphertext frames followed by a footer. In order, the footer holds `HBSEAL2\0` (8 bytes), encoding salt (32), UTF-8 key-reference byte length (u16 LE), key reference, full/delta flag (u8, 0 or 1), base keyed name (32 bytes only for delta), total decoded length (u64 LE), frame count (u32 LE), frame entries (ciphertext length and decoded length, each u32 LE), ciphertext BLAKE3 (32), footer length excluding the final eight bytes (u32 LE), and `HBS2` (4). At least one frame is required, including for empty content. The unkeyed ciphertext hash covers the frames and footer through the frame table; it detects damage without a key and does not authenticate an attacker-controlled footer.

Each frame's AAD is the footer prefix through total decoded length, followed by the keyed blob name (32), frame index (u32 LE), frame count (u32 LE), last-frame flag (u8), that frame's ciphertext and decoded lengths (u32 LE each), and its cumulative decoded start (u64 LE). These fields are known when that frame is compressed. No future frame table or whole encoding is buffered for AAD. The decoded start is passed from the existing incremental encode/read position, without rescanning earlier entries. Changing a current entry, an earlier ciphertext offset, or an earlier decoded length changes the selected ciphertext or its authenticated metadata and fails; a range read still reads only the footer and required frames. Version 1 is refused, with no compatibility decoder. The prototype has no public re-encoding API; this rule covers the existing encoding attempts rather than claiming such an API was added. Existing v1 files are not migrated or overwritten; install refuses an existing unsupported representation under the unchanged name. Use fresh disposable roots for the v2 prototype.

**Tested:**
- round trip, dedup, empty and frame-exact inputs, file layout and mode 0600;
- a stock decoder reads plain blobs;
- range reads across frame boundaries;
- extra roots are read in order and never written;
- leftover tmp files are harmless;
- deltas: small, `base_of`, `put_delta_within` storing a delta only under a size bound (for a caller's delta-or-plain policy), delta on a delta refused, a missing base named in the error;
- `assemble` with seeks and a zero-length extent;
- the excision primitives;
- corruption found by the crate itself, never by btrfs (the archive is No_COW): a full read and `verify` fail and name the blob, `verify` repairs nothing, and stateless plain ranges refuse a damaged representation;
- a flipped byte in a raw (incompressible) block, which still decodes, is caught by the frame checksum;
- sealed:
  - round trip and range reads with an unwrap callback, refused without one;
  - names differ from the plaintext hash, per key, and from the plain copy;
  - no plaintext in the file;
  - keyless verify passes intact files and finds corruption;
  - a wrong key, or a shredded key the caller can no longer give, both fail;
  - truncation, reordering and a frame swapped in from another blob under the same key all fail authentication, on full and range reads, even when the attacker recomputes the ciphertext hash;
  - fixed independent HKDF salt vectors, distinct key/nonce pairs for repeated and differently encoded attempts before discarded duplicate installation, and unchanged installed dedup bytes;
  - tampered encoding salt/context refused even with a recomputed unkeyed hash, including on a range read;
  - version 1 refused and empty encodings retaining an authenticated frame;
  - sealed deltas within one key work;
  - deltas across key boundaries and deltas on deltas are refused.

**Deltas need the patch-from parameters** (measured 2026-10-12, synthetic text, a 100-byte change plus a short tail). With the base as a raw-content dictionary at level 3, the delta saved everything at 1 MiB and nothing from 4 MiB up (16, 64 and 256 MiB: 100-101% of plain compression, with or without long-distance matching). `zstd --patch-from` (base as prefix, large window, long mode) gave 441 B at 4 MiB, 7 KB at 64 MiB, 29 KB at 256 MiB and 123 KB at 1 GiB, all decoding identically. Decode memory is about twice the base (0.5 GB at 256 MiB, 2.1 GB at 1 GiB; encode similar), so a caller bounds the base size by its memory target; Polylogue's bound is 640 MiB (`polylogue-spec/spec/storage/blobs.md`). The crate now makes deltas this way; the test `delta_against_a_large_base_stays_small` (24 MiB base) fails if the window is capped at the default.

**Mutation checks** (run by hand, recorded here): with frame checksums turned off, the raw-block test fails; with the frame count and last flag removed from the AAD, the truncation range-read test fails. Before those two tests were added, both mutations passed the whole suite.

## Historical PT5 measurements before 0.1.1: real volumes on the operator's drive (2026-10-12)

`examples/pt5.rs` stored one real day of media and one real export into throwaway roots on `/realm` (Crucial P3 QLC NVMe), under `nice`, then deleted them. It printed counts only.

| Input | Mode | Stored / input | Rate | CPU per GiB |
|---|---|---|---|---|
| a day of media: 181 audio files (0.45 GiB, Opus) and 6,699 screen frames (0.96 GiB, WebP) | one durable `put` each, level 1 | 94.4% | 8 MiB/s (173 s) | 8.3 s |
| the same, level 3 | one durable `put` each | 94.4% | 7 MiB/s | 10.4 s |
| the same, syncs disabled (`eatmydata`, to measure) | | | 274 MiB/s (5.3 s) | 2.4 s |
| the same, `Batch` with `Durability::PerFile`, 1,000 files a batch | | | 25 MiB/s (58 s) | 3.5 s |
| the same, `Batch` with `Durability::Filesystem` | | | 397 MiB/s (3.6 s) | 2.2 s |
| a 15 GiB export (ZIP), one `put`, level 1 | | 100.0% | 75 MiB/s | 1.9 s |

On the export: 64 KiB range reads at random offsets took 10 ms at the median and 70 ms at p99; `verify` streamed at 530 MiB/s; peak memory was 11 MB.

What this settled:
- **Syncs, not compression, set the pace of small blobs.** Three syncs per blob (file, directory, tmp directory) took 97% of the time. `Batch` shares them. `Durability::Filesystem` syncs the filesystem once before the renames and once after, which is btrfs's own group commit: 48 times faster than individual puts. Because `syncfs` also flushes other programs' pending writes on `/realm`, it suits bulk imports and backfills; live capture keeps individual puts or small `PerFile` batches.
- **Media does not compress.** Opus and WebP stored at 94%, and level 3 bought nothing over level 1 at 25% more CPU. Media is written at level 1; text keeps level 3.
- **Reads are positional.** The first version read the whole file for every range read and every `verify`; on a 15 GiB blob that would have been 15 GiB per 64 KiB read. Readers now fetch the seek table, header or footer and the frames they need, and `verify` hashes frame by frame.

**Prototype limits:**
- `read` (the whole content) still returns it in memory; range reads, `assemble` and `verify` do not;
- `put_sealed` needs a seekable input, because the keyed name must be known before the frames are encrypted (two passes, no plaintext on disk);
- a delta holds base and content in memory to encode and decode (about twice the base).

**Delta or plain, compressing once** (2026-10-12): `put_delta_within(base, bytes, level, max_len)` stores the delta only if its file would be at most `max_len` bytes and returns `None`, writing nothing, otherwise. A caller choosing between a delta and a plain blob uses it instead of measuring first and writing after, which compresses twice (twice the time and the memory on a large base, about 0.5 GB per pass for a 256 MiB base). The delta header's size is one constant, `DELTA_HEAD_LEN` (56).

## Caller-owned unlock lifetime witness

`tests/sealed_lifetime.rs` exercises actual `Store::read_range_with` through a finite caller-owned privacy boundary. It leaves this crate's policy-free API unchanged. A synthetic single-material owner uses a fresh unlock identity, explicit monotonic test clock, current caller authorization and material revision; only its bounded cache/output admission holds the same guard as revocation.

Barriers pause work after unwrap or before publication. The disabled-fence negative control reproduces plaintext cache refill and output after lock acknowledgement. The guarded path rejects those late results across lock, expiry, relock/reunlock (including an unchanged key reference) and purge. It checks operator-populated cache denial to an agent, fresh-session reconstruction, and publication winning the guard before lock. Previously handed-off plaintext intentionally survives as the disclosure boundary.

This is one bounded range, one material/key and one in-memory cache slot, with a fixed synthetic key provider standing in for authorized age unwrapping. It is not Sinex's implemented unlock service, full-text index, token router, key backup lifecycle, stream transport or buffer zeroization. No assertion claims to erase plaintext already held by a worker/client. Public-key wrapping remains PT10's separate mechanism test. The extracted shared-file-layer qualification and normal workspace suite run these integration tests.
