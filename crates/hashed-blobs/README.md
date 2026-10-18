# hashed-blobs

A content-addressed file store for Unix, shared by Polylogue and Sinex (joint #11, #12, #14, #19, #23, #24). It follows Polylogue's API (`polylogue-spec/spec/storage/blobs.md`) and Sinex's sealed design (`sinex-spec/decisions/privacy.md`). It needs stable Rust 1.98.1 or newer and a C compiler for zstd. Releases are Git tags (`hashed-blobs-v0.1.0`, `hashed-blobs-v0.1.1`); see `docs/shared-file-layer-release.md` in the Polylogue repository.

The crate holds no policy. Inline-versus-file size, when to use a delta, snapshots, page maps, joining, the excision rewrite, privacy classes and key management belong to the caller.

| Layout | Name | Reads |
|---|---|---|
| plain | BLAKE3 of the content | zstd frames (4 MiB, each with its content checksum) plus a seek table in the zstd seekable format; stock `zstd -d` decompresses the file; `Store::read_range` checks all frames; `Reader` verifies the whole name once, then reuses its index and one cached frame |
| delta | BLAKE3 of the content | a skippable header naming the base, then one zstd frame made the way `zstd --patch-from` makes one: the base as prefix, a window over base and content, long-distance matching; base always plain (at most two blobs per read); not seekable |
| sealed | BLAKE3 keyed hash under HKDF(data key, "hashed-blobs name v1") (#24) | frames under ChaCha20-Poly1305 with a per-encoding frame key HKDF-SHA256(data key, salt = random 32-byte encoding salt, info = "hashed-blobs frame v2" ‖ name) and nonce = frame index; the AAD binds the encoding context and the frame's table entry (#23); a footer carries the frame table and a BLAKE3 of the ciphertext, so `verify` works without the key (#19b) |

## Writing

- `put` and `Batch` cannot be cancelled. `put_plain_checked` polls a caller callback while reading and compressing input and while verifying an existing copy; it refuses an existing delta or sealed blob of the same name. A blocking read or a single compression call is not interrupted.
- A new file is written to `blobs/tmp/`, synced and hard-linked into place, so concurrent writers never overwrite each other. If the name already exists, the existing file is fully verified first (with the key, for sealed data). An invalid existing file is refused and left alone, and the new file stays in `blobs/tmp/`. Nothing is repaired silently.
- A `Batch` shares syncs; it is not atomic. On error some names may be installed and others left in tmp. On success every returned name is durable. Retrying is safe. Never roll back installed names: another reference may use them. Record references only after the commit succeeds. `contains` reports presence only.

## Reading and verification

- Supported plain files are the ones this writer makes: known-size, checksummed zstd frames without a dictionary, and a seek table without checksums. Anything else is refused as corrupt. Decoding checks every frame's length and extent.
- `Store::read_range` decodes and checks every plain frame before returning, so it is slow for repeated small reads. It does not check the whole blob against its name, so another valid blob placed under that name would be returned.
- `Reader` holds one open plain blob and one decoded frame of up to 4 MiB. The first range verifies the whole name; `verify_plain` does the same with a callback that sees each chunk, which must not pass chunks on before success. Later ranges reuse the open file and frame. Switching blobs drops the old one. Each concurrent call may hold its own blob and frame, so the caller limits concurrency. The lock is released before IO and callbacks. A change to the open file is refused; replacing the path never redirects the open file. `Reader` is `Send + Sync`.
- Callbacks run before metadata work, per frame header, per read of at most 64 KiB, and between decoded chunks of at most 64 KiB, so cancellation works even on blocks that decode to nothing. Memory is the requested output, the seek table, one frame, fixed buffers and the zstd window (window log up to 31). Frames over 4 MiB are streamed, not cached, so each range decodes the whole frame again. `read` returns the whole content in memory.

## Sealed encoding version 2

Every encoding draws a fresh 32-byte salt from the OS random source before encrypting its first frame, including repeated puts of the same bytes and encodings later dropped by dedup. The salt is public. Naming does not use the salt, so dedup still finds the existing copy; but since dedup happens after encryption, the salt is what prevents key and nonce reuse. If the random source fails, the put fails.

The frame key is HKDF-SHA256 with the data key as input, the salt as HKDF salt, and `hashed-blobs frame v2` followed by the 32-byte keyed name as info. A nonce is eight zero bytes followed by the frame index as a big-endian u32.

The file is ciphertext frames followed by a footer: `HBSEAL2\0` (8 bytes), salt (32), key-reference length (u16 LE), key reference (UTF-8), full/delta flag (u8, 0 or 1), base keyed name (32, delta only), total decoded length (u64 LE), frame count (u32 LE), frame entries (ciphertext length and decoded length, u32 LE each), ciphertext BLAKE3 (32), footer length excluding the last eight bytes (u32 LE), and `HBS2` (4). There is at least one frame, even for empty content. The unkeyed ciphertext hash covers the frames and the footer through the frame table; it detects damage without the key but does not authenticate the footer.

Each frame's AAD is the footer up to the total decoded length, then the keyed name (32), frame index (u32 LE), frame count (u32 LE), last-frame flag (u8), the frame's ciphertext and decoded lengths (u32 LE each) and its decoded start offset (u64 LE). Changing any table entry or earlier length therefore fails authentication, while a range read still reads only the footer and the frames it needs. Version 1 files are refused; there is no migration.

## Measurements

Deltas need the patch-from parameters (2026-10-12, synthetic text, a 100-byte change). A raw-content dictionary saved nothing from a 4 MiB base up. `zstd --patch-from` settings gave 441 B at 4 MiB, 7 KB at 64 MiB, 29 KB at 256 MiB and 123 KB at 1 GiB. Decoding needs about twice the base in memory (0.5 GB at 256 MiB), so the caller limits base size; Polylogue's limit is 640 MiB. The test `delta_against_a_large_base_stays_small` fails if the window is capped at zstd's default.

`examples/pt5.rs` measured real volumes on the operator's QLC drive (2026-10-12):

| Input | Mode | Rate |
|---|---|---|
| a day of media, 1.4 GiB in 6,880 files (Opus, WebP) | one durable `put` each, level 1 | 8 MiB/s |
| the same | `Batch`, `Durability::PerFile`, 1,000 files a batch | 25 MiB/s |
| the same | `Batch`, `Durability::Filesystem` | 397 MiB/s |
| a 15 GiB ZIP export | one `put`, level 1 | 75 MiB/s |

Syncs, not compression, set the pace for small blobs. `Durability::Filesystem` calls `syncfs` once before the renames and once after; because that also flushes other programs' writes, use it for bulk imports, and individual puts or small `PerFile` batches for live capture. Media stored at 94% of its size and level 3 gained nothing over level 1, so media is written at level 1. On the 15 GiB export, random 64 KiB range reads took 10 ms median and 70 ms at p99, `verify` streamed at 530 MiB/s and peak memory was 11 MB.

## Tests

`src/tests.rs` covers round trips, layouts, ranges, deltas, `assemble`, sealed blobs and corruption detected by the crate itself (the Polylogue archive is No_COW, so btrfs checksums are off). By hand: turning off frame checksums fails the raw-block test, and removing the frame count and last flag from the AAD fails the truncation range-read test.

`tests/sealed_lifetime.rs` shows how a caller can revoke in-flight sealed reads: after a lock, expiry, relock or purge, a read that unwrapped its key earlier must not cache or return plaintext. Bytes already returned to a client stay returned. The unlock owner there is a test stand-in, not Sinex's unlock service.
