-- migration-safety: additive-no-backup
-- Durable verification-receipt cache for raw-authority frontier census blob
-- content-hash checks (polylogue-byw3y).
--
-- `_verified_blob_bytes` (storage/raw_reconciler.py) used to cache a
-- verified blob's on-disk identity fingerprint only in a PROCESS-LIFETIME
-- dict, so every daemon restart re-hashed every accepted frontier blob from
-- scratch even when nothing had changed underneath it -- tens of GiB of
-- wasted reads given the corpus's measured size skew (top 5% of raws = 69%
-- of bytes, polylogue-el374). This table persists the same receipt
-- durably: one row per blob_hash records the on-disk fingerprint
-- (device/inode/size/mtime/ctime) observed at the moment
-- `BlobStore.verify()` last proved the bytes match their content-addressed
-- hash.
--
-- A row is a HINT, never an authority: a census only skips re-hashing when
-- the blob's *current* `stat()` matches every column here exactly (see
-- `_verified_blob_bytes` in raw_reconciler.py). Any mismatch -- including
-- the blob having vanished, been replaced, or moved to a new inode --
-- forces a fresh re-hash; the receipt is then either confirmed identical
-- (fingerprint unchanged, row untouched) or overwritten with the new
-- fingerprint after a fresh verify() success. There is no upgrade path
-- that trusts a receipt whose fingerprint does not match: absent proof of
-- "unchanged", the default is always "re-verify from scratch".
--
-- This migration adds an empty table -- there is nothing to backfill,
-- since no receipt has ever been durably recorded before this lands. The
-- first census after upgrading still re-verifies every blob once, exactly
-- once, not once per restart thereafter.
CREATE TABLE verified_blob_receipts (
    blob_hash        BLOB NOT NULL CHECK(length(blob_hash) = 32),
    st_dev           INTEGER NOT NULL,
    st_ino           INTEGER NOT NULL,
    st_size          INTEGER NOT NULL CHECK(st_size >= 0),
    st_mtime_ns      INTEGER NOT NULL,
    st_ctime_ns      INTEGER NOT NULL,
    verified_at_ms   INTEGER NOT NULL CHECK(verified_at_ms >= 0),
    PRIMARY KEY(blob_hash)
) STRICT;
