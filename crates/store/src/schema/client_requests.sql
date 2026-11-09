CREATE TABLE client_request (
  id            TEXT PRIMARY KEY,        -- UUIDv7 chosen by the client, once per user action
  content_hash  BLOB NOT NULL,           -- BLAKE3 of the request's canonical JSON, without the ID
  created_at    INTEGER NOT NULL,
  request       BLOB NOT NULL,           -- the original request as JSON, zstd
  outcome       BLOB,                    -- the original typed outcome, zstd; set by the transaction that applies the request
  effect        BLOB,                    -- the immutable typed authored effect, zstd; set with the outcome
  commit_seq    INTEGER NOT NULL DEFAULT 0  -- the write counter of the transaction that wrote it
) STRICT, WITHOUT ROWID;
CREATE INDEX client_request_commit_seq ON client_request(commit_seq,id);
