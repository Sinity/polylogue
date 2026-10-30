CREATE TABLE client_request (
  id            TEXT PRIMARY KEY,        -- UUIDv7 chosen by the client, once per user action
  content_hash  BLOB NOT NULL,           -- BLAKE3 of the request's sorted-key JSON, without the ID
  created_at    INTEGER,
  request       BLOB,                    -- the original request as JSON, zstd
  outcome       BLOB,                    -- the original typed outcome, zstd
  effect        BLOB,                    -- the immutable typed authored effect, zstd
  imports       TEXT NOT NULL DEFAULT '{"observed":[],"declared":[]}', -- import evidence and historical claims, JSON
  historical    BLOB,                    -- typed retired accepted evidence with its exact original bytes
  original      BLOB,                    -- original encoding of a compatible frozen request after conversion
  commit_seq    INTEGER NOT NULL DEFAULT 0  -- the write counter of the transaction that wrote it
) STRICT, WITHOUT ROWID;
CREATE INDEX client_request_commit_seq ON client_request(commit_seq,id);
