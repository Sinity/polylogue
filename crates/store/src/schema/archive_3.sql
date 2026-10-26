-- Requests are kept whole, in commit order, so user state can be replayed
-- into another archive.
ALTER TABLE client_request ADD COLUMN request BLOB;
ALTER TABLE client_request ADD COLUMN commit_seq INTEGER NOT NULL DEFAULT 0;
