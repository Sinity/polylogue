-- Existing aliases belong to every historical dependency closure. Their local
-- creation order was not recorded; zero preserves current serialized batches.
ALTER TABLE machine ADD COLUMN first_commit_seq INTEGER NOT NULL DEFAULT 0 CHECK(first_commit_seq>=0);
