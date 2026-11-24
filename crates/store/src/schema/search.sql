-- One full-text file. The index's texts are split among several of these by
-- text key (search/files.rs), each written by one connection at a time.
-- Two contentless indexes keyed by text.id, holding folded text. text_fts:
-- prose (text and thinking blocks, documents). output_fts: tool output
-- (tool_result blocks, and bodies written through file-writing and
-- file-editing tool calls). Membership follows the blocks (search_triggers.sql).
CREATE VIRTUAL TABLE text_fts USING fts5(body, content='', contentless_delete=1,
                                         tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE output_fts USING fts5(body, content='', contentless_delete=1,
                                           tokenize='unicode61 remove_diacritics 2');
-- Merge segments less eagerly than the default (4), and buffer 16 MiB of
-- pending terms per index before writing a segment instead of the default
-- 1 MiB: with the default, flushing tiny segments and merging them was over
-- half of a first build's full-text time (perf, 2026-10-19). Over the files
-- that is the 64 MiB per index one file had.
INSERT INTO text_fts(text_fts, rank) VALUES ('automerge', 16);
INSERT INTO output_fts(output_fts, rank) VALUES ('automerge', 16);
INSERT INTO text_fts(text_fts, rank) VALUES ('hashsize', 16777216);
INSERT INTO output_fts(output_fts, rank) VALUES ('hashsize', 16777216);
-- The statistics row: key 0, holding one private-use character. A search
-- reads each file's statistics through it, so a file where nothing matches
-- still counts its texts (search/rank.rs); the statistics leave it out.
INSERT INTO text_fts(rowid, body) VALUES (0, char(57344));
INSERT INTO output_fts(rowid, body) VALUES (0, char(57344));
-- The entries and tokens each index holds, the statistics row aside. FTS5
-- does not take a removed entry out of a contentless-delete table's own
-- totals, so every entry written or removed adjusts these in the same
-- transaction (text_rows::write_entry), and ranking reads them.
CREATE TABLE totals (
  name              TEXT PRIMARY KEY,       -- text_fts | output_fts
  entries           INTEGER NOT NULL,
  tokens            INTEGER NOT NULL
) STRICT, WITHOUT ROWID;
INSERT INTO totals(name, entries, tokens) VALUES ('text_fts', 0, 0), ('output_fts', 0, 0);
