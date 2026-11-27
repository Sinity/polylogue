CREATE TABLE alternate_indexed (
  alternate TEXT PRIMARY KEY,
  commit_seq INTEGER NOT NULL
) STRICT;
CREATE TABLE alternate_text (
  alternate TEXT NOT NULL,
  position INTEGER NOT NULL,
  body TEXT NOT NULL,
  UNIQUE(alternate,position)
) STRICT;
CREATE VIRTUAL TABLE alternate_fts USING fts5(body,content='alternate_text',content_rowid='rowid');
CREATE TRIGGER alternate_text_insert AFTER INSERT ON alternate_text BEGIN
  INSERT INTO alternate_fts(rowid,body) VALUES(new.rowid,new.body);
END;
CREATE TRIGGER alternate_text_delete AFTER DELETE ON alternate_text BEGIN
  INSERT INTO alternate_fts(alternate_fts,rowid,body) VALUES('delete',old.rowid,old.body);
END;
