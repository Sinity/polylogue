CREATE TABLE variant_indexed (
  variant TEXT PRIMARY KEY,
  commit_seq INTEGER NOT NULL
) STRICT;
CREATE TABLE variant_text (
  variant TEXT NOT NULL,
  n INTEGER NOT NULL,
  body TEXT NOT NULL,
  UNIQUE(variant,n)
) STRICT;
CREATE VIRTUAL TABLE variant_fts USING fts5(body,content='variant_text',content_rowid='rowid');
CREATE TRIGGER variant_text_insert AFTER INSERT ON variant_text BEGIN
  INSERT INTO variant_fts(rowid,body) VALUES(new.rowid,new.body);
END;
CREATE TRIGGER variant_text_delete AFTER DELETE ON variant_text BEGIN
  INSERT INTO variant_fts(variant_fts,rowid,body) VALUES('delete',old.rowid,old.body);
END;
