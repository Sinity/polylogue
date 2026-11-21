CREATE TABLE file(id INTEGER PRIMARY KEY,name TEXT UNIQUE,offset INTEGER,compressed INTEGER,bytes INTEGER,hash BLOB);
CREATE TABLE message(
 pk INTEGER PRIMARY KEY,id TEXT NOT NULL UNIQUE,session TEXT NOT NULL,parent TEXT,
 file INTEGER NOT NULL REFERENCES file(id),start INTEGER NOT NULL,end INTEGER NOT NULL,
 priority INTEGER NOT NULL,hash BLOB NOT NULL,body_offset INTEGER NOT NULL,body_bytes INTEGER NOT NULL);
CREATE TABLE older(
 pk INTEGER PRIMARY KEY,message INTEGER NOT NULL REFERENCES message(pk),hash BLOB NOT NULL UNIQUE,
 base_hash BLOB NOT NULL,base_offset INTEGER NOT NULL,base_bytes INTEGER NOT NULL,
 prefix INTEGER NOT NULL,suffix INTEGER NOT NULL,patch BLOB NOT NULL);
CREATE TABLE extra_claim(
 pk INTEGER PRIMARY KEY,message INTEGER NOT NULL REFERENCES message(pk),
 older INTEGER REFERENCES older(pk),file INTEGER NOT NULL REFERENCES file(id),
 start INTEGER NOT NULL,end INTEGER NOT NULL,priority INTEGER NOT NULL,
 UNIQUE(message,file,start,end));
CREATE INDEX older_claim ON extra_claim(older,priority);
CREATE TABLE hit(pk INTEGER PRIMARY KEY,message INTEGER NOT NULL REFERENCES message(pk),channel INTEGER);
CREATE VIRTUAL TABLE terms USING fts5(body,content='',tokenize='unicode61 remove_diacritics 2');
