CREATE TABLE file(id INTEGER PRIMARY KEY,name TEXT UNIQUE,offset INTEGER,compressed INTEGER,bytes INTEGER,hash BLOB);
CREATE TABLE version(
 pk INTEGER PRIMARY KEY,hash BLOB NOT NULL UNIQUE,session TEXT NOT NULL,
 id TEXT NOT NULL,parent TEXT,body BLOB NOT NULL);
CREATE INDEX version_message ON version(session,id);
CREATE TABLE observation(
 pk INTEGER PRIMARY KEY,file INTEGER NOT NULL REFERENCES file(id),
 version INTEGER NOT NULL REFERENCES version(pk),start INTEGER NOT NULL,
 end INTEGER NOT NULL,priority INTEGER NOT NULL,
 UNIQUE(file,start,end,version));
CREATE INDEX observation_version ON observation(version);
CREATE TABLE selected(
 session TEXT NOT NULL,id TEXT NOT NULL,
 observation INTEGER NOT NULL REFERENCES observation(pk),PRIMARY KEY(session,id));
CREATE TABLE hit(pk INTEGER PRIMARY KEY,version INTEGER NOT NULL REFERENCES version(pk),channel INTEGER,text_id INTEGER NOT NULL,idx INTEGER NOT NULL);
CREATE INDEX hit_text ON hit(text_id,channel);
