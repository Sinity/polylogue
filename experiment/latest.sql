CREATE TABLE file(id INTEGER PRIMARY KEY,name TEXT,offset INTEGER,compressed INTEGER,bytes INTEGER,hash BLOB);
CREATE TABLE message(pk INTEGER PRIMARY KEY,id TEXT UNIQUE,session TEXT,parent TEXT,file INTEGER,start INTEGER,end INTEGER);
CREATE TABLE hit(pk INTEGER PRIMARY KEY,message INTEGER,channel INTEGER);
CREATE VIRTUAL TABLE terms USING fts5(body,content='',tokenize='unicode61 remove_diacritics 2');
