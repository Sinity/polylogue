-- Questions are semantic objects. The first admitted execution fixes first_run.
CREATE TABLE query (
    hash TEXT PRIMARY KEY CHECK(length(hash)=64),
    normal_form TEXT NOT NULL CHECK(json_valid(normal_form)),
    object BLOB NOT NULL,
    first_run TEXT
) STRICT, WITHOUT ROWID;

CREATE TABLE query_edge (
    query TEXT NOT NULL REFERENCES query(hash),
    uses TEXT NOT NULL,
    how TEXT NOT NULL CHECK(how IN ('saved','result','combine','in')),
    edge BLOB NOT NULL,
    PRIMARY KEY(query,uses,how)
) STRICT, WITHOUT ROWID;
CREATE INDEX query_edge_uses ON query_edge(uses,query);
