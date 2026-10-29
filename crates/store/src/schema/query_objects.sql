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

-- References can precede their question facts in a filtered confirmed stream.
CREATE TABLE query_run (
    id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    at TEXT NOT NULL,
    read_at TEXT NOT NULL,
    author TEXT NOT NULL CHECK(json_valid(author)),
    via TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK(json_valid(outcome)),
    count TEXT,
    exact INTEGER NOT NULL CHECK(exact IN (0,1)),
    commit_seq INTEGER NOT NULL CHECK(commit_seq>=0),
    source_order INTEGER NOT NULL CHECK(source_order>=0),
    run BLOB NOT NULL
) STRICT, WITHOUT ROWID;
CREATE INDEX query_run_query ON query_run(query,commit_seq,source_order,id);
