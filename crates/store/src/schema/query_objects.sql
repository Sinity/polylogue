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

-- Begin and completion can arrive independently in a filtered confirmed stream.
CREATE TABLE query_run (
    id TEXT PRIMARY KEY,
    binding_hash TEXT NOT NULL CHECK(length(binding_hash)=64),
    query TEXT,
    at TEXT,
    read_at TEXT,
    author TEXT CHECK(author IS NULL OR json_valid(author)),
    via TEXT,
    outcome TEXT CHECK(outcome IS NULL OR json_valid(outcome)),
    count TEXT,
    exact INTEGER CHECK(exact IS NULL OR exact IN (0,1)),
    finished_at TEXT,
    commit_seq INTEGER CHECK(commit_seq IS NULL OR commit_seq>=0),
    source_order INTEGER CHECK(source_order IS NULL OR source_order>=0),
    run BLOB,
    completion BLOB,
    CHECK(run IS NOT NULL OR completion IS NOT NULL),
    CHECK((run IS NULL) = (query IS NULL))
) STRICT, WITHOUT ROWID;
CREATE INDEX query_run_query ON query_run(query,commit_seq,source_order,id);
