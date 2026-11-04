-- Begin and completion can arrive independently in a filtered confirmed stream.
CREATE TABLE query_run (
    id TEXT PRIMARY KEY,
    binding_hash TEXT NOT NULL CHECK(length(binding_hash)=64),
    run_hash TEXT NOT NULL CHECK(length(run_hash)=64),
    query TEXT,
    object_ref TEXT CHECK(object_ref IS NULL
        OR (substr(object_ref,1,8)='request:' AND length(CAST(object_ref AS BLOB))>8)
        OR (substr(object_ref,1,4)='run:' AND length(CAST(object_ref AS BLOB))=40)),
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
    CHECK((run IS NULL AND query IS NULL AND object_ref IS NULL)
       OR (run IS NOT NULL AND ((query IS NOT NULL AND object_ref IS NULL)
                            OR (query IS NULL AND object_ref IS NOT NULL))))
) STRICT, WITHOUT ROWID;
CREATE INDEX query_run_query ON query_run(query,commit_seq,source_order,id);
