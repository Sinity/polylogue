//! Frozen Archive8 DDL from 579e1eaf390f36b919575b90cc0293739beac291.
//! The retained actual8 frontend was built from the same revision.
//! These are schema/data projection fixtures, not old-binary production witnesses.

use rusqlite::Connection;
use std::path::Path;

pub(super) fn replace(path: &Path) {
    let historical = path.with_extension("archive8");
    let old = Connection::open(&historical).unwrap();
    for sql in [
        include_str!("archive8/archive.sql"),
        include_str!("../../../fixture/archive4/002-excision-revision.sql"),
        include_str!("../../../fixture/archive4/003-vector-excision.sql"),
        include_str!("../../../fixture/archive4/004-local-blob-excision.sql"),
        include_str!("archive8/005-append-facts.sql"),
        include_str!("archive8/006-vector-deletion-policy.sql"),
        include_str!("archive8/007-document-excision.sql"),
        include_str!("archive8/008-extracted-blobs.sql"),
    ] {
        old.execute_batch(sql).unwrap();
    }
    old.execute("ATTACH DATABASE ?1 AS current", [path.to_str().unwrap()])
        .unwrap();
    // These fixtures contain no post8 source relationships. Copy only columns
    // declared by the frozen schema, including exact authored and unit bytes.
    let containers: i64 = old
        .query_row(
            "SELECT count(*) FROM current.observation_container",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(containers, 0);
    let tables = old
        .prepare("SELECT name FROM main.sqlite_master WHERE type='table' ORDER BY name")
        .unwrap()
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    for table in tables {
        let columns = old
            .prepare(&format!("PRAGMA main.table_info({table})"))
            .unwrap()
            .query_map([], |r| r.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .join(",");
        old.execute_batch(&format!(
            "DELETE FROM main.{table};
             INSERT INTO main.{table}({columns}) SELECT {columns} FROM current.{table};"
        ))
        .unwrap();
    }
    old.execute_batch("DETACH DATABASE current; PRAGMA user_version=8;")
        .unwrap();
    assert_eq!(crate::schema::version(&old).unwrap(), 8);
    assert_eq!(
        old.query_row(
            "SELECT count(*) FROM pragma_table_info('machine') WHERE name='first_commit_seq'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        old.query_row(
            "SELECT count(*) FROM sqlite_master WHERE name='observation_container'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
    drop(old);
    std::fs::rename(historical, path).unwrap();
}
