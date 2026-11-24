//! A first build has the same ranked search as incremental writes.
use crate::archive::{cache, open_write};
use crate::text_rows::{build_search_indexes, index_pending, insert_alone};
use rusqlite::Connection;

fn vocabulary(conn: &Connection, table: &str) -> Vec<(String, i64, i64)> {
    conn.prepare(&format!("SELECT term,doc,cnt FROM {table} ORDER BY term"))
        .unwrap()
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn built_search_vocabulary_matches_incremental_writes() {
    let dir = tempfile::tempdir().unwrap();
    let built = open_write(&dir.path().join("built.db"), cache::SETUP).unwrap();
    let incremental = open_write(&dir.path().join("incremental.db"), cache::SETUP).unwrap();
    crate::schema::create_index(&built).unwrap();
    crate::schema::create_index(&incremental).unwrap();
    let inputs = [
        ("amber cedar amber", 1),
        ("Łódź Straße Øresund", 2),
        ("cedar amber birch", 3),
        ("only unindexed words", 0),
        ("", 3),
        ("shrub's limbs: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢", 1),
    ];
    for (n, (text, bits)) in inputs.into_iter().enumerate() {
        let id = n as i64 + 1;
        let hash = blake3::hash(text.as_bytes());
        let normalized = crate::encoding::normalized_text_hash(text);
        for conn in [&built, &incremental] {
            assert_eq!(
                insert_alone(conn, Some(id), hash.as_bytes(), &normalized, text, bits),
                id
            );
        }
        for bit in [1, 2].into_iter().filter(|bit| bits & bit != 0) {
            incremental
                .execute(
                    "INSERT INTO search_pending(text_id,fts_table) VALUES (?1,?2)",
                    [id, bit],
                )
                .unwrap();
        }
    }
    // Both contiguous and non-contiguous texts share the first group's body.
    // Keep separate groups in the incremental oracle.
    let group: i64 = built
        .query_row("SELECT group_id FROM text WHERE id=1", [], |row| row.get(0))
        .unwrap();
    let mut body = String::new();
    for (id, text) in [(1, inputs[0].0), (2, inputs[1].0), (5, inputs[4].0)] {
        let start = body.len() as i64;
        body.push_str(text);
        built
            .execute(
                "UPDATE text SET group_id=?1,start=?2 WHERE id=?3",
                [group, start, id],
            )
            .unwrap();
    }
    built
        .execute(
            "UPDATE text_group SET body=?1 WHERE id=?2",
            rusqlite::params![crate::encoding::pack_bytes(body.as_bytes()).unwrap(), group],
        )
        .unwrap();
    index_pending(&incremental).unwrap();
    assert!(vocabulary(&built, "fts_vocab_output").is_empty());
    build_search_indexes(&built).unwrap();
    for table in ["fts_vocab_text", "fts_vocab_output"] {
        assert_eq!(
            vocabulary(&built, table),
            vocabulary(&incremental, table),
            "{table}"
        );
    }
    // Reading the copied tables on this same connection also exercises the
    // virtual tables' transaction state, before a reopen can discard it.
    assert_eq!(
        vocabulary(&built, "fts_vocab_output")
            .iter()
            .find(|row| row.0 == "lodz"),
        Some(&("lodz".into(), 1, 1))
    );

    // The writer continues on its connection after installation. Later inserts
    // and removals must use the copied postings, not an earlier cached structure.
    for conn in [&built, &incremental] {
        let text = "maple amber maple";
        let hash = blake3::hash(text.as_bytes());
        let normalized = crate::encoding::normalized_text_hash(text);
        insert_alone(conn, Some(100), hash.as_bytes(), &normalized, text, 2);
        conn.execute(
            "INSERT INTO search_pending(text_id,fts_table) VALUES (100,2)",
            [],
        )
        .unwrap();
        index_pending(conn).unwrap();
        crate::text_rows::remove(conn, 2).unwrap();
    }
    let reopened = crate::archive::open_read(&dir.path().join("built.db")).unwrap();
    for reader in [&built, &reopened] {
        for table in ["fts_vocab_text", "fts_vocab_output"] {
            assert_eq!(
                vocabulary(reader, table),
                vocabulary(&incremental, table),
                "after continuing: {table}"
            );
        }
        for (table, expected) in [("text_fts", vec![1, 3]), ("output_fts", vec![3, 100])] {
            let ids = reader
                .prepare(&format!(
                    "SELECT rowid FROM {table} WHERE {table} MATCH 'amber' ORDER BY rowid"
                ))
                .unwrap()
                .query_map([], |row| row.get::<_, i64>(0))
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            assert_eq!(ids, expected);
        }
    }
}
