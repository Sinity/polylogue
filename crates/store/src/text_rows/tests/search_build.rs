//! The final search copy keeps the same term frequencies as incremental writes.
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
    for (n, (text, bits)) in [
        ("amber cedar amber", 1),
        ("Łódź Straße Øresund", 2),
        ("cedar amber birch", 3),
        ("only unindexed words", 0),
        ("", 3),
        ("shrub's limbs: 𝔘𝔫𝔦𝔠𝔬𝔡𝔢", 1),
    ]
    .into_iter()
    .enumerate()
    {
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
    index_pending(&incremental, || ()).unwrap();
    assert!(vocabulary(&built, "fts_vocab_output").is_empty());
    build_search_indexes(&built, |_| Ok(())).unwrap();
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
        index_pending(conn, || ()).unwrap();
        crate::text_rows::remove(conn, 2, 2).unwrap();
    }
    for table in ["fts_vocab_text", "fts_vocab_output"] {
        assert_eq!(
            vocabulary(&built, table),
            vocabulary(&incremental, table),
            "after continuing: {table}"
        );
    }
}
