use super::*;

#[test]
fn every_schema_applies_to_an_empty_database() {
    for (name, create) in [
        ("archive", ensure_archive as fn(&Connection) -> Result<()>),
        ("sinex", ensure_sinex),
        ("index", ensure_index),
        ("embeddings", ensure_embeddings),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
        create(&conn).unwrap_or_else(|e| panic!("{name} again: {e}"));
    }
}

fn refused(conn: &Connection, database: &str) -> bool {
    let result = match database {
        "archive" => ensure_archive(conn),
        _ => ensure_embeddings(conn),
    };
    matches!(result, Err(StoreError::OtherSchema { database: found }) if found == database)
}

#[test]
fn a_database_with_another_version_or_schema_is_refused() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    set_version(&conn, 99).unwrap();
    assert!(refused(&conn, "archive"));
    let conn = Connection::open_in_memory().unwrap();
    ensure_embeddings(&conn).unwrap();
    set_version(&conn, 2).unwrap();
    assert!(refused(&conn, "embeddings"));
    // The same version with a table missing, as an archive written before
    // that table was declared.
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    conn.execute_batch("DROP TABLE excised").unwrap();
    assert!(refused(&conn, "archive"));
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    conn.execute_batch("ALTER TABLE note ADD COLUMN synthetic TEXT")
        .unwrap();
    assert!(refused(&conn, "archive"));
}

#[test]
fn comments_and_spacing_in_the_declarations_do_not_change_the_schema() {
    let respaced: String = ARCHIVE
        .lines()
        .map(|line| {
            format!(
                "{}   -- synthetic remark, 'quoted'\n\n",
                line.split_whitespace().collect::<Vec<_>>().join("  ")
            )
        })
        .collect();
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch(&respaced).unwrap();
    set_version(&conn, 1).unwrap();
    ensure_archive(&conn).unwrap();
}

#[test]
fn index_format_changes_require_a_version_and_digest_update() {
    assert_eq!(
        (
            INDEX_VERSION,
            blake3::hash(INDEX.as_bytes()).to_hex().to_string(),
            blake3::hash(include_bytes!("../encoding/rows.dict"))
                .to_hex()
                .to_string()
        ),
        (
            42,
            "c6b7dd232c37df7287ed426b22ad2ba51916bddc1822d1a172fc6e08fdac779f".to_owned(),
            "e27529a9578815c3ee02058decc971ecf7c7125774a6d0e9b885eddd8a928490".to_owned()
        ),
        "record current source digests; bump INDEX_VERSION for schema or row-format changes, not comments"
    );
}
