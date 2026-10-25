use super::*;

#[test]
fn every_schema_applies_to_an_empty_database() {
    for (name, create) in [
        ("archive", create_archive as fn(&Connection) -> Result<()>),
        ("index", create_index),
        ("embeddings", create_embeddings),
    ] {
        let conn = Connection::open_in_memory().unwrap();
        create(&conn).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn a_database_at_another_version_is_refused() {
    let conn = Connection::open_in_memory().unwrap();
    ensure_archive(&conn).unwrap();
    ensure_archive(&conn).unwrap();
    set_version(&conn, 99).unwrap();
    assert!(matches!(
        ensure_archive(&conn),
        Err(StoreError::Version {
            database: "archive",
            found: 99
        })
    ));
    let conn = Connection::open_in_memory().unwrap();
    ensure_embeddings(&conn).unwrap();
    set_version(&conn, 2).unwrap();
    assert!(matches!(
        ensure_embeddings(&conn),
        Err(StoreError::Version {
            database: "embeddings",
            found: 2
        })
    ));
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
            39,
            "4dc316e938886634315eacbc553cd80d51467b59a892ee9f1f262bc29e6524bb".to_owned(),
            "e27529a9578815c3ee02058decc971ecf7c7125774a6d0e9b885eddd8a928490".to_owned()
        ),
        "record current source digests; bump INDEX_VERSION for schema or row-format changes, not comments"
    );
}
