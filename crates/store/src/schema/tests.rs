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
fn a_newer_archive_is_refused() {
    let mut conn = Connection::open_in_memory().unwrap();
    create_archive(&conn).unwrap();
    set_version(&conn, 99).unwrap();
    assert!(matches!(
        migrate_archive(&mut conn),
        Err(StoreError::TooNew { found: 99, .. })
    ));
}

#[test]
fn index_schema_changes_require_a_version_and_digest_update() {
    assert_eq!(
        (
            INDEX_VERSION,
            blake3::hash(INDEX.as_bytes()).to_hex().to_string()
        ),
        (
            17,
            "6a4cebb9297e587d0e7b91b158a6ee95eba00396e9e9b19648d3f5ec590356f1".to_owned()
        ),
        "bump INDEX_VERSION and record its schema digest when index.sql changes"
    );
}
