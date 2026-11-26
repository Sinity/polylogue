//! The byte placement check: the rows that place each revision's bytes come
//! over exactly, or the conversion stops.
use super::*;

/// Rows placing two revisions' bytes: an input renamed by the conversion and
/// one that keeps its name.
const PLACED: &str = "
INSERT INTO input(id, origin) VALUES ('codex:rollout:s1', 'codex'), ('in2', 'claude-code');
INSERT INTO blob(hash, size, stored_size, storage) VALUES (zeroblob(32), 10, 8, X'01'), (X'11', 4, 4, X'02');
INSERT INTO revision(input, number, produced, chain, length, commit_seq)
  VALUES ('codex:rollout:s1', 1, 'first', 1, 10, 1), ('in2', 1, 'first', 1, 4, 2);
INSERT INTO extent(input, chain, offset, length, blob, blob_offset, commit_seq)
  VALUES ('codex:rollout:s1', 1, 0, 10, zeroblob(32), 0, 1), ('in2', 1, 0, 4, X'11', 0, 2);
";

/// A new file holding the converted rows, with the format-6 rows attached as
/// `old`, after `tamper` changes the new file.
fn checked(tamper: &str) -> Result<Carried, Error> {
    let dir = tempfile::tempdir().unwrap();
    let old = Connection::open(dir.path().join("old.db")).unwrap();
    polylogue_store::schema::ensure_archive(&old).unwrap();
    old.execute_batch(PLACED).unwrap();
    drop(old);
    let conn = Connection::open(dir.path().join("new.db")).unwrap();
    polylogue_store::schema::ensure_archive(&conn).unwrap();
    conn.execute_batch(&PLACED.replace("codex:rollout:s1", "codex:rollout:s1@m1"))
        .unwrap();
    conn.execute_batch(tamper).unwrap();
    conn.execute(
        "ATTACH DATABASE ?1 AS old",
        [dir.path().join("old.db").to_str().unwrap()],
    )
    .unwrap();
    conn.execute_batch(
        "CREATE TEMP TABLE renamed (old TEXT PRIMARY KEY, new TEXT NOT NULL) WITHOUT ROWID;
         INSERT INTO temp.renamed VALUES ('codex:rollout:s1', 'codex:rollout:s1@m1');",
    )
    .unwrap();
    carried_over(&conn)
}

#[test]
fn rows_carried_over_exactly_under_renamed_inputs_pass() {
    let carried = checked("").unwrap();
    assert_eq!(
        (carried.extents, carried.blobs, carried.revisions),
        (2, 2, 2)
    );
}

#[test]
fn a_dropped_extent_stops_the_conversion() {
    let error = checked("DELETE FROM extent WHERE input = 'in2'")
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("extent row missing from the new file") && error.contains("in2"),
        "{error}"
    );
}

#[test]
fn a_changed_blob_size_stops_the_conversion() {
    let error = checked("UPDATE blob SET size = 5 WHERE hash = X'11'")
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("blob row missing from the new file"),
        "{error}"
    );
}

#[test]
fn a_revision_on_another_chain_stops_the_conversion() {
    let error = checked("UPDATE revision SET chain = 2 WHERE input = 'in2'")
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("revision row"), "{error}");
}
