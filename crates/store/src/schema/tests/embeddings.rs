use super::*;

/// `embeddings.db` as each earlier build at the baseline created it, copied
/// from that build's declarations.
const BEFORE_REVISION: &str = include_str!("embeddings_before_revision.sql");
const BEFORE_QUERY_VECTOR: &str = include_str!("embeddings_before_query_vector.sql");
/// A build that kept excision receipts beside the vectors; dropping them is
/// not an addition.
const WITH_POLICY_RECEIPTS: &str = include_str!("embeddings_with_policy_receipts.sql");

type Vector = (i64, Vec<u8>, String, i64, i64, Vec<u8>, i64);
type Failure = (Vec<u8>, String, String, i64);

/// A file at `schema` with synthetic vectors and failures, closed.
fn earlier_file(dir: &std::path::Path, schema: &str) -> std::path::PathBuf {
    let path = dir.join("embeddings.db");
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(schema).unwrap();
    conn.execute_batch(
        "PRAGMA user_version = 1;
         INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at) VALUES
           (x'01','m-a',0,2,x'0000803f000000c0',100),
           (x'01','m-a',1,2,x'cdcc4c3e9a99993f',101),
           (x'02','m-b',0,2,x'0000c07f00008000',102);
         INSERT INTO embed_failure(text_hash,model,error,failed_at) VALUES
           (x'03','m-b','synthetic rejection',103),
           (x'04','m-c','synthetic rejection',104);",
    )
    .unwrap();
    drop(conn);
    path
}

fn rows(conn: &Connection) -> (Vec<Vector>, Vec<Failure>) {
    let vectors = conn
        .prepare(
            "SELECT id,text_hash,model,chunk,dims,vector,embedded_at FROM embedded ORDER BY id",
        )
        .unwrap()
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
            ))
        })
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let failures = conn
        .prepare("SELECT text_hash,model,error,failed_at FROM embed_failure ORDER BY text_hash")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    (vectors, failures)
}

fn revisions(conn: &Connection) -> Vec<(String, i64)> {
    conn.prepare("SELECT model,revision FROM embedding_revision ORDER BY model")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn an_embeddings_db_from_each_earlier_build_gains_the_added_tables_and_keeps_its_vectors() {
    let fresh = Connection::open_in_memory().unwrap();
    ensure_embeddings(&fresh).unwrap();
    let current = shape(&fresh).unwrap();
    // The revision table was added after the first build, which backfills one
    // revision per model; a file that already had it keeps its counts.
    for (schema, before_revisions) in [
        (BEFORE_REVISION, None),
        (
            BEFORE_QUERY_VECTOR,
            Some(vec![
                ("m-a".to_owned(), 2),
                ("m-b".to_owned(), 2),
                ("m-c".to_owned(), 1),
            ]),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = earlier_file(dir.path(), schema);
        let before = rows(&Connection::open(&path).unwrap());
        assert_eq!(before.0.len(), 3);
        for _ in 0..2 {
            let conn = crate::archive::open_write(&path, 8).unwrap();
            ensure_embeddings(&conn).unwrap();
            assert_eq!(shape(&conn).unwrap(), current);
            assert_eq!(version(&conn).unwrap(), 1);
            assert_eq!(rows(&conn), before);
            let expected = before_revisions.clone().unwrap_or_else(|| {
                vec![
                    ("m-a".to_owned(), 1),
                    ("m-b".to_owned(), 1),
                    ("m-c".to_owned(), 1),
                ]
            });
            assert_eq!(revisions(&conn), expected);
        }
        // The added tables work: a new vector advances its model's revision,
        // and a query vector is stored.
        let conn = crate::archive::open_write(&path, 8).unwrap();
        ensure_embeddings(&conn).unwrap();
        conn.execute_batch(
            "INSERT INTO embedded(text_hash,model,chunk,dims,vector,embedded_at)
               VALUES (x'05','m-a',0,2,x'0000803f0000803f',105);
             INSERT INTO query_vector(key,text_hash,vector,embedded_at)
               VALUES ('m-a',x'06',x'0000803f0000803f',106);",
        )
        .unwrap();
        let m_a = revisions(&conn)[0].1;
        assert_eq!(m_a, before_revisions.map_or(1, |r| r[0].1) + 1);
        assert_eq!(
            conn.query_row("SELECT count(*) FROM query_vector", [], |r| r
                .get::<_, i64>(0))
                .unwrap(),
            1
        );
    }
}

#[test]
fn an_embeddings_db_that_adding_tables_cannot_reach_is_refused_unchanged() {
    for (schema, change, difference) in [
        (
            BEFORE_REVISION,
            "ALTER TABLE embedded ADD COLUMN synthetic TEXT",
            "table embedded differs",
        ),
        (
            BEFORE_QUERY_VECTOR,
            "DROP TRIGGER embedded_revision_delete",
            "trigger embedded_revision_delete is missing",
        ),
        (
            WITH_POLICY_RECEIPTS,
            "",
            "table vector_meta is not declared; table vector_policy_receipt is not declared; \
             trigger vector_receipt_delete is not declared; trigger vector_receipt_insert is not \
             declared; trigger vector_receipt_update is not declared",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let path = earlier_file(dir.path(), schema);
        Connection::open(&path)
            .unwrap()
            .execute_batch(change)
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let conn = Connection::open(&path).unwrap();
        let error = ensure_embeddings(&conn).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "embeddings.db has a schema this Polylogue build cannot reach by adding tables \
                 ({difference}); it holds paid vectors and is left as it was"
            ),
            "{change}"
        );
        drop(conn);
        assert_eq!(std::fs::read(&path).unwrap(), bytes, "{change}");
    }
}
