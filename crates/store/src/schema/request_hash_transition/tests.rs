use super::*;
use crate::schema::{ARCHIVE_VERSION, PRIMARY_INDEX_VERSION};
use polylogue_model::user::UserWrite;
use rusqlite::{Connection, types::Value};

// Frozen neutral predecessor bytes: unlike assertion values, the signal
// structs were never sorted by the predecessor's intent function.
const SIGNAL_INTENT: &[u8] = br#"["synthetic-machine",{"author":"script:signal-fixture","via":"daemon"},{"write":"signal_commit","prior":null,"checkpoint":{"owner":"synthetic-machine","evaluator":{"now":0,"rules":{},"products":[],"inputs":[],"consumed":[],"pending":{},"open":{},"retractions":[],"reports":[]},"rules":[]},"outbox":[],"events":[]}]"#;

fn signal(id: &str) -> UserRequest {
    let (machine, authorship, write) = serde_json::from_slice(SIGNAL_INTENT).unwrap();
    UserRequest {
        id: id.into(),
        machine,
        authorship,
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        write,
    }
}

#[test]
fn request_hash_transition_recognizes_the_exact_prior_signal_encoding() {
    let request = signal("signal-fixture");
    let prior = *blake3::hash(SIGNAL_INTENT).as_bytes();
    assert_eq!(prior_hash(&request).unwrap(), prior);
    assert_ne!(
        crate::primary::authored::request_hash(&request).unwrap(),
        prior
    );
}

fn saved(id: &str) -> UserRequest {
    crate::fixture::authored::operator_request(
        id.into(),
        UserWrite::SaveQuery {
            name: "example".into(),
            text: "sessions".into(),
            denominator: None,
        },
    )
}

fn insert(conn: &Connection, request: &UserRequest, sequence: i64) {
    let tx = conn.unchecked_transaction().unwrap();
    crate::user_writes::apply(&tx, request, sequence).unwrap();
    tx.commit().unwrap();
}

fn source(primary: bool) -> (Connection, crate::source::Source) {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    if primary {
        crate::schema::ensure_primary_index(&conn).unwrap();
    } else {
        crate::schema::ensure_archive(&conn).unwrap();
    }
    let source = crate::source::Source::of(&conn).unwrap();
    (conn, source)
}

fn rows(conn: &Connection, source: crate::source::Source) -> Vec<Vec<Value>> {
    conn.prepare(&source.sql("SELECT id,created_at,request,outcome,effect,commit_seq FROM @source.client_request ORDER BY id"))
        .unwrap()
        .query_map([], |row| (0..6).map(|i| row.get(i)).collect())
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn previous(conn: &Connection, primary: bool) {
    crate::schema::set_version(conn, 2).unwrap();
    assert_eq!(crate::schema::version(conn).unwrap(), 2);
    assert_eq!(
        matches!(
            crate::source::Source::of(conn).unwrap(),
            crate::source::Source::Cache
        ),
        primary
    );
}

fn reopen(conn: &Connection, primary: bool) -> Result<()> {
    if primary {
        crate::schema::ensure_primary_index(conn)
    } else {
        crate::schema::ensure_archive(conn)
    }
}

#[test]
fn request_hash_transition_preserves_mixed_records_and_pending_bytes_and_retry() {
    for primary in [false, true] {
        let (conn, source) = source(primary);
        let prior = saved("prior");
        let current = saved("current");
        insert(&conn, &prior, 1);
        insert(&conn, &current, 2);
        // This is the predecessor's exact tuple/struct encoding, independently
        // of the transition's implementation. SaveQuery carries no resolved data.
        let old_bytes =
            serde_json::to_vec(&(&prior.machine, &prior.authorship, &prior.write)).unwrap();
        let old_hash = blake3::hash(&old_bytes);
        assert_ne!(
            old_hash.as_bytes(),
            &crate::primary::authored::request_hash(&prior).unwrap()
        );
        conn.execute(
            &source.sql("UPDATE @source.client_request SET content_hash=?1 WHERE id=?2"),
            params![old_hash.as_bytes().as_slice(), prior.id],
        )
        .unwrap();
        let pending = saved("pending");
        let pending_hash = blake3::hash(
            &serde_json::to_vec(&(&pending.machine, &pending.authorship, &pending.write)).unwrap(),
        );
        conn.execute(&source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,commit_seq) VALUES(?1,?2,?3,?4,3)"), params![pending.id,pending_hash.as_bytes().as_slice(),pending.at.unix_ms(),crate::encoding::pack_json(&pending).unwrap()]).unwrap();
        let before = rows(&conn, source);
        previous(&conn, primary);
        assert!(matches!(
            crate::primary::authored::frozen_request(&conn, &prior.id),
            Err(StoreError::Conflict(_))
        ));
        reopen(&conn, primary).unwrap();
        assert_eq!(rows(&conn, source), before);
        assert_eq!(
            crate::schema::version(&conn).unwrap(),
            if primary {
                PRIMARY_INDEX_VERSION
            } else {
                ARCHIVE_VERSION
            }
        );
        for request in [&prior, &current] {
            let stored = crate::primary::authored::frozen_request(&conn, &request.id)
                .unwrap()
                .unwrap();
            assert_eq!(&stored.request, request);
            let mut retry = request.clone();
            retry.at = polylogue_model::Timestamp::from_unix_ms(request.at.unix_ms() + 99).unwrap();
            assert_eq!(
                crate::primary::authored::known_request(&conn, &retry).unwrap(),
                Some(stored)
            );
        }
        let pending_stored: Vec<u8> = conn
            .query_row(
                &source.sql("SELECT content_hash FROM @source.client_request WHERE id=?1"),
                [&pending.id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            pending_stored,
            crate::primary::authored::request_hash(&pending).unwrap()
        );
        let changes = conn.total_changes();
        reopen(&conn, primary).unwrap();
        assert_eq!(conn.total_changes(), changes);
        assert_eq!(rows(&conn, source), before);
    }
}

#[test]
fn request_hash_transition_refuses_corruption_without_partial_conversion() {
    for primary in [false, true] {
        for corrupt in ["hash", "request identity", "request body"] {
            let (conn, source) = source(primary);
            let first = saved("a-valid-prior");
            let later = saved("z-corrupt");
            insert(&conn, &first, 1);
            insert(&conn, &later, 2);
            let old_hash = blake3::hash(
                &serde_json::to_vec(&(&first.machine, &first.authorship, &first.write)).unwrap(),
            );
            conn.execute(
                &source.sql("UPDATE @source.client_request SET content_hash=?1 WHERE id=?2"),
                params![old_hash.as_bytes().as_slice(), first.id],
            )
            .unwrap();
            match corrupt {
                "hash" => conn.execute(&source.sql("UPDATE @source.client_request SET content_hash=zeroblob(32) WHERE id=?1"), [&later.id]).unwrap(),
                "request identity" => conn.execute(&source.sql("UPDATE @source.client_request SET id='zz-other' WHERE id=?1"), [&later.id]).unwrap(),
                _ => conn.execute(&source.sql("UPDATE @source.client_request SET request=x'0001' WHERE id=?1"), [&later.id]).unwrap(),
            };
            previous(&conn, primary);
            let before = rows(&conn, source);
            assert!(reopen(&conn, primary).is_err(), "{corrupt}");
            assert_eq!(crate::schema::version(&conn).unwrap(), 2);
            assert_eq!(rows(&conn, source), before);
            let retained: Vec<u8> = conn
                .query_row(
                    &source.sql("SELECT content_hash FROM @source.client_request WHERE id=?1"),
                    [&first.id],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(retained, old_hash.as_bytes());
            assert!(
                conn.pragma_query_value(None, "foreign_keys", |row| row.get::<_, bool>(0))
                    .unwrap()
            );
        }
    }
}

#[test]
fn register_request_hash_transition_reopens_both_predecessors_and_reads_unrelated_authored_content()
{
    use std::os::unix::fs::MetadataExt;
    for predecessor in [1, 2] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("archive");
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("meta.json"), b"{\"mode\":\"standalone\"}\n").unwrap();
        let conn = Connection::open(root.join("archive.db")).unwrap();
        if predecessor == 1 {
            crate::schema::tests::archive_before_unknown_time(&conn);
        } else {
            crate::schema::ensure_archive(&conn).unwrap();
        }
        let prior = signal("old-request");
        insert(&conn, &prior, 20);
        let old_hash = blake3::hash(
            &serde_json::to_vec(&(&prior.machine, &prior.authorship, &prior.write)).unwrap(),
        );
        conn.execute(
            "UPDATE client_request SET content_hash=?1 WHERE id=?2",
            params![old_hash.as_bytes().as_slice(), prior.id],
        )
        .unwrap();
        let note = crate::fixture::authored::operator_request(
            "unrelated-note".into(),
            UserWrite::Assert {
                assertions: vec![polylogue_model::assertion::NewAssertion {
                    kind: polylogue_model::assertion::AssertionKind::Note,
                    key: None,
                    value: Some(serde_json::json!("Retain this note")),
                    links: vec![polylogue_model::assertion::NewAssertionLink {
                        role: polylogue_model::assertion::LinkRole::About,
                        reference: polylogue_model::Reference::parse("repo:synthetic").unwrap(),
                    }],
                    valid_until: None,
                }],
            },
        );
        insert(&conn, &note, 21);
        crate::schema::set_version(&conn, predecessor).unwrap();
        let before = rows(&conn, crate::source::Source::Ledger);
        drop(conn);
        let index_path = root.join("index.db");
        let index = Connection::open(&index_path).unwrap();
        crate::schema::ensure_index(&index).unwrap();
        index
            .execute(
                "INSERT INTO meta(key,value) VALUES('retained-progress','exact')",
                [],
            )
            .unwrap();
        let identity = crate::schema::index_identity(&index).unwrap();
        drop(index);
        let inode = std::fs::metadata(&index_path).unwrap().ino();
        let (archive, lock) = crate::Archive::open_or_create(&root).unwrap();
        assert_eq!(std::fs::metadata(&index_path).unwrap().ino(), inode);
        let index = Connection::open(&index_path).unwrap();
        assert_eq!(crate::schema::index_identity(&index).unwrap(), identity);
        assert_eq!(
            index
                .query_row(
                    "SELECT value FROM meta WHERE key='retained-progress'",
                    [],
                    |r| r.get::<_, String>(0)
                )
                .unwrap(),
            "exact"
        );
        let conn = Connection::open(archive.archive_db()).unwrap();
        assert_eq!(rows(&conn, crate::source::Source::Ledger), before);
        assert_eq!(crate::schema::version(&conn).unwrap(), ARCHIVE_VERSION);
        let reads = crate::ReadPool::open(&archive).unwrap();
        let reader = reads.reader().unwrap();
        let reference = polylogue_model::Reference::parse("assertion:unrelated-note").unwrap();
        assert!(matches!(
            reader.authored_record(&reference).unwrap(),
            Some(crate::read::authored_evidence::AuthoredReadEvidence::Local { .. })
        ));
        assert!(
            crate::primary::authored::known_request(&conn, &prior)
                .unwrap()
                .is_some()
        );
        drop(reader);
        drop(reads);
        drop(conn);
        drop(index);
        drop(lock);
        drop(archive);
        let (archive, _lock) = crate::Archive::open_or_create(&root).unwrap();
        let conn = Connection::open(archive.archive_db()).unwrap();
        assert_eq!(rows(&conn, crate::source::Source::Ledger), before);
    }
}

#[cfg(feature = "faults")]
#[test]
fn request_hash_transition_fault_preserves_all_bytes_and_version_then_retries() {
    for primary in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("request-hash-fault.db");
        let conn = Connection::open(&path).unwrap();
        if primary {
            crate::schema::ensure_primary_index(&conn).unwrap();
        } else {
            crate::schema::ensure_archive(&conn).unwrap();
        }
        let source = crate::source::Source::of(&conn).unwrap();
        let request = saved("old");
        insert(&conn, &request, 1);
        let old_hash = blake3::hash(
            &serde_json::to_vec(&(&request.machine, &request.authorship, &request.write)).unwrap(),
        );
        conn.execute(
            &source.sql("UPDATE @source.client_request SET content_hash=?1 WHERE id=?2"),
            params![old_hash.as_bytes().as_slice(), request.id],
        )
        .unwrap();
        previous(&conn, primary);
        let before = rows(&conn, source);
        crate::fault::arm(
            "archive.request_hash_transition",
            conn.path().unwrap(),
            crate::fault::Action::Fail,
            0,
            1,
        );
        assert!(reopen(&conn, primary).is_err());
        assert_eq!(crate::schema::version(&conn).unwrap(), 2);
        assert_eq!(rows(&conn, source), before);
        let retained: Vec<u8> = conn
            .query_row(
                &source.sql("SELECT content_hash FROM @source.client_request WHERE id=?1"),
                [&request.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(retained, old_hash.as_bytes());
        reopen(&conn, primary).unwrap();
        assert!(
            crate::primary::authored::known_request(&conn, &request)
                .unwrap()
                .is_some()
        );
        assert_eq!(rows(&conn, source), before);
    }
}
