use super::*;
use crate::schema::{ARCHIVE_VERSION, PRIMARY_INDEX_VERSION};
use polylogue_model::user::UserWrite;
use rusqlite::{Connection, types::Value};

// A neutral signal initializer. The wire oracle below uses the actual
// predecessor serializer, rather than treating this initializer as its bytes.
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

// The pre-canonical serializer and intent edits, independent of the raw-JSON
// conversion. Its output supplies the stored digest in these controls.
fn predecessor_bytes(request: &UserRequest) -> Vec<u8> {
    let mut intent = request.write.intent();
    match &mut intent {
        UserWrite::Assert { assertions } => {
            for assertion in assertions {
                if let Some(value) = &mut assertion.value {
                    value.sort_all_objects();
                }
            }
        }
        UserWrite::AssertQuery { assertion, .. } => {
            if let Some(value) = &mut assertion.value {
                value.sort_all_objects();
            }
        }
        UserWrite::ReviseAssertions { patch, .. } => {
            if let polylogue_model::assertion::Change::Set(value) = &mut patch.value {
                value.sort_all_objects();
            }
        }
        _ => {}
    }
    serde_json::to_vec(&(&request.machine, &request.authorship, intent)).unwrap()
}

#[test]
fn request_hash_transition_recognizes_the_exact_prior_signal_encoding() {
    let request = signal("signal-fixture");
    let prior = *blake3::hash(&predecessor_bytes(&request)).as_bytes();
    assert_eq!(
        prior_hash(&request, &crate::encoding::pack_json(&request).unwrap()).unwrap(),
        prior
    );
    assert_ne!(
        crate::primary::authored::request_hash(&request).unwrap(),
        prior
    );
}

#[test]
fn request_hash_transition_retains_prior_nested_map_order_escaping_and_intent_edits() {
    use polylogue_model::assertion::{AssertionKind, AssertionPatch, Change, NewAssertion};
    use std::collections::HashMap;

    let nested = HashMap::from([
        (
            "z",
            serde_json::json!({"second": "Café\n\"\\", "first": [1, 2]}),
        ),
        ("a", serde_json::json!({"right": true, "left": false})),
    ]);
    let value = serde_json::to_value(nested).unwrap();
    let assertion = NewAssertion {
        kind: AssertionKind::Note,
        key: None,
        value: Some(value.clone()),
        links: vec![],
        valid_until: None,
    };
    let writes = [
        UserWrite::ChatEnqueue(polylogue_model::user::NewChatAction {
            id: "chat".into(),
            request_key: "chat-key".into(),
            request_hash: polylogue_model::BlobHash::of(b"synthetic action"),
            request: value.clone(),
            stage_only: true,
            needs_approval: false,
            attachments: vec![],
        }),
        UserWrite::Assert {
            assertions: vec![assertion.clone()],
        },
        UserWrite::AssertQuery {
            query: "sessions".into(),
            assertion,
            resolved: vec![polylogue_model::Reference::parse("repo:resolved").unwrap()],
        },
        UserWrite::ReviseAssertions {
            bases: vec![],
            patch: AssertionPatch {
                value: Change::Set(value),
                ..Default::default()
            },
        },
        UserWrite::KeepResult {
            name: Some("result".into()),
            query: "sessions".into(),
            rows_json: "{\"resolved\":\"Café\n\\\"}".into(),
            read_at: "{\"now\":100}".into(),
        },
    ];
    let (conn, source) = source(false);
    for (number, write) in writes.into_iter().enumerate() {
        let request =
            crate::fixture::authored::operator_request(format!("request-{number}"), write);
        let original = crate::encoding::pack_json(&request).unwrap();
        let old = *blake3::hash(&predecessor_bytes(&request)).as_bytes();
        assert_eq!(prior_hash(&request, &original).unwrap(), old);
        conn.execute("INSERT INTO client_request(id,content_hash,created_at,request,commit_seq) VALUES(?1,?2,1,?3,1)",params![request.id,&old[..],original]).unwrap();
    }
    previous(&conn, false);
    let before = rows(&conn, source);
    reopen(&conn, false).unwrap();
    assert_eq!(rows(&conn, source), before);
    let mut statement = conn
        .prepare("SELECT request,content_hash FROM client_request ORDER BY id")
        .unwrap();
    for row in statement
        .query_map([], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .unwrap()
    {
        let (packed, hash) = row.unwrap();
        let request: UserRequest = crate::encoding::unpack_json(&packed).unwrap();
        assert_eq!(
            hash,
            crate::primary::authored::request_hash(&request).unwrap()
        );
    }
}

fn delivery(id: &str) -> UserRequest {
    crate::fixture::authored::operator_request(
        id.into(),
        UserWrite::Deliver(polylogue_model::user::Delivery {
            recipient_kind: "operator".into(),
            recipient_id: None,
            sender: "synthetic".into(),
            event: "test".into(),
            cwd: None,
            text: "Retain this delivered content".into(),
            refs: vec![],
            left_out: vec![],
        }),
    )
}

fn insert(conn: &Connection, request: &UserRequest) -> polylogue_model::user::UserOutcome {
    let tx = conn.unchecked_transaction().unwrap();
    let sequence = crate::archive::bump_counter(&tx).unwrap();
    let outcome = crate::user_writes::apply(&tx, request, sequence).unwrap();
    tx.commit().unwrap();
    outcome
}

fn source() -> (Connection, crate::source::Source) {
    let conn = Connection::open_in_memory().unwrap();
    conn.pragma_update(None, "foreign_keys", true).unwrap();
    crate::schema::ensure_archive(&conn).unwrap();
    (conn, crate::source::Source::Ledger)
}

fn rows(conn: &Connection, source: crate::source::Source) -> Vec<Vec<Value>> {
    conn.prepare(&source.sql("SELECT id,created_at,request,outcome,effect,commit_seq FROM @source.client_request ORDER BY id"))
        .unwrap()
        .query_map([], |row| (0..6).map(|i| row.get(i)).collect())
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

fn previous(conn: &Connection) {
    crate::schema::set_version(conn, 2).unwrap();
    assert_eq!(crate::schema::version(conn).unwrap(), 2);
    assert!(matches!(
        crate::source::Source::of(conn).unwrap(),
        crate::source::Source::Ledger
    ));
}
fn reopen(conn: &Connection) -> Result<()> {
    crate::schema::ensure_archive(conn)
}

#[test]
fn request_hash_transition_preserves_mixed_records_and_pending_bytes_and_retry() {
    {
        let (conn, source) = source();
        let prior = delivery("prior");
        let current = delivery("current");
        insert(&conn, &prior);
        insert(&conn, &current);
        // This is the predecessor's exact tuple/struct encoding, independently
        // of the transition's implementation. Deliver carries no resolved data.
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
        let pending = crate::fixture::authored::operator_request(
            "pending".into(),
            UserWrite::SaveQuery {
                name: "example".into(),
                text: "sessions".into(),
                denominator: None,
            },
        );
        let pending_hash = blake3::hash(
            &serde_json::to_vec(&(&pending.machine, &pending.authorship, &pending.write)).unwrap(),
        );
        conn.execute(&source.sql("INSERT INTO @source.client_request(id,content_hash,created_at,request,commit_seq) VALUES(?1,?2,?3,?4,3)"), params![pending.id,pending_hash.as_bytes().as_slice(),pending.at.unix_ms(),crate::encoding::pack_json(&pending).unwrap()]).unwrap();
        let before = rows(&conn, source);
        previous(&conn);
        assert!(matches!(
            crate::primary::authored::frozen_request(&conn, &prior.id),
            Err(StoreError::Conflict(_))
        ));
        reopen(&conn).unwrap();
        assert_eq!(rows(&conn, source), before);
        assert_eq!(crate::schema::version(&conn).unwrap(), ARCHIVE_VERSION);
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
        reopen(&conn).unwrap();
        assert_eq!(conn.total_changes(), changes);
        assert_eq!(rows(&conn, source), before);
    }
}

#[test]
fn request_hash_transition_refuses_corruption_without_partial_conversion() {
    {
        for corrupt in ["hash", "request identity", "request body"] {
            let (conn, source) = source();
            let first = delivery("a-valid-prior");
            let later = delivery("z-corrupt");
            insert(&conn, &first);
            insert(&conn, &later);
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
            previous(&conn);
            let before = rows(&conn, source);
            assert!(reopen(&conn).is_err(), "{corrupt}");
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
        insert(&conn, &prior);
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
        let written_note = insert(&conn, &note);
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
        let note_root: String = conn
            .query_row(
                "SELECT root FROM assertion WHERE id=?1",
                [&written_note.id],
                |row| row.get(0),
            )
            .unwrap();
        let reference = polylogue_model::Reference::Assertion(
            polylogue_model::RevisionReference::exact(note_root, written_note.id),
        );
        let evidence = reader.authored_record(&reference).unwrap();
        assert!(
            matches!(
                &evidence,
                Some(crate::read::authored_evidence::AuthoredReadEvidence::Local { .. })
            ),
            "{evidence:?}"
        );
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
    for predecessor in [1, 2] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("request-hash-fault.db");
        let conn = Connection::open(&path).unwrap();
        if predecessor == 1 {
            crate::schema::tests::archive_before_unknown_time(&conn);
        } else {
            crate::schema::ensure_archive(&conn).unwrap();
        }
        let source = crate::source::Source::of(&conn).unwrap();
        let request = delivery("old");
        insert(&conn, &request);
        let old_hash = blake3::hash(
            &serde_json::to_vec(&(&request.machine, &request.authorship, &request.write)).unwrap(),
        );
        conn.execute(
            &source.sql("UPDATE @source.client_request SET content_hash=?1 WHERE id=?2"),
            params![old_hash.as_bytes().as_slice(), request.id],
        )
        .unwrap();
        crate::schema::set_version(&conn, predecessor).unwrap();
        let before = rows(&conn, source);
        let shape_before = crate::schema::shape(&conn).unwrap();
        crate::fault::arm(
            "archive.request_hash_transition",
            conn.path().unwrap(),
            crate::fault::Action::Fail,
            0,
            1,
        );
        assert!(reopen(&conn).is_err());
        assert_eq!(crate::schema::version(&conn).unwrap(), predecessor);
        assert_eq!(crate::schema::shape(&conn).unwrap(), shape_before);
        assert_eq!(rows(&conn, source), before);
        let retained: Vec<u8> = conn
            .query_row(
                &source.sql("SELECT content_hash FROM @source.client_request WHERE id=?1"),
                [&request.id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(retained, old_hash.as_bytes());
        reopen(&conn).unwrap();
        assert!(
            crate::primary::authored::known_request(&conn, &request)
                .unwrap()
                .is_some()
        );
        assert_eq!(rows(&conn, source), before);
    }
}
