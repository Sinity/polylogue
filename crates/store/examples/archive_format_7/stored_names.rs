//! The format-8 stored names: a format-6 archive with rows in every table
//! whose names changed converts to a file this build opens, with each row
//! as it was under its new names.
use super::*;
use rusqlite::types::Value;

/// Rows written under the current names, then given their earlier names
/// back, as a format-6 archive holds them.
const ROWS: &str = "
INSERT INTO machine(id, host, first_seen_at) VALUES ('m1', 'host', 5);
INSERT INTO input(id, origin) VALUES ('in1', 'claude-code'), ('in2', 'export-file');
INSERT INTO observation(id, input, machine, class, polylogue_version, seen_at, commit_seq, path_bytes, path)
  VALUES ('o1', 'in2', 'm1', 'deleted', 'test', 6, 1, CAST('/synthetic/export.zip' AS BLOB), '/synthetic/export.zip');
INSERT INTO observation_container(observation_id, container_input, member_position, kind, path_bytes)
  VALUES ('o1', 'in1', 3, 'zip', CAST('member.json' AS BLOB));
INSERT INTO saved_rows(id, name, query, ran_at, read_position, rows_json)
  VALUES ('r1', 'kept', 'sessions', 7, '{\"now\":7}', '{\"rows\":[]}');
INSERT INTO delivery(id, recipient_kind, sender, event, at, content_hash, bytes, body, refs_json, commit_seq)
  VALUES ('d1', 'operator', 'hook', 'session_start', 8, zeroblob(32), 0, X'', '[]', 1);
INSERT INTO delivery_event(delivery, outcome, at, detail, request, commit_seq)
  VALUES ('d1', 'delivered', 9, NULL, 'req1', 1), ('d1', 'used', 10, 'cited', 'req2', 1);
INSERT INTO chat_action(id, request_key, request_hash, request_json, stage_only, needs_approval, created_at, commit_seq)
  VALUES ('c1', 'key1', zeroblob(32), '{}', 1, 0, 11, 1);
INSERT INTO chat_action_event(action, position, at, kind, phase, request, commit_seq)
  VALUES ('c1', 1, 12, 'queued', 'queued', 'req3', 1), ('c1', 2, 13, 'claimed', 'claim', 'req4', 1);
INSERT INTO import_job(id, provider, created_at, commit_seq) VALUES ('j1', 'chatgpt', 14, 1);
INSERT INTO import_job_event(job, position, at, kind, provider_position, listed, captured, unchanged, failed, request, commit_seq)
  VALUES ('j1', 1, 15, 'started', NULL, 0, 0, 0, 0, 'req5', 1), ('j1', 2, 16, 'progress', 'page-2', 40, 30, 5, 5, 'req6', 1);
INSERT INTO annotation(id, root, kind, author, via, machine, commit_seq)
  VALUES ('a1', 'a1', 'note', 'operator', 'cli', 'm1', 1);
INSERT INTO annotation_link(annotation, role, ref) VALUES ('a1', 'about', 'session:claude-code:s1');
INSERT INTO annotation_unique_key(key_hash, id) VALUES (zeroblob(32), 'a1');
INSERT INTO alternate(id, root, source, source_hash, kind, author, via, machine, created_at, commit_seq)
  VALUES ('alt1', 'alt1', 'session:claude-code:s1', zeroblob(32), 'translation', 'operator', 'cli', 'm1', '2026-01-01T00:00:00Z', 1);
INSERT INTO alternate_part(alternate, position, source, source_hash, relation, body)
  VALUES ('alt1', 0, 'block:claude-code:s1#0', zeroblob(32), 'translates', X'00');
INSERT INTO query(hash, normal_form, object) VALUES (pl_test_query_hash(), pl_test_normal_form(), pl_test_query());
INSERT INTO query_edge(query, uses, kind, edge) VALUES (pl_test_query_hash(), 'saved:other', 'saved', X'00');
INSERT INTO signal_event(request, request_position, delivery, position, at, body, commit_seq)
  VALUES ('req7', 0, 'd1', 1, 17, pl_test_signal_event(), 1);
INSERT INTO query_run(id, binding_hash, run_hash, object_ref, at, read_position, run)
  VALUES (pl_test_run_id(), pl_test_binding_hash(), pl_test_run_hash(), 'request:req1', '18', '{\"index\":1}', pl_test_run());
";

/// Payloads the later steps decode with the build's types, already in the
/// current shape so they read back unchanged: a stored query object, a
/// signal event and an object read's run, with the run's own hashes.
fn payloads() -> Vec<(&'static str, rusqlite::types::Value)> {
    use polylogue_model::query_objects::{
        ExecutionPresentation, ObjectExecutionBinding, QueryRun, RunBinding,
    };
    use rusqlite::types::Value::{Blob, Text};
    let pack = |value: &serde_json::Value| {
        Blob(zstd::bulk::compress(&serde_json::to_vec(value).unwrap(), 1).unwrap())
    };
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../query/src/plan/tests/pre-sinex-binding.json"
    ))
    .unwrap();
    let object: serde_json::Value =
        serde_json::from_str(fixture["original_object"].as_str().unwrap()).unwrap();
    let event = serde_json::to_value(polylogue_model::signal::EventRecord {
        position: 1,
        request: "req7".into(),
        request_position: 0,
        at: polylogue_model::Timestamp::from_unix_ms(17).unwrap(),
        event: polylogue_model::signal::Event::Outlet {
            receipt: shared_signals::DeliveryReceipt {
                delivery_id: "d1".into(),
                at: 17,
                outlet: "value".into(),
                outcome: shared_signals::DeliveryOutcome::Unknown,
            },
        },
    })
    .unwrap();
    let run = QueryRun {
        id: polylogue_model::RunId::parse("0190f6a2-7c51-7a3e-9b1d-4e2f8a6c0d13").unwrap(),
        query_id: None,
        at: polylogue_model::Timestamp::from_unix_ms(18).unwrap(),
        read_position: polylogue_model::ChangePosition {
            index_instance: "synthetic".into(),
            index: 1,
            archive: 1,
        },
        machine: "m1".into(),
        authorship: polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        },
        binding: RunBinding::OriginalObject {
            execution: Box::new(ObjectExecutionBinding {
                reference: polylogue_model::Reference::parse("request:req1").unwrap(),
                content_hash: polylogue_model::BlobHash::of(b"request"),
                access_fingerprint: None,
                versions: Default::default(),
                presentation: ExecutionPresentation::Read {
                    options: Default::default(),
                    delivery: polylogue_model::read::ReadDelivery::Page,
                    format: polylogue_model::read::Format::Json,
                    page_budget: None,
                    direct_source: None,
                },
                accepted_request: None,
            }),
        },
    };
    let id: [u8; 32] = serde_json::from_value(object["id"].clone()).unwrap();
    vec![
        (
            "pl_test_query_hash",
            Text(id.iter().map(|b| format!("{b:02x}")).collect()),
        ),
        (
            "pl_test_normal_form",
            Text(object["normal_form"].to_string()),
        ),
        ("pl_test_query", pack(&object)),
        ("pl_test_signal_event", pack(&event)),
        ("pl_test_run_id", Text(run.id.to_string())),
        (
            "pl_test_binding_hash",
            Text(run.binding_hash().unwrap().to_string()),
        ),
        (
            "pl_test_run_hash",
            Text(run.record_hash().unwrap().to_string()),
        ),
        ("pl_test_run", pack(&serde_json::to_value(&run).unwrap())),
    ]
}

/// A format-6 `archive.db` in `root`: this build's tables with the names
/// format 6 used, its observation display path, and the rows above.
fn earlier_archive(root: &Path) -> Connection {
    let conn = Connection::open(root.join("archive.db")).unwrap();
    polylogue_store::schema::ensure_archive(&conn).unwrap();
    conn.execute_batch("ALTER TABLE observation ADD COLUMN path TEXT")
        .unwrap();
    for (name, value) in payloads() {
        conn.create_scalar_function(
            name,
            0,
            rusqlite::functions::FunctionFlags::SQLITE_UTF8,
            move |_| Ok(value.clone()),
        )
        .unwrap();
    }
    conn.execute_batch(ROWS).unwrap();
    for (earlier, now, columns) in RENAMED {
        // A column that takes another's earlier name is renamed after it.
        for (was, is) in columns.iter() {
            conn.execute_batch(&format!(
                "ALTER TABLE \"{now}\" RENAME COLUMN \"{is}\" TO \"{was}\""
            ))
            .unwrap();
        }
        if earlier != now {
            conn.execute_batch(&format!("ALTER TABLE \"{now}\" RENAME TO \"{earlier}\""))
                .unwrap();
        }
    }
    // A column and a table added after format 6.
    conn.execute_batch("ALTER TABLE delivery DROP COLUMN signal; DROP TABLE signal_state")
        .unwrap();
    conn.pragma_update(None, "user_version", 6).unwrap();
    conn
}

/// Every row of `table`, in its column order.
fn rows(conn: &Connection, table: &str) -> Vec<Vec<Value>> {
    let mut statement = conn
        .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY 1, 2"))
        .unwrap();
    let width = statement.column_count();
    statement
        .query_map([], |row| (0..width).map(|i| row.get(i)).collect())
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap()
}

#[test]
fn renamed_tables_keep_every_row_under_their_new_names() {
    let dir = tempfile::tempdir().unwrap();
    let conn = earlier_archive(dir.path());
    let expected: Vec<(&str, Vec<Vec<Value>>)> = RENAMED
        .iter()
        .map(|(earlier, now, _)| (*now, rows(&conn, earlier)))
        .collect();
    drop(conn);
    for (table, rows) in &expected {
        assert!(!rows.is_empty(), "the fixture has rows in {table}");
    }

    convert(dir.path()).unwrap();

    let conn = Connection::open(dir.path().join("archive.db")).unwrap();
    // The product opens it: this build's version and schema text.
    polylogue_store::schema::ensure_archive(&conn).unwrap();
    for (table, rows_before) in &expected {
        assert_eq!(&rows(&conn, table), rows_before, "{table}");
    }
    assert_eq!(
        conn.query_row(
            "SELECT outcome FROM delivery_event WHERE request = 'req2'",
            [],
            |r| r.get::<_, String>(0)
        )
        .unwrap(),
        "used"
    );
    assert_eq!(
        conn.query_row(
            "SELECT position, provider_position FROM import_job_event WHERE request = 'req6'",
            [],
            |r| <(i64, String)>::try_from(r)
        )
        .unwrap(),
        (2, "page-2".to_owned())
    );
    // What format 6 did not have is there, empty.
    assert_eq!(
        conn.query_row("SELECT id, signal FROM delivery", [], |r| {
            <(String, Option<Vec<u8>>)>::try_from(r)
        })
        .unwrap(),
        ("d1".to_owned(), None)
    );
    assert_eq!(
        conn.query_row("SELECT count(*) FROM signal_state", [], |r| r
            .get::<_, i64>(0))
            .unwrap(),
        0
    );
    drop(conn);

    // A second run finds a converted archive and changes nothing.
    let bytes = std::fs::read(dir.path().join("archive.db")).unwrap();
    let error = convert(dir.path()).unwrap_err().to_string();
    assert!(error.contains("exists already"), "{error}");
    assert_eq!(std::fs::read(dir.path().join("archive.db")).unwrap(), bytes);
}
