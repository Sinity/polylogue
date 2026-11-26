use super::claude_account_text::{
    document, import_report, joined_items, joined_items_with_options, portable, reading, records,
};
use super::*;
use polylogue_model::bundle::Record;
use serde_json::json;

#[test]
fn hermes_observer_keeps_events_and_rebuilds_without_conversations() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let source = dir.path().join("synthetic.atof.jsonl");
    let display = "<script>synthetic()</script>\u{1b}[2J";
    let credential = "--secret=\"unfinished --token=\"synthetic-later-value\" tail";
    let raw = format!(
        "{}\n",
        json!({"atof_version":"0.1","kind":"mark","uuid":"synthetic-native-observation","name":"hermes.approval.response","timestamp":"2026-01-02T03:04:05Z","metadata":{"session_id":"default"},"data":{"approved":false,"optional":null,"text":" exact λ🙂 ","credential":credential,"display":display}})
    );
    std::fs::write(&source, &raw).unwrap();
    let first = {
        let _daemon = start_daemon(&env);
        let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
        let _ = import_report(&imported);
        let out = cli(
            &env,
            &[
                "--json",
                "--fields",
                "id,session,input,source,source_chain,source_start,source_end,data",
                "events where source:hermes",
            ],
        );
        assert!(out.status.success(), "{out:?}");
        let first = listing(json(&out));
        assert_eq!(first["count"], 1);
        let row = &first["rows"][0];
        assert!(row["session"].is_null());
        assert_eq!(row["source"], "hermes");
        assert_eq!(row["source_start"], 0);
        assert_eq!(row["source_end"], raw.len() as u64);
        assert_eq!(row["data"]["uuid"], "synthetic-native-observation");
        assert_eq!(row["data"]["data"]["approved"], false);
        assert!(row["data"]["data"]["optional"].is_null());
        assert_eq!(row["data"]["data"]["text"], " exact λ🙂 ");
        assert_eq!(row["data"]["data"]["credential"], credential);
        assert_eq!(row["data"]["data"]["display"], display);
        let exact = format!("events where id:{}", row["id"].as_str().unwrap());
        let lookup = cli(&env, &["--json", &exact]);
        assert!(lookup.status.success(), "{lookup:?}");
        assert_eq!(listing(json(&lookup))["count"], 1);
        let reference = format!("event:{}", row["id"].as_str().unwrap());
        let page = reading(&env, &reference, &[]);
        assert!(!page.document.items.is_empty());
        for format in ["terminal", "html", "markdown"] {
            let result = cli(&env, &[&reference, "read", "--format", format]);
            assert!(result.status.success(), "{format}: {result:?}");
            let rendered = String::from_utf8(result.stdout).unwrap();
            assert!(rendered.contains("synthetic-later-value"), "{rendered}");
            assert!(
                !rendered.contains('\u{1b}'),
                "raw terminal control: {rendered:?}"
            );
            if format == "html" {
                assert!(!rendered.contains("<script>synthetic()</script>"));
                assert!(rendered.contains("&lt;script&gt;"), "{rendered}");
            }
        }
        let imported_path = dir.path().join("portable-event");
        let bundle = portable(&env, &reference, &imported_path);
        let Record::Manifest(manifest) = &bundle[0] else {
            unreachable!()
        };
        assert_eq!(manifest.version, polylogue_model::bundle::VERSION);
        let events: Vec<_> = bundle
            .iter()
            .filter_map(|record| match record {
                Record::Event(event) => Some(event),
                _ => None,
            })
            .collect();
        assert_eq!(events.len(), 1);
        let event = events[0];
        assert_eq!(event.identity.id, row["id"].as_str().unwrap());
        assert_eq!(
            serde_json::to_value(&event.event.data).unwrap(),
            row["data"]
        );
        assert!(event.identity.sources.iter().any(|stated| {
            stated.input == row["input"].as_str().unwrap()
                && stated.source == "hermes"
                && stated.source_chain == row["source_chain"].as_i64().unwrap()
                && stated.source_start == Some(0)
                && stated.source_end == Some(raw.len() as u64)
        }));
        let second_archive = dir.path().join("second-archive");
        let second_env =
            crate::Fixture::new(second_archive.as_path(), home.as_path(), config.as_path());
        {
            let _second = start_daemon(&second_env);
            let result = cli(
                &second_env,
                &["--json", "import", imported_path.to_str().unwrap()],
            );
            let _ = import_report(&result);
            let imported = cli(
                &second_env,
                &[
                    "--json",
                    "--fields",
                    "id,input,source,source_chain,source_start,source_end,session,data",
                    "events",
                ],
            );
            assert!(imported.status.success(), "{imported:?}");
            let imported = listing(json(&imported));
            assert_eq!(imported["count"], 1);
            assert_ne!(imported["rows"][0]["input"], row["input"]);
            assert_eq!(imported["rows"][0]["source"], "polylogue-bundle");
            assert_eq!(imported["rows"][0]["id"], row["id"]);
            assert_eq!(imported["rows"][0]["data"], row["data"]);
            assert!(imported["rows"][0]["session"].is_null());
            let received = records(&second_env, &reference);
            let received = received
                .iter()
                .find_map(|record| match record {
                    Record::Event(event) => Some(event),
                    _ => None,
                })
                .unwrap();
            // The receiving bundle has its own record coordinates; the
            // supplied native ID and stated sources stay unchanged.
            let mut expected_event = event.event.clone();
            assert_eq!(received.event.range, Some((0, 0)));
            let receiving = &imported["rows"][0];
            assert!(received.identity.sources.iter().any(|stated| {
                stated.input == receiving["input"].as_str().unwrap()
                    && stated.source == "polylogue-bundle"
                    && stated.source_chain == receiving["source_chain"].as_i64().unwrap()
                    && stated.source_start == receiving["source_start"].as_u64()
                    && stated.source_end == receiving["source_end"].as_u64()
            }));
            // The current importer marks this event's logical position as
            // a point; its bytes are the kept JSONL item.
            let bytes = std::fs::read(imported_path.join("records.jsonl")).unwrap();
            let mut offset = 0;
            let mut item_range = None;
            for line in bytes.split_inclusive(|byte| *byte == b'\n') {
                let record: Record = serde_json::from_slice(line).unwrap();
                if matches!(&record, Record::Event(record) if record.identity.id == event.identity.id)
                {
                    assert_eq!(record, Record::Event(event.clone()));
                    item_range = Some((offset, offset + line.len()));
                }
                offset += line.len();
            }
            let (start, end) = item_range.unwrap();
            assert!(start < end);
            let pool =
                polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&second_archive))
                    .unwrap();
            let reader = pool.reader().unwrap();
            let kept = reader
                .stored_range(
                    receiving["input"].as_str().unwrap(),
                    receiving["source_chain"].as_i64().unwrap(),
                    start as u64,
                    end as u64,
                    &reader.blob_reader().unwrap(),
                )
                .unwrap();
            assert_eq!(kept, bytes[start..end]);
            let kept: Record = serde_json::from_slice(&kept).unwrap();
            assert_eq!(kept, Record::Event(event.clone()));
            expected_event.range = received.event.range;
            assert_eq!(received.event, expected_event);
            assert!(
                event
                    .identity
                    .sources
                    .iter()
                    .all(|stated| received.identity.sources.contains(stated))
            );
        }
        let joined = joined_items(&env, &reference, "16");
        assert_eq!(
            joined,
            reading(&env, &reference, &["--budget", "100000"])
                .document
                .items
        );
        let joined = serde_json::to_string(&joined).unwrap();
        assert!(joined.contains("exact λ🙂"));
        assert!(joined.contains("synthetic-later-value"));
        let rendered = cli(&env, &["events where text:λ🙂"]);
        assert!(rendered.status.success(), "{rendered:?}");
        assert!(
            String::from_utf8(rendered.stdout)
                .unwrap()
                .contains("synthetic-later-value")
        );
        let missing = cli(&env, &["--json", "events where text:no-such-text"]);
        assert_eq!(query_rows(&missing)["count"], 0, "{missing:?}");
        for unit in ["sessions", "messages"] {
            let out = cli(&env, &["--json", unit]);
            assert_eq!(query_rows(&out)["count"], 0);
        }
        assert_eq!(std::fs::read_to_string(&source).unwrap(), raw);
        let pool =
            polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
        let reader = pool.reader().unwrap();
        let stored = reader
            .query("SELECT (SELECT pl_text_at(g.body,e.content_start+e.content_length,e.data_length) FROM content_group g WHERE g.id=e.content_group) AS data FROM event e", &[])
            .unwrap();
        let data: serde_json::Value =
            serde_json::from_str(stored[0]["data"].as_str().unwrap()).unwrap();
        assert_eq!(data["data"]["credential"], credential);
        first
    };
    std::fs::remove_file(&source).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _daemon = start_daemon(&env);
    wait_rows(&env, "events where source:hermes | sort by id", |rows| {
        count_kind(rows, "hermes_approval_response") == 1
    });
    let out = cli(
        &env,
        &[
            "--json",
            "--fields",
            "id,session,input,source,source_chain,source_start,source_end,data",
            "events where source:hermes",
        ],
    );
    assert!(out.status.success(), "{out:?}");
    assert_eq!(listing(json(&out))["rows"], first["rows"]);
}

fn hermes_event_rows(env: &crate::Fixture) -> serde_json::Value {
    let out = cli(env, &["--json", "events where source:hermes | sort by id"]);
    query_rows(&out)
}
fn count_kind(rows: &serde_json::Value, kind: &str) -> usize {
    rows["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["kind"] == kind)
        .count()
}

#[test]
fn hermes_verification_restart_and_rebuild_agree() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let verification_db = home.join(".hermes/verification_evidence.db");
    std::fs::create_dir_all(verification_db.parent().unwrap()).unwrap();
    let db = rusqlite::Connection::open(&verification_db).unwrap();
    db.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO meta VALUES('schema_version','1');
        CREATE TABLE verification_events(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,session_id TEXT NOT NULL,cwd TEXT NOT NULL,root TEXT NOT NULL,command TEXT NOT NULL,canonical_command TEXT NOT NULL,kind TEXT NOT NULL,scope TEXT NOT NULL,status TEXT NOT NULL,exit_code INTEGER NOT NULL,output_summary TEXT NOT NULL);
        INSERT INTO verification_events VALUES(1,'2026-01-02T03:04:05Z','default','/neutral','/neutral','printf synthetic','printf synthetic','test','workspace','passed',0,' exact λ result ');
        CREATE TABLE verification_state(session_id TEXT,root TEXT,last_event_id INTEGER,last_edit_at TEXT,changed_paths_json TEXT,PRIMARY KEY(session_id,root));
        INSERT INTO verification_state VALUES('default','/neutral',1,NULL,'[]');").unwrap();
    drop(db);
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let before_rebuild = {
        let daemon = start_daemon(&env);
        let initial = wait_rows(&env, "events where source:hermes | sort by id", |rows| {
            count_kind(rows, "hermes_verification_event") == 1
                && count_kind(rows, "hermes_verification_state") == 1
        });
        let first = initial["rows"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["kind"] == "hermes_verification_event")
            .unwrap();
        assert!(first["session"].is_null());
        assert_eq!(first["data"]["r"]["output_summary"], " exact λ result ");
        drop(daemon);
        let db = rusqlite::Connection::open(&verification_db).unwrap();
        db.execute_batch("BEGIN;
            UPDATE meta SET value='2' WHERE key='schema_version';
            INSERT INTO verification_events VALUES(2,'2026-01-02T03:04:06Z','default','/neutral','/neutral','printf future','printf future','test','workspace','future',23,'unread future data');
            COMMIT;").unwrap();
        drop(db);
        let daemon = start_daemon(&env);
        let started = Instant::now();
        loop {
            let out = cli(
                &env,
                &[
                    "--json",
                    "dev",
                    "sql",
                    "SELECT SUM(skipped) AS n FROM parse_state",
                ],
            );
            assert!(out.status.success(), "{out:?}");
            if json(&out)[0]["n"].as_u64().is_some_and(|n| n >= 2) {
                break;
            }
            assert!(
                started.elapsed() < READY_DEADLINE,
                "future-schema row was never skipped: {out:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            count_kind(&hermes_event_rows(&env), "hermes_verification_event"),
            1
        );
        drop(daemon);
        let db = rusqlite::Connection::open(&verification_db).unwrap();
        db.execute_batch("BEGIN;
            UPDATE meta SET value='1' WHERE key='schema_version';
            INSERT INTO verification_events VALUES(3,'2026-01-02T03:04:07Z','default','/neutral','/neutral','printf later','printf later','test','workspace','failed',1,'later recorded result');
            UPDATE verification_state SET last_event_id=3,last_edit_at='2026-01-02T03:04:08Z',changed_paths_json='[\"neutral.rs\"]';
            DELETE FROM verification_events WHERE id=1;
            COMMIT;").unwrap();
        drop(db);
        let _daemon = start_daemon(&env);
        let final_rows = wait_rows(&env, "events where source:hermes | sort by id", |rows| {
            count_kind(rows, "hermes_verification_event") == 2
                && count_kind(rows, "hermes_verification_state") == 2
                && count_kind(rows, "hermes_verification_row_deleted") == 1
        });
        let observations: Vec<_> = final_rows["rows"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|row| row["kind"] == "hermes_verification_event")
            .map(|row| row["data"]["r"]["id"].as_u64().unwrap())
            .collect();
        assert!(observations.contains(&1) && observations.contains(&3));
        assert!(!observations.contains(&2));
        for unit in ["sessions", "messages"] {
            let out = cli(&env, &["--json", unit]);
            assert_eq!(query_rows(&out)["count"], 0);
        }
        final_rows
    };
    std::fs::remove_file(&verification_db).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _daemon = start_daemon(&env);
    let rebuilt = wait_rows(&env, "events where source:hermes | sort by id", |rows| {
        count_kind(rows, "hermes_verification_event") == 2
            && count_kind(rows, "hermes_verification_state") == 2
            && count_kind(rows, "hermes_verification_row_deleted") == 1
    });
    assert_eq!(rebuilt["rows"], before_rebuild["rows"]);
}

#[test]
fn hermes_atif_documents_preserve_roles_unknown_media_and_ordered_parts() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let value = json!({"schema_version":"ATIF-v1.7","trajectory_id":"native-trajectory","session_id":"default","agent":{"name":"synthetic","version":"1"},"steps":[
        {"step_id":1,"source":"user","message":[{"type":"text","text":" exact user λ🙂 "},{"type":"image","source":{"media_type":"image/png","path":"images/input.png"}}]},
        {"step_id":2,"source":"agent","llm_call_count":0,"message":[{"type":"text","text":"-----BEGIN PRIVATE KEY-----\n"},{"type":"text","text":"synthetic-atif-message-tail\n-----END PRIVATE KEY-----"}],"tool_calls":[{"tool_call_id":"native-call","function_name":"terminal","arguments":{"command":"printf synthetic"}}],"observation":{"results":[{"source_call_id":"native-call","content":[{"type":"text","text":"-----BEGIN PRIVATE KEY-----\n"},{"type":"image","source":{"media_type":"image/png","path":"images/result.png"}},{"type":"text","text":"synthetic-atif-result-tail\n-----END PRIVATE KEY-----"}]}]}}
    ]});
    let raw = serde_json::to_vec_pretty(&value).unwrap();
    let source = dir.path().join("trajectory-synthetic.json");
    std::fs::write(&source, &raw).unwrap();
    let _daemon = start_daemon(&env);
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    let _ = import_report(&imported);
    let listed = wait_rows(&env, "documents where origin:hermes | sort by id", |rows| {
        rows["count"] == 2
    });
    assert_eq!(listed["count"], 2);
    for row in listed["rows"].as_array().unwrap() {
        let reference = format!("document:{}", row["id"].as_str().unwrap());
        assert_eq!(row["purpose"], "transcript");
        assert_eq!(row["authorship"], "unknown");
        assert!(
            row["title"].is_null(),
            "Polylogue's rendering label is not a title the source gave"
        );
        let doc = document(&env, &reference);
        if doc["metadata"]["source"] == "user" {
            assert_eq!(doc["blocks"][0]["text"], " exact user λ🙂 ");
            assert_eq!(doc["blocks"][1]["direction"], "unknown");
            assert_eq!(doc["blocks"][1]["source_id"], "images/input.png");
            assert!(doc["blocks"][1].get("blob").is_none());
        } else {
            assert_eq!(doc["metadata"]["llm_call_count"], 0);
            assert_eq!(doc["blocks"][0]["text"], "-----BEGIN PRIVATE KEY-----\n");
            assert_eq!(
                doc["blocks"][1]["text"],
                "synthetic-atif-message-tail\n-----END PRIVATE KEY-----"
            );
            assert_eq!(doc["blocks"][3]["call_id"], "native-call");
            assert_eq!(doc["blocks"][3]["outcome"]["kind"], "unknown");
            // Tool output is shown in full only with `--tools full`
            // (docs/reading.md, tools).
            let joined =
                joined_items_with_options(&env, &reference, "24", &["--show", "tools:full"]);
            assert_eq!(
                joined,
                reading(
                    &env,
                    &reference,
                    &["--show", "tools:full", "--budget", "100000"]
                )
                .document
                .items
            );
            let joined = serde_json::to_string(&joined).unwrap();
            assert!(joined.contains("synthetic-atif-message-tail"));
            assert!(joined.contains("synthetic-atif-result-tail"));
        }
        let exported = records(&env, &reference);
        assert!(
            !exported
                .iter()
                .any(|r| matches!(r, Record::Session(_) | Record::Message(_)))
        );
        let documents: Vec<_> = exported
            .iter()
            .filter_map(|r| match r {
                Record::Document(record) => Some(&record.document),
                _ => None,
            })
            .collect();
        assert_eq!(documents.len(), 1);
        assert_eq!(
            serde_json::to_value(documents[0]).unwrap()["purpose"],
            "transcript"
        );
    }
    for unit in ["sessions", "messages", "tool_calls"] {
        let out = cli(&env, &["--json", unit]);
        assert_eq!(query_rows(&out)["count"], 0);
    }
    assert_eq!(std::fs::read(&source).unwrap(), raw);
    let pool = polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader
        .query("SELECT id FROM document ORDER BY id", &[])
        .unwrap();
    let mut originals = String::new();
    for row in rows {
        let id = polylogue_model::DocumentId::parse(row["id"].as_str().unwrap()).unwrap();
        originals
            .push_str(&serde_json::to_string(&reader.document(&id).unwrap().unwrap()).unwrap());
    }
    assert!(
        originals.contains("synthetic-atif-message-tail")
            && originals.contains("synthetic-atif-result-tail")
    );
}
