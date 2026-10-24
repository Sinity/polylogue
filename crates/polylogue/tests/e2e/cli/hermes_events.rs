use super::*;
use serde_json::json;

#[test]
fn hermes_observer_public_query_masks_evidence_and_rebuilds_without_conversations() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir_all(&home).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let source = dir.path().join("synthetic.atof.jsonl");
    let credential = "--secret=\"unfinished --token=\"synthetic-later-value\" tail";
    let raw = format!(
        "{}\n",
        json!({"atof_version":"0.1","kind":"mark","uuid":"synthetic-native-observation","name":"hermes.approval.response","timestamp":"2026-01-02T03:04:05Z","metadata":{"session_id":"default"},"data":{"approved":false,"optional":null,"text":" exact λ🙂 ","credential":credential}})
    );
    std::fs::write(&source, &raw).unwrap();
    let first = {
        let _daemon = start_daemon(&env);
        let imported = cli(&env, &["import", source.to_str().unwrap()]);
        assert!(imported.status.success(), "{imported:?}");
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
        let text = String::from_utf8(out.stdout.clone()).unwrap();
        assert!(!text.contains("synthetic-later-value"));
        assert!(text.contains("tail"));
        let first = json(&out);
        assert_eq!(first["count"], 1);
        let row = &first["rows"][0];
        assert!(row["session"].is_null());
        assert_eq!(row["source"], "hermes");
        assert_eq!(row["source_start"], 0);
        assert_eq!(row["source_end"], raw.len() as u64);
        assert_eq!(row["data"]["data"]["approved"], false);
        assert!(row["data"]["data"]["optional"].is_null());
        assert_eq!(row["data"]["data"]["text"], " exact λ🙂 ");
        let exact = format!("events where id:{}", row["id"].as_str().unwrap());
        let lookup = cli(&env, &["--json", &exact]);
        assert!(lookup.status.success(), "{lookup:?}");
        assert_eq!(json(&lookup)["count"], 1);
        let rendered = cli(&env, &["events where text:λ🙂"]);
        assert!(rendered.status.success(), "{rendered:?}");
        assert!(
            !String::from_utf8(rendered.stdout)
                .unwrap()
                .contains("synthetic-later-value")
        );
        for unit in ["sessions", "messages"] {
            let out = cli(&env, &["--json", unit]);
            assert!(out.status.success(), "{out:?}");
            assert_eq!(json(&out)["count"], 0);
        }
        assert_eq!(std::fs::read_to_string(&source).unwrap(), raw);
        let pool =
            polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
        let reader = pool.reader().unwrap();
        let stored = reader
            .query("SELECT pl_block(data) AS data FROM event", &[])
            .unwrap();
        let data: serde_json::Value =
            serde_json::from_str(stored[0]["data"].as_str().unwrap()).unwrap();
        assert_eq!(data["data"]["credential"], credential);
        first
    };
    std::fs::remove_file(&source).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _daemon = start_daemon(&env);
    wait_ledger(&env, |rows| {
        ledger_count(rows, "hermes_approval_response") == 1
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
    assert_eq!(json(&out)["rows"], first["rows"]);
    let target = format!("input:{}", first["rows"][0]["input"].as_str().unwrap());
    let excised = cli(&env, &["--json", "excise", &target, "--yes"]);
    assert!(excised.status.success(), "{excised:?}");
    let remaining = cli(&env, &["--json", "events"]);
    assert!(remaining.status.success(), "{remaining:?}");
    assert_eq!(json(&remaining)["count"], 0);
}

fn ledger_rows(env: &[(&str, &Path)]) -> serde_json::Value {
    let out = cli(env, &["--json", "events where source:hermes | sort by id"]);
    assert!(out.status.success(), "{out:?}");
    json(&out)
}
fn ledger_count(rows: &serde_json::Value, kind: &str) -> usize {
    rows["rows"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|row| row["kind"] == kind)
        .count()
}
fn wait_ledger(
    env: &[(&str, &Path)],
    ready: impl Fn(&serde_json::Value) -> bool,
) -> serde_json::Value {
    let began = Instant::now();
    loop {
        let rows = ledger_rows(env);
        if ready(&rows) {
            return rows;
        }
        assert!(began.elapsed() < Duration::from_secs(20), "{rows}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn hermes_verification_native_restart_append_and_fresh_rebuild_keep_the_same_observations() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let native = home.join(".hermes/verification_evidence.db");
    std::fs::create_dir_all(native.parent().unwrap()).unwrap();
    let db = rusqlite::Connection::open(&native).unwrap();
    db.execute_batch("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        INSERT INTO meta VALUES('schema_version','1');
        CREATE TABLE verification_events(id INTEGER PRIMARY KEY AUTOINCREMENT,created_at TEXT NOT NULL,session_id TEXT NOT NULL,cwd TEXT NOT NULL,root TEXT NOT NULL,command TEXT NOT NULL,canonical_command TEXT NOT NULL,kind TEXT NOT NULL,scope TEXT NOT NULL,status TEXT NOT NULL,exit_code INTEGER NOT NULL,output_summary TEXT NOT NULL);
        INSERT INTO verification_events VALUES(1,'2026-01-02T03:04:05Z','default','/neutral','/neutral','printf synthetic','printf synthetic','test','workspace','passed',0,' exact λ result ');
        CREATE TABLE verification_state(session_id TEXT,root TEXT,last_event_id INTEGER,last_edit_at TEXT,changed_paths_json TEXT,PRIMARY KEY(session_id,root));
        INSERT INTO verification_state VALUES('default','/neutral',1,NULL,'[]');").unwrap();
    drop(db);
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let before_rebuild = {
        let daemon = start_daemon(&env);
        let initial = wait_ledger(&env, |rows| {
            ledger_count(rows, "hermes_verification_event") == 1
                && ledger_count(rows, "hermes_verification_state") == 1
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
        let db = rusqlite::Connection::open(&native).unwrap();
        db.execute_batch("BEGIN;
            UPDATE meta SET value='2' WHERE key='schema_version';
            INSERT INTO verification_events VALUES(2,'2026-01-02T03:04:06Z','default','/neutral','/neutral','printf future','printf future','test','workspace','future',23,'uninterpreted future evidence');
            COMMIT;").unwrap();
        drop(db);
        let daemon = start_daemon(&env);
        let began = Instant::now();
        loop {
            let out = cli(
                &env,
                &[
                    "--json",
                    "sql",
                    "SELECT SUM(unrecognized_n) AS n FROM parse_state",
                ],
            );
            assert!(out.status.success(), "{out:?}");
            if json(&out)[0]["n"].as_u64().is_some_and(|n| n >= 2) {
                break;
            }
            assert!(began.elapsed() < Duration::from_secs(20), "{out:?}");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            ledger_count(&ledger_rows(&env), "hermes_verification_event"),
            1
        );
        drop(daemon);
        let db = rusqlite::Connection::open(&native).unwrap();
        db.execute_batch("BEGIN;
            UPDATE meta SET value='1' WHERE key='schema_version';
            INSERT INTO verification_events VALUES(3,'2026-01-02T03:04:07Z','default','/neutral','/neutral','printf later','printf later','test','workspace','failed',1,'later recorded result');
            UPDATE verification_state SET last_event_id=3,last_edit_at='2026-01-02T03:04:08Z',changed_paths_json='[\"neutral.rs\"]';
            DELETE FROM verification_events WHERE id=1;
            COMMIT;").unwrap();
        drop(db);
        let _daemon = start_daemon(&env);
        let final_rows = wait_ledger(&env, |rows| {
            ledger_count(rows, "hermes_verification_event") == 2
                && ledger_count(rows, "hermes_verification_state") == 2
                && ledger_count(rows, "hermes_verification_row_deleted") == 1
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
            assert!(out.status.success(), "{out:?}");
            assert_eq!(json(&out)["count"], 0);
        }
        final_rows
    };
    std::fs::remove_file(&native).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _daemon = start_daemon(&env);
    let rebuilt = wait_ledger(&env, |rows| {
        ledger_count(rows, "hermes_verification_event") == 2
            && ledger_count(rows, "hermes_verification_state") == 2
            && ledger_count(rows, "hermes_verification_row_deleted") == 1
    });
    assert_eq!(rebuilt["rows"], before_rebuild["rows"]);
}
