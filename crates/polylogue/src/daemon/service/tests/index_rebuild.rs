use super::*;

fn status(d: &test_daemon::TestDaemon) -> serde_json::Value {
    crate::protocol::into_value(test_daemon::call(
        &d.daemon,
        "status",
        serde_json::json!({}),
    ))
    .unwrap()
}

#[test]
fn a_daemon_over_an_index_from_another_format_rebuilds_it_and_serves() {
    let mut d = test_daemon::start("text the rebuild reads again");
    let root = d.daemon.archive.root.clone();
    let index = root.join("index.db");
    d.restart_after_stop(|| {
        rusqlite::Connection::open(&index)
            .unwrap()
            .execute_batch("PRAGMA user_version=1; CREATE TABLE synthetic_older(value)")
            .unwrap();
    });
    d.wait_for(|status| {
        status["store"]["messages"] == 1
            && status["index"]["rebuilding"]["state"] == "no"
            && status["store"]["index_rebuild"].is_object()
    });
    let rebuild = status(&d)["store"]["index_rebuild"].clone();
    assert_eq!(
        rebuild["reason"],
        serde_json::json!({"kind": "format_version", "found": 1, "wanted": polylogue_store::schema::INDEX_VERSION})
    );
    assert_ne!(rebuild["previous_schema"], rebuild["schema"], "{rebuild}");
    let found = crate::test_answers::ok(
        test_daemon::Call::new("rows", serde_json::json!({"selection": "sessions"}))
            .answer(&d.daemon),
    );
    assert_eq!(found["kind"], "listing", "{found}");
    assert_eq!(found["reply"]["count"], 1, "{found}");
    let tables: Vec<String> = rusqlite::Connection::open(&index)
        .unwrap()
        .prepare("SELECT name FROM sqlite_master WHERE name = 'synthetic_older'")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(tables, Vec::<String>::new());
}
