use super::*;

fn answer(d: &test_daemon::TestDaemon, request: Request) -> Response {
    handle(request, &d.daemon)
}

fn ok(d: &test_daemon::TestDaemon, request: Request) -> serde_json::Value {
    match answer(d, request) {
        Response::Ok { data } => data,
        other => panic!("{other:?}"),
    }
}

fn save(id: &str, name: &str, text: &str) -> Request {
    Request::Save {
        request_id: id.into(),
        name: name.into(),
        text: text.into(),
        author: Author::Operator,
    }
}

fn query(text: &str) -> Request {
    Request::Query {
        text: text.into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        fields: vec![],
    }
}

#[test]
fn a_saved_query_runs_by_name_and_saving_again_revises_it() {
    let d = test_daemon::start("first prompt");
    let first = ok(&d, save("r1", "everything", "kind:*"));
    let v = ok(&d, query("query:everything"));
    assert_eq!(v["count"], 1, "{v}");
    assert_eq!(v["definitions"][0]["revision"], first["id"]);
    let second = ok(&d, save("r2", "everything", "kind:subagent"));
    let v = ok(&d, query("query:everything"));
    assert_eq!(v["count"], 0, "the newer revision runs: {v}");
    assert_eq!(v["definitions"][0]["revision"], second["id"]);
    let notes = ok(&d, query("notes where kind:saved_query"));
    assert_eq!(notes["count"], 1, "one family, revised: {notes}");
}

#[test]
fn save_retries_keep_the_first_revision_after_the_definition_changes() {
    let d = test_daemon::start("first prompt");
    ok(&d, save("first", "everything", "sessions"));
    ok(&d, save("second", "everything", "messages"));
    for (id, text) in [("first", "sessions"), ("second", "messages")] {
        let repeated = ok(&d, save(id, "everything", text));
        assert_eq!(repeated["id"], id);
        assert_eq!(repeated["repeat"], true);
    }
    let current = ok(&d, query("query:everything"));
    assert_eq!(current["unit"], "messages");
    assert_eq!(current["definitions"][0]["revision"], "second");
    for request in [
        save("first", "everything", "documents"),
        save("first", "other", "sessions"),
    ] {
        assert!(matches!(answer(&d, request), Response::Error { kind, .. } if kind == "conflict"));
    }
}

#[test]
fn keep_retries_return_the_original_rows_after_membership_changes() {
    let d = test_daemon::start("first prompt");
    let keep = |id: &str, text: &str, name: &str| Request::Keep {
        request_id: id.into(),
        text: text.into(),
        name: Some(name.into()),
    };
    ok(&d, keep("kept", "notes", "before"));
    ok(&d, save("saved", "everything", "sessions"));
    let repeated = ok(&d, keep("kept", "notes", "before"));
    assert_eq!(repeated["repeat"], true);
    let rows = ok(&d, query("result:kept"));
    assert_eq!(rows["count"], 0);
    for request in [
        keep("kept", "sessions", "before"),
        keep("kept", "notes", "after"),
    ] {
        assert!(matches!(answer(&d, request), Response::Error { kind, .. } if kind == "conflict"));
    }
}

#[test]
fn a_save_that_query_names_could_not_say_is_refused() {
    let d = test_daemon::start("first prompt");
    for (name, text) in [
        ("two words", "kind:*"),
        ("", "kind:*"),
        ("broken", "\"unclosed"),
    ] {
        match answer(&d, save("r", name, text)) {
            Response::Error { .. } => {}
            other => panic!("{name:?} {text:?}: {other:?}"),
        }
    }
}

fn watch(id: &str, query: &str, hold: Option<&str>) -> Request {
    Request::Watch {
        request_id: id.into(),
        query: query.into(),
        name: Some("my watch".into()),
        hold: hold.map(str::to_owned),
        notify: false,
        author: Author::Operator,
    }
}

#[test]
fn a_watch_is_a_signal_rule_delivering_notices() {
    let d = test_daemon::start("first prompt");
    ok(&d, watch("w1", "sessions where live:waiting", Some("20m")));
    let rules = ok(&d, query("notes where kind:signal"));
    assert_eq!(rules["count"], 1, "{rules}");
    let value: serde_json::Value =
        serde_json::from_str(rules["rows"][0]["value"].as_str().unwrap()).unwrap();
    assert_eq!(value["query"], "sessions where live:waiting");
    assert_eq!(value["for"], "20m");
    assert_eq!(value["interrupt"], false);
    for bad in [
        watch("w2", "sessions where nonsense_field:1", None),
        watch("w3", "kind:*", Some("soon")),
    ] {
        match answer(&d, bad) {
            Response::Error { .. } => {}
            other => panic!("{other:?}"),
        }
    }
}
