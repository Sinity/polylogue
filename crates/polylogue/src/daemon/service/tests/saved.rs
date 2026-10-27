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

#[test]
fn search_retries_keep_the_original_population_and_do_not_revise_newer_saved_queries() {
    let mut d = test_daemon::start("first prompt");
    let original = ok(&d, save("first", "example", "sessions"));
    let newer = ok(&d, save("second", "example", "sessions where title:absent"));
    assert_ne!(original["id"], newer["id"]);
    let latest = ok(&d, save("third", "example", "sessions where title:latest"));
    d.restart();
    let repeat = ok(&d, save("first", "example", "sessions"));
    assert_eq!(repeat["repeat"], true);
    let repeat = ok(&d, save("second", "example", "sessions where title:absent"));
    assert_eq!(repeat["repeat"], true);
    assert_eq!(
        ok(&d, query("query:example"))["definitions"][0]["revision"],
        latest["id"]
    );
    assert_eq!(ok(&d, query("query:example"))["count"], 0);
    let keep = || Request::Keep {
        request_id: "keep".into(),
        text: "sessions".into(),
        name: None,
    };
    ok(&d, keep());
    std::fs::write(
        d.home.join(".claude/projects/-p/other.jsonl"),
        polylogue_testkit::claude_user("u-other", "other", "another prompt"),
    )
    .unwrap();
    d.wait_for(|s| s["store"]["sessions"] == 2);
    assert_eq!(ok(&d, keep())["repeat"], true);
    assert_eq!(ok(&d, query("result:keep"))["count"], 1);
    assert!(
        matches!(answer(&d,Request::Keep {request_id:"keep".into(),text:"messages".into(),name:None}),Response::Error {kind,..} if kind=="conflict")
    );
}

#[test]
fn a_pipe_in_a_quoted_value_keeps_the_match_rows() {
    let d = test_daemon::start("synthetic|prompt");
    ok(
        &d,
        Request::Keep {
            request_id: "quoted-pipe".into(),
            text: "sessions where exists message(\"synthetic|prompt\")".into(),
            name: None,
        },
    );
    assert_eq!(ok(&d, query("result:quoted-pipe.matches"))["count"], 1);
}
