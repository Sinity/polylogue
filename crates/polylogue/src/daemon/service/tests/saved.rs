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
        ("broken", "cost >"),
    ] {
        match answer(&d, save("r", name, text)) {
            Response::Error { .. } => {}
            other => panic!("{name:?} {text:?}: {other:?}"),
        }
    }
}
