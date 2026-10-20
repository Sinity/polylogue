use super::*;
use crate::test_daemon;

fn read(
    d: &Daemon,
    view: &str,
    format: &str,
    budget: Option<usize>,
    cursor: Option<&str>,
) -> Response {
    handle(
        Request::Read {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            view: view.into(),
            format: format.into(),
            logical: false,
            budget,
            cursor: cursor.map(Into::into),
        },
        d,
    )
}

#[test]
fn a_read_honours_its_view_format_and_budget() {
    let w = test_daemon::start("a prompt long enough to need two pages of one token each");
    let Response::Ok { data } = read(&w.daemon, "full", "markdown", None, None) else {
        panic!("markdown read failed")
    };
    assert_eq!(data["format"], "markdown");
    assert!(
        data["content"]
            .as_str()
            .unwrap()
            .contains("a prompt long enough")
    );

    let Response::Ok { data } = read(&w.daemon, "full", "json", None, None) else {
        panic!("json read failed")
    };
    assert_eq!(data["messages"].as_array().unwrap().len(), 1);
    // A view that keeps no block of the session leaves no message.
    let Response::Ok { data } = read(&w.daemon, "actions", "json", None, None) else {
        panic!("json read failed")
    };
    assert!(data["messages"].as_array().unwrap().is_empty());

    let Response::Ok { data: first } = read(&w.daemon, "full", "json", Some(5), None) else {
        panic!("paged read failed")
    };
    let next = first["next"].as_str().expect("a second page").to_owned();
    let Response::Ok { data: second } = read(&w.daemon, "full", "json", Some(5), Some(&next))
    else {
        panic!("paged read failed")
    };
    assert_ne!(first["text"], second["text"]);

    for (view, format, cursor) in [
        ("nope", "json", None),
        ("full", "pdf", None),
        ("full", "json", Some("m0.b0.c0")),
    ] {
        assert!(
            matches!(read(&w.daemon, view, format, None, cursor), Response::Error { ref kind, .. } if kind == "bad_request"),
            "{view} {format} {cursor:?}"
        );
    }
}
