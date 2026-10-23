use super::*;
use base64::Engine as _;

fn page(d: &Daemon, session: &str, logical: bool, view: &str, cursor: Option<&str>) -> Response {
    handle(
        Request::Read {
            session: polylogue_model::SessionId::parse(session).unwrap(),
            logical,
            view: view.into(),
            format: "json".into(),
            budget: Some(2),
            cursor: cursor.map(str::to_owned),
        },
        d,
    )
}

fn next(response: Response) -> String {
    let Response::Ok { data } = response else {
        panic!("{response:?}")
    };
    data["next"].as_str().unwrap().to_owned()
}

#[test]
fn read_pages_refuse_malformed_cross_view_and_out_of_range_positions() {
    let w = test_daemon::start("a synthetic prompt long enough for many pages λ🙂");
    let token = next(page(&w.daemon, "claude-code:s1", false, "full", None));
    for malformed in ["broken", "m0.b0.c0", "m0.b0.c0.extra", "m999.b0.c0"] {
        assert!(
            matches!(page(&w.daemon, "claude-code:s1", false, "full", Some(malformed)), Response::Error { kind, .. } if kind == "bad_request"),
            "{malformed}"
        );
    }
    for (logical, view) in [(true, "full"), (false, "dialogue")] {
        assert!(
            matches!(page(&w.daemon, "claude-code:s1", logical, view, Some(&token)), Response::Error { kind, .. } if kind == "bad_request")
        );
    }
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(&token)
        .unwrap();
    let cursor: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(
        !String::from_utf8(bytes)
            .unwrap()
            .contains("synthetic prompt")
    );
    for position in [[999, 0, 0], [0, 999, 0], [0, 0, 999999]] {
        let mut changed = cursor.clone();
        changed["position"] = serde_json::json!(position);
        let bad = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&changed).unwrap());
        assert!(
            matches!(page(&w.daemon, "claude-code:s1", false, "full", Some(&bad)), Response::Error { kind, .. } if kind == "bad_request"),
            "{position:?}"
        );
    }
}

#[test]
fn read_pages_survive_unrelated_writes_and_refuse_changed_earlier_content() {
    let w = test_daemon::start("the original synthetic prompt λ🙂 has enough text");
    let token = next(page(&w.daemon, "claude-code:s1", false, "full", None));
    w.set("day_start", "06:00");
    assert!(matches!(
        page(&w.daemon, "claude-code:s1", false, "full", Some(&token)),
        Response::Ok { .. }
    ));
    let project = w.home.join(".claude/projects/-p");
    std::fs::write(
        project.join("s2.jsonl"),
        polylogue_testkit::claude_user("u2", "s2", "different session"),
    )
    .unwrap();
    w.wait_for(|data| data["store"]["sessions"] == 2 && data["store"]["search_pending"] == 0);
    assert!(
        matches!(page(&w.daemon, "claude-code:s2", false, "full", Some(&token)), Response::Error { kind, .. } if kind == "bad_request")
    );
    assert!(matches!(
        page(&w.daemon, "claude-code:s1", false, "full", Some(&token)),
        Response::Ok { .. }
    ));
    std::fs::write(
        project.join("s1.jsonl"),
        polylogue_testkit::claude_user(
            "u1",
            "s1",
            "rewritten earlier content before the continuation",
        ),
    )
    .unwrap();
    w.wait_for(|_| {
        let Response::Ok { data } = read(&w.daemon, "full", "json", None, None) else {
            return false;
        };
        data.to_string().contains("rewritten earlier")
    });
    assert!(
        matches!(page(&w.daemon, "claude-code:s1", false, "full", Some(&token)), Response::Error { kind, .. } if kind == "stale_cursor")
    );
}

#[test]
fn read_pages_refuse_positions_from_the_previous_masking_version() {
    let world = test_daemon::start("synthetic text with enough bytes to paginate");
    let token = next(page(&world.daemon, "claude-code:s1", false, "full", None));
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .unwrap();
    let mut cursor: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cursor["version"], 2);
    cursor["version"] = serde_json::json!(1);
    let old = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&cursor).unwrap());
    assert!(
        matches!(page(&world.daemon, "claude-code:s1", false, "full", Some(&old)), Response::Error { kind, .. } if kind == "bad_request")
    );
}
