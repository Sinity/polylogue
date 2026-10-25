use super::*;
use base64::Engine as _;

fn page(d: &Daemon, session: &str, logical: bool, view: &str, cursor: Option<&str>) -> Response {
    handle(
        Request::Read {
            target: crate::daemon::ReadTarget::Session(polylogue_model::SessionId::parse(session).unwrap()),
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
fn read_pages_refuse_positions_from_the_previous_rendering_version() {
    let world = test_daemon::start("synthetic text with enough bytes to paginate");
    let token = next(page(&world.daemon, "claude-code:s1", false, "full", None));
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .unwrap();
    let mut cursor: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cursor["version"], 3);
    cursor["version"] = serde_json::json!(2);
    let old = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&cursor).unwrap());
    assert!(
        matches!(page(&world.daemon, "claude-code:s1", false, "full", Some(&old)), Response::Error { kind, .. } if kind == "bad_request")
    );
}

#[test]
fn tied_source_positions_keep_page_order_and_cursor_binding_across_index_plan_changes() {
    use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:ties",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for (id, text) in [
        ("claude-code:z-tied", "ZETA_SECOND"),
        ("claude-code:a-tied", "ALPHA_FIRST"),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    text: Some(text),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let conn = rusqlite::Connection::open(fixture.archive.index_db()).unwrap();
    conn.execute("UPDATE message SET seq=0", []).unwrap();
    let pool = fixture.reads().unwrap();
    let id = polylogue_model::SessionId::parse("claude-code:ties").unwrap();
    for logical in [false, true] {
        let mut cursor = None;
        let mut text = String::new();
        let mut changed = false;
        for _ in 0..100 {
            let response = super::super::read::run(
                &pool.reader().unwrap(),
                &id,
                super::super::read::Options {
                    view: "full",
                    format: "json",
                    logical,
                    budget: Some(2),
                    cursor: cursor.as_deref(),
                },
                &super::super::read::checkpoints::Cache::default(),
                0,
            );
            let Response::Ok { data } = response else {
                panic!("{response:?}")
            };
            text.push_str(data["text"].as_str().unwrap());
            cursor = data["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
            if !changed {
                let order = if logical { "ASC" } else { "DESC" };
                conn.execute_batch(&format!("DROP INDEX message_session; CREATE INDEX message_session ON message(session_pk,order_input_id,seq,id {order})")).unwrap();
                changed = true;
            }
        }
        assert!(changed, "control requires an actual continuation");
        assert!(cursor.is_none(), "pagination did not finish");
        assert_eq!(text.matches("ALPHA_FIRST").count(), 1);
        assert_eq!(text.matches("ZETA_SECOND").count(), 1);
        assert!(
            text.find("ALPHA_FIRST").unwrap() < text.find("ZETA_SECOND").unwrap(),
            "{text}"
        );
    }
}

#[test]
fn cached_session_pages_equal_replay_across_views_lineage_and_split_pem() {
    let world = test_daemon::start("temporary synthetic prompt");
    let lines = [
        "λ🙂 prefix -----BEGIN VENDOR PRIVATE KEY-----private-one-----END VEND",
        "OR PRIVATE KEY----- visible tail λ🙂",
        "third ordinary source message with enough text for continuation",
    ]
    .iter()
    .enumerate()
    .map(|(n, text)| polylogue_testkit::claude_user(&format!("u{n}"), "s1", text))
    .collect::<String>();
    std::fs::write(world.home.join(".claude/projects/-p/s1.jsonl"), lines).unwrap();
    world.wait_for(|data| data["store"]["messages"] == 3 && data["store"]["search_pending"] == 0);
    for logical in [false, true] {
        for view in ["full", "dialogue", "prompts", "actions"] {
            world.daemon.read_checkpoints.clear();
            let mut cursor: Option<String> = None;
            let mut joined = String::new();
            for _ in 0..500 {
                let cached = page(
                    &world.daemon,
                    "claude-code:s1",
                    logical,
                    view,
                    cursor.as_deref(),
                );
                world.daemon.read_checkpoints.clear();
                let replayed = page(
                    &world.daemon,
                    "claude-code:s1",
                    logical,
                    view,
                    cursor.as_deref(),
                );
                let (Response::Ok { data: cached }, Response::Ok { data: replayed }) =
                    (cached, replayed)
                else {
                    panic!("page failed");
                };
                assert_eq!(cached, replayed, "private cache changed a page or citation");
                joined.push_str(cached["text"].as_str().unwrap());
                cursor = cached["next"].as_str().map(str::to_owned);
                if cursor.is_none() {
                    break;
                }
            }
            assert!(cursor.is_none());
            assert!(!joined.contains("private-one"));
        }
    }
    assert!(
        world.daemon.read_checkpoints.test_counts().1 > 0,
        "control must exercise actual cache hits"
    );
    let token = next(page(&world.daemon, "claude-code:s1", false, "full", None));
    assert!(world.daemon.read_checkpoints.test_counts().0 > 0);
    world.set("day_start", "06:00");
    assert_eq!(
        world.daemon.read_checkpoints.test_counts().0,
        0,
        "service writes clear private carry"
    );
    // A forged position must fail without populating a new checkpoint.
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .unwrap();
    let mut token: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    token["position"] = serde_json::json!([999, 0, 0]);
    let forged = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&token).unwrap());
    assert!(
        matches!(page(&world.daemon, "claude-code:s1", false, "full", Some(&forged)), Response::Error { kind, .. } if kind == "bad_request")
    );
    assert_eq!(world.daemon.read_checkpoints.test_counts().0, 0);
    let _ = page(&world.daemon, "claude-code:s1", false, "full", None);
    assert!(world.daemon.read_checkpoints.test_counts().0 > 0);
    let excised = handle(
        Request::Excise {
            request_id: "private-checkpoint-excision".into(),
            target: "session:claude-code:s1".into(),
            reason: None,
            dry_run: false,
        },
        &world.daemon,
    );
    assert!(matches!(excised, Response::Ok { .. }), "{excised:?}");
    assert_eq!(world.daemon.read_checkpoints.test_counts().0, 0);
}

#[test]
fn inherited_pem_context_has_identical_cached_and_replayed_continuations() {
    use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    for (session_id, messages) in [
        (
            "claude-code:parent",
            vec![(
                "claude-code:parent-0",
                "inherited -----BEGIN VENDOR PRIVATE KEY-----hidden-----END VEND",
            )],
        ),
        (
            "claude-code:child",
            vec![
                ("claude-code:child-0", "OR PRIVATE KEY----- visible λ🙂"),
                ("claude-code:child-1", "another child message"),
            ],
        ),
    ] {
        let session = fixture
            .session(SessionFacts {
                id: session_id,
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        for (id, text) in messages {
            fixture
                .message(
                    session,
                    MessageFacts {
                        id,
                        role: "user",
                        written_by: "human",
                        text: Some(text),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }
    fixture
        .link(
            "claude-code:parent",
            "claude-code:child",
            "continuation",
            "fork",
        )
        .unwrap();
    let conn = rusqlite::Connection::open(fixture.archive.index_db()).unwrap();
    conn.execute("UPDATE link SET branch_point='claude-code:parent-0'", [])
        .unwrap();
    let pool = fixture.reads().unwrap();
    let cache = super::super::read::checkpoints::Cache::default();
    let id = polylogue_model::SessionId::parse("claude-code:child").unwrap();
    for logical in [false, true] {
        let mut cursor: Option<String> = None;
        let mut joined = String::new();
        for _ in 0..500 {
            let render = |cursor: Option<&str>| {
                super::super::read::run(
                    &pool.reader().unwrap(),
                    &id,
                    super::super::read::Options {
                        view: "full",
                        format: "json",
                        logical,
                        budget: Some(2),
                        cursor,
                    },
                    &cache,
                    cache.epoch(),
                )
            };
            let cached = render(cursor.as_deref());
            cache.clear();
            let replayed = render(cursor.as_deref());
            let (Response::Ok { data: cached }, Response::Ok { data: replayed }) =
                (cached, replayed)
            else {
                panic!("inherited page failed");
            };
            assert_eq!(cached, replayed);
            joined.push_str(cached["text"].as_str().unwrap());
            cursor = cached["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
        }
        assert!(cursor.is_none());
        assert!(joined.contains("visible λ🙂"));
        if logical {
            assert!(!joined.contains("hidden"));
        } else {
            assert!(joined.contains("OR PRIVATE KEY"));
        }
    }
    assert!(cache.test_counts().1 > 0);
}
