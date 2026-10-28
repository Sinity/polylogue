use super::*;
use base64::Engine as _;

fn page(
    d: &Daemon,
    session: &str,
    with_parents: bool,
    view: &str,
    cursor: Option<&str>,
) -> Response {
    handle(
        Request::Read {
            slices: true,
            selection: None,
            target: crate::daemon::ReadTarget::Session(
                polylogue_model::SessionId::parse(session).unwrap(),
            ),
            with_parents,
            view: view.into(),
            format: "json".into(),
            budget: Some(2),
            cursor: cursor.map(str::to_owned),
        },
        d,
    )
}

fn next(response: Response) -> String {
    let Response::Ok { data, .. } = response else {
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
    for (with_parents, view) in [(true, "full"), (false, "dialogue")] {
        assert!(
            matches!(page(&w.daemon, "claude-code:s1", with_parents, view, Some(&token)), Response::Error { kind, .. } if kind == "bad_request")
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
        let Response::Ok { data, .. } = read(&w.daemon, "full", "json", None, None) else {
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
    let daemon = test_daemon::start("synthetic text with enough bytes to paginate");
    let token = next(page(&daemon.daemon, "claude-code:s1", false, "full", None));
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .unwrap();
    let mut cursor: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(cursor["version"], 6);
    cursor["version"] = serde_json::json!(5);
    let old = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&cursor).unwrap());
    assert!(
        matches!(page(&daemon.daemon, "claude-code:s1", false, "full", Some(&old)), Response::Error { kind, .. } if kind == "bad_request")
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
    for with_parents in [false, true] {
        let mut cursor = None;
        let mut text = String::new();
        let mut changed = false;
        for _ in 0..100 {
            let response = super::super::read::run(
                &pool.reader().unwrap(),
                &id,
                super::super::read::Options {
                    slices: true,
                    selection: None,
                    view: "full",
                    format: "json",
                    with_parents,
                    budget: Some(2),
                    cursor: cursor.as_deref(),
                },
                &super::super::read::page_ends::Cache::default(),
                0,
            );
            let Response::Ok { data, .. } = response else {
                panic!("{response:?}")
            };
            text.push_str(data["text"].as_str().unwrap());
            cursor = data["next"].as_str().map(str::to_owned);
            if cursor.is_none() {
                break;
            }
            if !changed {
                let order = if with_parents { "ASC" } else { "DESC" };
                conn.execute_batch(&format!("DROP INDEX message_session; CREATE INDEX message_session ON message(session_pk,order_input_id,seq,record_position,id {order})")).unwrap();
                changed = true;
            }
        }
        assert!(changed, "the test needs a real continuation");
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
fn cached_session_pages_equal_replay_across_views_parents_and_split_pem() {
    let daemon = test_daemon::start("temporary synthetic prompt");
    let lines = [
        "λ🙂 prefix -----BEGIN VENDOR PRIVATE KEY-----private-one-----END VEND",
        "OR PRIVATE KEY----- visible tail λ🙂",
        "third ordinary source message with enough text for continuation",
    ]
    .iter()
    .enumerate()
    .map(|(n, text)| polylogue_testkit::claude_user(&format!("u{n}"), "s1", text))
    .collect::<String>();
    std::fs::write(daemon.home.join(".claude/projects/-p/s1.jsonl"), lines).unwrap();
    daemon.wait_for(|data| data["store"]["messages"] == 3 && data["store"]["search_pending"] == 0);
    for with_parents in [false, true] {
        for view in ["full", "dialogue", "prompts", "actions"] {
            daemon.daemon.page_ends.clear();
            let mut cursor: Option<String> = None;
            let mut joined = String::new();
            for _ in 0..500 {
                let cached = page(
                    &daemon.daemon,
                    "claude-code:s1",
                    with_parents,
                    view,
                    cursor.as_deref(),
                );
                daemon.daemon.page_ends.clear();
                let replayed = page(
                    &daemon.daemon,
                    "claude-code:s1",
                    with_parents,
                    view,
                    cursor.as_deref(),
                );
                let (Response::Ok { data: cached, .. }, Response::Ok { data: replayed, .. }) =
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
        daemon.daemon.page_ends.test_counts().1 > 0,
        "the test must hit the cache"
    );
    let token = next(page(&daemon.daemon, "claude-code:s1", false, "full", None));
    assert!(daemon.daemon.page_ends.test_counts().0 > 0);
    daemon.set("day_start", "06:00");
    assert_eq!(
        daemon.daemon.page_ends.test_counts().0,
        0,
        "service writes clear private masking state"
    );
    // A forged position must fail without populating a new page_end.
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(token)
        .unwrap();
    let mut token: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    token["position"] = serde_json::json!([999, 0, 0]);
    let forged = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .encode(serde_json::to_vec(&token).unwrap());
    assert!(
        matches!(page(&daemon.daemon, "claude-code:s1", false, "full", Some(&forged)), Response::Error { kind, .. } if kind == "bad_request")
    );
    assert_eq!(daemon.daemon.page_ends.test_counts().0, 0);
    let _ = page(&daemon.daemon, "claude-code:s1", false, "full", None);
    assert!(daemon.daemon.page_ends.test_counts().0 > 0);
    let excised = handle(
        Request::Excise {
            request_id: "private-page_end-excision".into(),
            target: "session:claude-code:s1".into(),
            reason: None,
            dry_run: false,
        },
        &daemon.daemon,
    );
    assert!(matches!(excised, Response::Ok { .. }), "{excised:?}");
    assert_eq!(daemon.daemon.page_ends.test_counts().0, 0);
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
                "inherited -----BEGIN VENDOR PRIVATE KEY-----hidden",
            )],
        ),
        (
            "claude-code:child",
            vec![
                (
                    "claude-code:child-0",
                    "still hidden -----END VENDOR PRIVATE KEY----- visible λ🙂",
                ),
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
    let cache = super::super::read::page_ends::Cache::default();
    let id = polylogue_model::SessionId::parse("claude-code:child").unwrap();
    for with_parents in [false, true] {
        let mut cursor: Option<String> = None;
        let mut joined = String::new();
        for _ in 0..500 {
            let render = |cursor: Option<&str>| {
                super::super::read::run(
                    &pool.reader().unwrap(),
                    &id,
                    super::super::read::Options {
                        slices: true,
                        selection: None,
                        view: "full",
                        format: "json",
                        with_parents,
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
            let (Response::Ok { data: cached, .. }, Response::Ok { data: replayed, .. }) =
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
        if with_parents {
            assert!(!joined.contains("hidden"));
        } else {
            assert!(joined.contains("still hidden -----END VENDOR PRIVATE KEY-----"));
        }
    }
    assert!(cache.test_counts().1 > 0);
}

#[test]
fn direct_anchors_restore_private_key_masking_and_refuse_unselected_messages() {
    let daemon = test_daemon::start("temporary prompt");
    let text = [
        "before -----BEGIN PRIVATE KEY-----private-first",
        "private-tail-----END PRIVATE KEY----- visible λ🙂",
        "last visible prompt",
    ]
    .iter()
    .enumerate()
    .map(|(i, text)| {
        polylogue_testkit::claude_user(
            &if i == 0 {
                "u1".into()
            } else {
                format!("anchor-{i}")
            },
            "s1",
            text,
        )
    })
    .collect::<String>();
    std::fs::write(daemon.home.join(".claude/projects/-p/s1.jsonl"), text).unwrap();
    daemon.wait_for(|data| data["store"]["messages"] == 3);
    let request = |start, view: &str| Request::Read {
        slices: true,
        target: crate::daemon::ReadTarget::Session(
            polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
        ),
        view: view.into(),
        format: "json".into(),
        with_parents: true,
        budget: Some(8192),
        cursor: None,
        selection: Some(start),
    };
    let response = handle(
        request(
            crate::daemon::PageSelection::Message(
                polylogue_model::MessageId::parse("claude-code:anchor-1").unwrap(),
            ),
            "full",
        ),
        &daemon.daemon,
    );
    let Response::Ok { data, .. } = response else {
        panic!("{response:?}")
    };
    assert_eq!(data["slices"][0]["message"]["id"], "claude-code:anchor-1");
    assert!(data["text"].as_str().unwrap().contains("visible λ🙂"));
    assert!(!data.to_string().contains("private-tail"));
    assert_eq!(data["position"]["message"], 1);
    let Response::Ok { data, .. } = handle(
        request(crate::daemon::PageSelection::Latest, "full"),
        &daemon.daemon,
    ) else {
        panic!("latest failed")
    };
    assert_eq!(data["slices"][0]["message"]["id"], "claude-code:anchor-2");
    assert_eq!(data["position"]["message"], 2);
    // An anchor whose message is hidden by this view is a valid empty read.
    assert!(matches!(
        handle(
            request(crate::daemon::PageSelection::Latest, "actions"),
            &daemon.daemon
        ),
        Response::Ok { .. }
    ));
    assert!(
        matches!(handle(request(crate::daemon::PageSelection::Message(polylogue_model::MessageId::parse("claude-code:missing").unwrap()),"full"),&daemon.daemon),Response::Error {kind,..} if kind=="not_found")
    );
}

#[test]
fn hundred_thousand_message_pages_and_jumps_preserve_identities_and_characters() {
    use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:large",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let sources = [
        "synthetic zero λ🙂 <text> ",
        "synthetic one λ🙂 <text> ",
        "synthetic two λ🙂 <text> ",
    ];
    for (n, text) in sources.iter().enumerate() {
        fixture
            .message(
                session,
                MessageFacts {
                    id: &format!("claude-code:large-{n}"),
                    role: "user",
                    written_by: "human",
                    text: Some(text),
                    seq: Some(n as i64),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let conn = rusqlite::Connection::open(fixture.archive.index_db()).unwrap();
    conn.execute_batch("WITH RECURSIVE n(i) AS (SELECT 3 UNION ALL SELECT i+1 FROM n WHERE i<99999)
        INSERT INTO message(id,session_pk,order_input_pk,order_input_id,seq,role,written_by,input_pk)
        SELECT 'claude-code:large-'||i,m.session_pk,m.order_input_pk,m.order_input_id,i,m.role,m.written_by,m.input_pk FROM n CROSS JOIN message m WHERE m.id='claude-code:large-0';
        INSERT INTO block(message_pk,session_pk,idx,kind,text_id,body)
        SELECT m.pk,m.session_pk,b.idx,b.kind,b.text_id,b.body FROM message m JOIN block b ON b.message_pk=(SELECT pk FROM message WHERE id='claude-code:large-'||(m.seq%3)) WHERE m.seq>=3;").unwrap();
    let pool = fixture.reads().unwrap();
    let id = polylogue_model::SessionId::parse("claude-code:large").unwrap();
    let cache = super::super::read::page_ends::Cache::default();
    let render = |cursor: Option<&str>, start: Option<&crate::daemon::PageSelection>| {
        super::super::read::run(
            &pool.reader().unwrap(),
            &id,
            super::super::read::Options {
                slices: true,
                view: "full",
                format: "json",
                with_parents: true,
                budget: Some(8192),
                cursor,
                selection: start,
            },
            &cache,
            cache.epoch(),
        )
    };
    let mut cursor = None;
    let mut texts = std::collections::HashMap::<String, String>::new();
    let mut page_count = 0;
    let mut last_message = None;
    loop {
        let response = render(cursor.as_deref(), None);
        let Response::Ok { data, .. } = response else {
            panic!("{response:?}")
        };
        assert_eq!(data["position"]["total_messages"], 100000);
        assert!(data["text"].as_str().unwrap().chars().count() <= 32768);
        for slice in data["slices"].as_array().unwrap() {
            let message = slice["message"]["id"].as_str().unwrap();
            let number = message
                .rsplit_once('-')
                .unwrap()
                .1
                .parse::<usize>()
                .unwrap();
            if last_message != Some(number) {
                assert_eq!(number, last_message.map_or(0, |last| last + 1));
                last_message = Some(number);
            }
            texts
                .entry(slice["message"]["id"].as_str().unwrap().into())
                .or_default()
                .push_str(slice["text"].as_str().unwrap());
        }
        cursor = data["next"].as_str().map(str::to_owned);
        page_count += 1;
        if cursor.is_none() {
            break;
        }
        assert!(page_count < 1000, "pages must advance");
    }
    assert_eq!(texts.len(), 100000);
    for (id, text) in &texts {
        let number = id.rsplit_once('-').unwrap().1.parse::<usize>().unwrap();
        assert_eq!(text, sources[number % 3]);
    }
    for (start, expected) in [
        (
            crate::daemon::PageSelection::Message(
                polylogue_model::MessageId::parse("claude-code:large-50000").unwrap(),
            ),
            "claude-code:large-50000",
        ),
        (
            crate::daemon::PageSelection::Latest,
            "claude-code:large-99999",
        ),
    ] {
        let Response::Ok { data, .. } = render(None, Some(&start)) else {
            panic!("jump failed")
        };
        assert_eq!(data["slices"][0]["message"]["id"], expected);
        let number = expected
            .rsplit_once('-')
            .unwrap()
            .1
            .parse::<usize>()
            .unwrap();
        assert_eq!(data["slices"][0]["text"], sources[number % 3]);
    }
}

#[test]
fn outline_continuations_cover_every_reference_and_bind_to_selected_content() {
    let daemon = test_daemon::start("placeholder");
    let text = (0..125)
        .map(|i| {
            polylogue_testkit::claude_user(
                &if i == 0 {
                    "u1".into()
                } else {
                    format!("outline-{i}")
                },
                "s1",
                "synthetic prompt",
            )
        })
        .collect::<String>();
    std::fs::write(daemon.home.join(".claude/projects/-p/s1.jsonl"), text).unwrap();
    daemon.wait_for(|data| data["store"]["messages"] == 125);
    let request = |cursor| Request::Outline {
        session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
        with_parents: true,
        page_size: 17,
        cursor,
    };
    let mut cursor = None;
    let mut references = Vec::new();
    let mut first_cursor = None;
    loop {
        let Response::Ok { data, .. } = handle(request(cursor), &daemon.daemon) else {
            panic!("outline failed")
        };
        references.extend(
            data["entries"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["message"].as_str().unwrap().to_owned()),
        );
        cursor = data["next"].as_str().map(str::to_owned);
        if first_cursor.is_none() {
            first_cursor = cursor.clone();
        }
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(
        references,
        (0..125)
            .map(|i| if i == 0 {
                "claude-code:u1".into()
            } else {
                format!("claude-code:outline-{i}")
            })
            .collect::<Vec<_>>()
    );
    daemon.set("day_start", "06:00");
    assert!(matches!(
        handle(request(first_cursor.clone()), &daemon.daemon),
        Response::Ok { .. }
    ));
    let mut text = std::fs::OpenOptions::new()
        .append(true)
        .open(daemon.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap();
    std::io::Write::write_all(
        &mut text,
        polylogue_testkit::claude_user("outline-new", "s1", "new prompt").as_bytes(),
    )
    .unwrap();
    daemon.wait_for(|data| data["store"]["messages"] == 126);
    assert!(
        matches!(handle(request(first_cursor),&daemon.daemon),Response::Error {kind,..} if kind=="stale_cursor")
    );
}

#[test]
fn latest_uses_visible_content_and_current_cursors_skip_hidden_bodies() {
    use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&dir.path().join("archive")).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:visible",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:visible-user",
                role: "user",
                written_by: "human",
                text: Some("last human prompt"),
                ..Default::default()
            },
        )
        .unwrap();
    let assistant = fixture
        .message(
            session,
            MessageFacts {
                id: "claude-code:visible-assistant",
                role: "assistant",
                written_by: "model",
                text: Some("hidden assistant response"),
                ..Default::default()
            },
        )
        .unwrap();
    let conn = rusqlite::Connection::open(fixture.archive.index_db()).unwrap();
    conn.execute(
        "UPDATE block SET body=x'0011' WHERE message_pk=?1",
        [assistant],
    )
    .unwrap();
    let pool = fixture.reads().unwrap();
    let id = polylogue_model::SessionId::parse("claude-code:visible").unwrap();
    let cache = super::super::read::page_ends::Cache::default();
    let read = |view, start, cursor| {
        super::super::read::run(
            &pool.reader().unwrap(),
            &id,
            super::super::read::Options {
                slices: true,
                view,
                format: "json",
                with_parents: true,
                budget: Some(8192),
                selection: start,
                cursor,
            },
            &cache,
            cache.epoch(),
        )
    };
    let latest = crate::daemon::PageSelection::Latest;
    let Response::Ok { data, .. } = read("prompts", Some(&latest), None) else {
        panic!("latest prompt decoded a hidden body")
    };
    assert_eq!(
        data["slices"][0]["message"]["id"],
        "claude-code:visible-user"
    );
    let cursor = data["cursor"].as_str().unwrap();
    let Response::Ok { data: reopened, .. } = read("prompts", None, Some(cursor)) else {
        panic!("current page cursor decoded a hidden body")
    };
    assert_eq!(data, reopened);
    let Response::Ok { data, .. } = read("actions", Some(&latest), None) else {
        panic!("empty view decoded a hidden body")
    };
    assert_eq!(data["slices"], serde_json::json!([]));
    assert!(matches!(
        read("full", Some(&latest), None),
        Response::Error { .. }
    ));
}

#[test]
fn previous_pages_end_at_the_current_start_and_next_reopens_it() {
    let daemon = test_daemon::start("placeholder");
    let source = (0..12)
        .map(|n| {
            polylogue_testkit::claude_user(
                &if n == 0 {
                    "u1".to_owned()
                } else {
                    format!("reverse-{n}")
                },
                "s1",
                &format!("prompt number {n} λ🙂"),
            )
        })
        .collect::<String>();
    std::fs::write(daemon.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    daemon.wait_for(|data| data["store"]["messages"] == 12);
    let read = |cursor, start| {
        handle(
            Request::Read {
                target: crate::daemon::ReadTarget::Session(
                    polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                ),
                slices: true,
                view: "full".into(),
                format: "json".into(),
                with_parents: false,
                budget: Some(40),
                cursor,
                selection: start,
            },
            &daemon.daemon,
        )
    };
    let Response::Ok { data: current, .. } = read(None, Some(crate::daemon::PageSelection::Latest))
    else {
        panic!("latest")
    };
    let Response::Ok { data: previous, .. } =
        read(Some(current["previous"].as_str().unwrap().into()), None)
    else {
        panic!("previous")
    };
    assert_eq!(previous["next"], current["cursor"]);
    assert!(
        !previous["text"]
            .as_str()
            .unwrap()
            .contains("prompt number 11")
    );
    assert_eq!(previous["text"].as_str().unwrap().chars().count(), 160);
    let Response::Ok { data: again, .. } =
        read(Some(previous["next"].as_str().unwrap().into()), None)
    else {
        panic!("next")
    };
    assert_eq!(again["text"], current["text"]);
    assert_eq!(again["slices"], current["slices"]);
}
