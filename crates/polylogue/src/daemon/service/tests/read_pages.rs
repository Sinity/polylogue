use super::*;
use crate::protocol::read::ResumeArguments;
use polylogue_model::read::{
    ItemKind, ReadDelivery, ReadItem, ReadOptions, ReadPage, ReadPosition, View,
};

fn request(
    target: &str,
    inherited: bool,
    view: View,
    budget: usize,
    cursor: Option<&ResumeArguments>,
) -> Request {
    if let Some(resume) = cursor {
        return Request::ReadResume {
            root_run: resume.root_run.clone(),
            cursor: resume.cursor.clone(),
            budget: Some(std::num::NonZeroU64::new(budget as u64).unwrap()),
        };
    }
    Request::Read {
        target: target.into(),
        options: ReadOptions {
            view,
            inherited,
            ..Default::default()
        },
        delivery: ReadDelivery::Page,
        format: "json".into(),
        budget: Some(budget),
        with_blobs: false,
    }
}
fn page(
    d: &Daemon,
    target: &str,
    inherited: bool,
    view: View,
    budget: usize,
    cursor: Option<&ResumeArguments>,
) -> ReadPage {
    let response = handle(request(target, inherited, view, budget, cursor), d);
    let Response::Ok { data, .. } = response else {
        panic!("{response:?}")
    };
    serde_json::from_value(data).unwrap()
}
fn refused(response: Response, expected: &str) {
    assert!(
        matches!(&response, Response::Error { kind, .. } if kind == expected),
        "{response:?}"
    );
}
fn call_cursor(call: &polylogue_model::read::ReadyCall) -> ResumeArguments {
    let envelope: command_surface::RunnableRequest =
        serde_json::from_value(call.http.arguments.clone()).unwrap();
    assert_eq!(envelope.operation, "read");
    serde_json::from_value(envelope.arguments).unwrap()
}
fn next(page: &ReadPage) -> Option<ResumeArguments> {
    page.document.footer.next.as_ref().map(call_cursor)
}
#[derive(Default)]
struct Text {
    value: String,
    item: String,
    position: Option<ReadPosition>,
}
impl Text {
    fn push(&mut self, item: &ReadItem) {
        match &item.kind {
            ItemKind::Text { text } => {
                assert!(self.item.is_empty(), "an item interrupted a JSON fragment");
                self.value.push_str(text);
            }
            ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } => {
                assert_eq!(position.character, self.item.chars().count() as u64);
                assert!(*until > position.character && *until <= *total);
                if let Some(first) = self.position {
                    assert_eq!(position.selection, first.selection);
                    assert_eq!(position.item, first.item);
                    assert_eq!(position.item_hash, first.item_hash);
                } else {
                    self.position = Some(*position);
                }
                self.item.push_str(text);
                assert_eq!(*until, self.item.chars().count() as u64);
                if until == total {
                    assert_eq!(
                        position.item_hash.0,
                        *blake3::hash(self.item.as_bytes()).as_bytes()
                    );
                    let original: ReadItem = serde_json::from_str(&self.item).unwrap();
                    self.item.clear();
                    self.position = None;
                    self.push(&original);
                }
            }
            _ => assert!(self.item.is_empty(), "an item interrupted a JSON fragment"),
        }
    }
    fn page(&mut self, page: &ReadPage) {
        for item in &page.document.items {
            self.push(item);
        }
    }
}
fn all(d: &Daemon, target: &str, inherited: bool, view: View) -> String {
    let mut cursor = None;
    let mut text = Text::default();
    loop {
        let page = page(d, target, inherited, view, 8, cursor.as_ref());
        assert!(serde_json::to_vec(&page).unwrap().len() <= 25_000);
        assert!(page.records.is_empty());
        text.page(&page);
        let continued = next(&page);
        if let (Some(previous), Some(next)) = (&cursor, &continued) {
            assert_eq!(
                next.root_run, previous.root_run,
                "continuation changed its original root"
            );
        }
        cursor = continued;
        if cursor.is_none() {
            break;
        }
    }
    assert!(
        text.item.is_empty(),
        "transfer ended inside a structured item"
    );
    text.value
}
fn replace(d: &test_daemon::TestDaemon, texts: &[(&str, &str)]) {
    let source = texts
        .iter()
        .map(|(id, text)| polylogue_testkit::claude_user(id, "s1", text))
        .collect::<String>();
    std::fs::write(d.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    d.wait_for(|status| {
        status["store"]["messages"] == texts.len() && status["store"]["search_pending"] == 0
    });
}

#[test]
fn read_pages_refuse_malformed_cross_view_and_tampered_positions() {
    let d = test_daemon::start("a synthetic prompt long enough for many pages λ🙂");
    let first = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        None,
    );
    let token = next(&first).expect("real continuation");
    for malformed in ["broken", "m0.b0.c0", "m0.b0.c0.extra", "m999.b0.c0"] {
        let mut malformed_resume = token.clone();
        malformed_resume.cursor = query_language::cursor::SealedCursor::from_wire(malformed);
        refused(
            handle(
                request(
                    "session:claude-code:s1",
                    false,
                    View::Transcript,
                    1,
                    Some(&malformed_resume),
                ),
                &d.daemon,
            ),
            "invalid_query",
        );
    }
    let command = crate::commands::command("read").unwrap();
    let caller = crate::commands::Caller {
        surface: crate::commands::Surface::Http,
        cwd: None,
        json: true,
    };
    for (field, value) in [
        ("inherited", serde_json::json!(true)),
        ("view", serde_json::json!("dialogue")),
        ("target", serde_json::json!("message:claude-code:u1")),
    ] {
        let mut arguments = serde_json::to_value(&token).unwrap();
        arguments[field] = value;
        let refused = command.build_typed.unwrap()(arguments, &caller, "closed-resume-witness");
        assert!(refused.is_err(), "resume accepted {field} override");
    }
    let wire = token.cursor.as_str();
    for at in [0, wire.len() / 2, wire.len() - 1] {
        let mut changed = wire.as_bytes().to_vec();
        changed[at] = if changed[at] == b'a' { b'b' } else { b'a' };
        let mut changed_resume = token.clone();
        changed_resume.cursor =
            query_language::cursor::SealedCursor::from_wire(String::from_utf8(changed).unwrap());
        refused(
            handle(
                request(
                    "session:claude-code:s1",
                    false,
                    View::Transcript,
                    1,
                    Some(&changed_resume),
                ),
                &d.daemon,
            ),
            "invalid_query",
        );
    }
}

#[test]
fn read_pages_survive_unrelated_writes_and_refuse_changed_earlier_content() {
    let d = test_daemon::start("the original synthetic prompt λ🙂 has enough text");
    let token = next(&page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        None,
    ))
    .unwrap();
    d.set("browser.actions", "on");
    let _ = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        Some(&token),
    );
    replace(
        &d,
        &[("u1", "rewritten earlier content before the continuation")],
    );
    assert!(
        matches!(handle(request("session:claude-code:s1", false, View::Transcript, 1, Some(&token)), &d.daemon), Response::Error { kind, .. } if kind == "stale_cursor")
    );
}

#[test]
fn native_display_order_and_original_references_survive_page_boundaries() {
    let d = test_daemon::start("placeholder");
    replace(
        &d,
        &[
            ("z-first", "ZETA_FIRST λ🙂"),
            ("a-second", "ALPHA_SECOND λ🙂"),
        ],
    );
    for inherited in [false, true] {
        assert_eq!(
            all(
                &d.daemon,
                "session:claude-code:s1",
                inherited,
                View::Transcript
            ),
            "ZETA_FIRST λ🙂ALPHA_SECOND λ🙂"
        );
    }
    assert_eq!(
        all(
            &d.daemon,
            "message:claude-code:z-first",
            false,
            View::Transcript
        ),
        "ZETA_FIRST λ🙂"
    );
    assert_eq!(
        all(
            &d.daemon,
            "block:claude-code:a-second#0",
            false,
            View::Transcript
        ),
        "ALPHA_SECOND λ🙂"
    );
}

#[test]
fn paged_views_preserve_exact_source_text_across_credential_looking_boundaries() {
    let d = test_daemon::start("placeholder");
    let texts = [
        (
            "u1",
            "λ🙂 prefix -----BEGIN VENDOR PRIVATE KEY-----private-one-----END VEND",
        ),
        ("u2", "OR PRIVATE KEY----- visible tail λ🙂"),
        ("u3", "ordinary third message"),
    ];
    replace(&d, &texts);
    let expected = texts.iter().map(|(_, text)| *text).collect::<String>();
    let source = std::fs::read(d.home.join(".claude/projects/-p/s1.jsonl")).unwrap();
    for inherited in [false, true] {
        for view in [View::Transcript, View::Dialogue] {
            assert_eq!(
                all(&d.daemon, "session:claude-code:s1", inherited, view),
                expected
            );
        }
        assert_eq!(
            all(
                &d.daemon,
                "session:claude-code:s1",
                inherited,
                View::Changes
            ),
            ""
        );
    }
    assert_eq!(
        std::fs::read(d.home.join(".claude/projects/-p/s1.jsonl")).unwrap(),
        source
    );
}

#[test]
fn direct_anchors_keep_their_own_original_text_and_refuse_missing_sources() {
    let d = test_daemon::start("placeholder");
    let source = "-----BEGIN PRIVATE KEY-----synthetic-only-----END PRIVATE KEY-----";
    replace(&d, &[("u1", "before"), ("u2", source), ("u3", "after")]);
    assert_eq!(
        all(&d.daemon, "message:claude-code:u2", false, View::Transcript),
        source
    );
    assert_eq!(
        all(&d.daemon, "block:claude-code:u2#0", false, View::Transcript),
        source
    );
    assert!(matches!(
        handle(
            request(
                "message:claude-code:missing",
                false,
                View::Transcript,
                10,
                None
            ),
            &d.daemon
        ),
        Response::Error { .. }
    ));
}

#[test]
fn hundred_thousand_message_pages_and_direct_jumps_keep_source_identity() {
    let d = test_daemon::start("placeholder");
    let path = d.home.join(".claude/projects/-p/s1.jsonl");
    {
        use std::io::Write;
        let mut source = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for n in 0..100_000 {
            source
                .write_all(
                    polylogue_testkit::claude_user(
                        &format!("large-{n}"),
                        "s1",
                        &format!("source {n} λ🙂"),
                    )
                    .as_bytes(),
                )
                .unwrap();
        }
        source.flush().unwrap();
    }
    d.wait_for(|status| status["store"]["messages"] == 100_000);
    let first = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        None,
    );
    assert!(next(&first).is_some());
    assert!(serde_json::to_vec(&first).unwrap().len() <= 25_000);
    for n in [0, 50_000, 99_999] {
        assert_eq!(
            all(
                &d.daemon,
                &format!("message:claude-code:large-{n}"),
                false,
                View::Transcript
            ),
            format!("source {n} λ🙂")
        );
    }
}

#[test]
fn outline_continuations_cover_every_reference_and_bind_to_selected_content() {
    let d = test_daemon::start("placeholder");
    let texts = (0..125)
        .map(|i| (format!("outline-{i}"), "synthetic prompt"))
        .collect::<Vec<_>>();
    replace(
        &d,
        &texts
            .iter()
            .map(|(id, text)| (id.as_str(), *text))
            .collect::<Vec<_>>(),
    );
    let request = |cursor| Request::Outline {
        session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
        with_parents: true,
        page_size: 17,
        cursor,
    };
    let mut cursor = None;
    let mut first = None;
    let mut references = Vec::new();
    loop {
        let response = handle(request(cursor), &d.daemon);
        let Response::Ok { data, .. } = response else {
            panic!("{response:?}")
        };
        references.extend(
            data["entries"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["message"].as_str().unwrap().to_owned()),
        );
        cursor = data["next"].as_str().map(str::to_owned);
        if first.is_none() {
            first = cursor.clone();
        }
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(
        references,
        (0..125)
            .map(|i| format!("claude-code:outline-{i}"))
            .collect::<Vec<_>>()
    );
    d.set("browser.actions", "on");
    assert!(matches!(
        handle(request(first.clone()), &d.daemon),
        Response::Ok { .. }
    ));
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(d.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap();
    file.write_all(polylogue_testkit::claude_user("outline-new", "s1", "new prompt").as_bytes())
        .unwrap();
    file.flush().unwrap();
    d.wait_for(|status| status["store"]["messages"] == 126);
    assert!(
        matches!(handle(request(first), &d.daemon), Response::Error { kind, .. } if kind == "stale_cursor")
    );
}

#[test]
fn previous_pages_end_at_the_current_start_and_next_reopens_it() {
    let d = test_daemon::start("a long synthetic λ🙂 prompt with enough text for several pages");
    let first = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        None,
    );
    let next = next(&first).unwrap();
    let second = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        Some(&next),
    );
    let previous = call_cursor(
        second
            .document
            .footer
            .previous
            .as_ref()
            .expect("previous page is required after a continuation"),
    );
    let prior = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        Some(&previous),
    );
    assert_eq!(prior.document.items, first.document.items);
    let reopened = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        Some(&call_cursor(prior.document.footer.next.as_ref().unwrap())),
    );
    assert_eq!(reopened.document.items, second.document.items);
}

#[test]
fn inherited_source_text_is_exact_across_parent_child_page_boundaries() {
    let parent = "inherited -----BEGIN VENDOR PRIVATE KEY-----synthetic";
    let child = "still synthetic -----END VENDOR PRIVATE KEY----- visible λ🙂";
    let d = test_daemon::start(parent);
    let mut record: serde_json::Value =
        serde_json::from_str(&polylogue_testkit::claude_user("child-u1", "child", child)).unwrap();
    record["forkedFrom"] = serde_json::json!({"sessionId":"s1", "messageUuid":"u1"});
    std::fs::write(
        d.home.join(".claude/projects/-p/child.jsonl"),
        format!("{record}\n"),
    )
    .unwrap();
    d.wait_for(|status| status["store"]["sessions"] == 2 && status["store"]["messages"] == 2);
    assert_eq!(
        all(
            &d.daemon,
            "session:claude-code:child",
            false,
            View::Transcript
        ),
        child
    );
    assert_eq!(
        all(
            &d.daemon,
            "session:claude-code:child",
            true,
            View::Transcript
        ),
        format!("{parent}{child}")
    );
}

#[test]
fn selected_latest_user_message_does_not_include_later_assistant_bodies() {
    let d = test_daemon::start("first prompt");
    let source = [
        polylogue_testkit::claude_user("u1", "s1", "first prompt"),
        polylogue_testkit::claude_assistant("a1", "s1", "earlier assistant answer"),
        polylogue_testkit::claude_user("u2", "s1", "latest human prompt λ🙂")
            .replace("2026-03-01T10:00:00Z", "2026-03-01T10:00:02Z"),
        polylogue_testkit::claude_assistant("a2", "s1", "later assistant body")
            .replace("2026-03-01T10:00:01Z", "2026-03-01T10:00:03Z"),
    ]
    .concat();
    std::fs::write(d.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    d.wait_for(|status| status["store"]["messages"] == 4);
    assert_eq!(
        all(
            &d.daemon,
            "messages where role:user | sort by at desc, id desc | limit 1",
            false,
            View::Transcript
        ),
        "latest human prompt λ🙂"
    );
}
