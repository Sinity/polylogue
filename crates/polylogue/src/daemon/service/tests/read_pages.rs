use super::*;
use command_surface::RunnableRequest as ResumeArguments;
use polylogue_model::read::{
    ItemKind, ItemPlace, ReadDelivery, ReadItem, ReadOptions, ReadPage, View,
};

fn request(
    target: &str,
    inherited: bool,
    view: View,
    budget: usize,
    cursor: Option<&ResumeArguments>,
) -> Request {
    if let Some(resume) = cursor {
        let mut resume = resume.clone();
        resume.arguments["page_tokens"] = serde_json::json!(budget);
        return crate::protocol::operations::admit(
            crate::commands::command("read").unwrap(),
            &resume,
            &crate::commands::Caller {
                surface: crate::commands::Surface::Http,
                cwd: None,
                json: true,
            },
        )
        .unwrap();
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
        cursor: None,
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
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    serde_json::from_value(data).unwrap()
}
fn refused(response: command_surface::Answer<serde_json::Value>, expected: &str) {
    assert!(
        matches!(&response, command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } if error.code.as_deref() == Some(expected)),
        "{response:?}"
    );
}
fn call_cursor(call: &polylogue_model::read::ReadyCall) -> ResumeArguments {
    let envelope: command_surface::RunnableRequest =
        serde_json::from_value(call.http.arguments.clone()).unwrap();
    assert_eq!(envelope.operation, "read");
    envelope
}
fn next(page: &ReadPage) -> Option<ResumeArguments> {
    page.document.footer.next.as_ref().map(call_cursor)
}
#[derive(Default)]
struct Text {
    value: String,
    item: String,
    position: Option<ItemPlace>,
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
                next.arguments["target"], previous.arguments["target"],
                "a continuation reads the same target"
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
/// The session's rewrite version: what a cursor binds.
fn selected_fingerprint(d: &test_daemon::TestDaemon, session: &str) -> String {
    let reader = d.daemon.reads.reader().unwrap();
    let index = reader
        .item_index(&polylogue_model::SessionId::parse(session).unwrap())
        .unwrap()
        .unwrap();
    format!("{}:{}", index.rewrite_version, index.entries.len())
}

fn replace(d: &test_daemon::TestDaemon, texts: &[(&str, &str)]) {
    let previous = selected_fingerprint(d, "claude-code:s1");
    let source = texts
        .iter()
        .map(|(id, text)| polylogue_testkit::claude_user(id, "s1", text))
        .collect::<String>();
    std::fs::write(d.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    let started = std::time::Instant::now();
    d.wait_for(|status| {
        let ready = status["store"]["messages"] == texts.len()
            && status["store"]["search_pending"] == 0
            && selected_fingerprint(d, "claude-code:s1") != previous;
        assert!(
            ready || started.elapsed() < std::time::Duration::from_secs(20),
            "replacement did not reach {} messages with no pending search work; final status: {status}",
            texts.len()
        );
        ready
    });
}

// The daemon's seeded S1 remains historical; these records belong to a fresh S2.
fn add_selected_session(d: &test_daemon::TestDaemon, texts: &[(&str, &str)]) {
    let source = texts
        .iter()
        .map(|(id, text)| polylogue_testkit::claude_user(id, "s2", text))
        .collect::<String>();
    std::fs::write(d.home.join(".claude/projects/-p/s2.jsonl"), source).unwrap();
    let total = texts.len() + 1;
    let started = std::time::Instant::now();
    d.wait_for(|status| {
        let ready = status["store"]["messages"] == total && status["store"]["search_pending"] == 0;
        assert!(
            ready || started.elapsed() < std::time::Duration::from_secs(20),
            "fresh selected session did not reach {total} archival messages; final status: {status}"
        );
        ready
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
        malformed_resume.cursor = Some(malformed.to_owned());
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
        ("show", serde_json::json!(["inherited"])),
        ("view", serde_json::json!("dialogue")),
        ("target", serde_json::json!("message:claude-code:u1")),
    ] {
        // A cursor is bound to its reading: other arguments are refused
        // when the page is read (docs/reading.md, Budgets).
        let mut changed = token.clone();
        changed.arguments[field] = value;
        let request = crate::protocol::operations::admit(command, &changed, &caller).unwrap();
        refused(handle(request, &d.daemon), "invalid_query");
    }
    let wire = token.cursor.clone().unwrap();
    for at in [0, wire.len() / 2, wire.len() - 1] {
        let mut changed = wire.as_bytes().to_vec();
        changed[at] = if changed[at] == b'a' { b'b' } else { b'a' };
        let mut changed_resume = token.clone();
        changed_resume.cursor = Some(String::from_utf8(changed).unwrap());
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
    let response = handle(
        request(
            "session:claude-code:s1",
            false,
            View::Transcript,
            1,
            Some(&token),
        ),
        &d.daemon,
    );
    assert!(
        matches!(&response, command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } if error.code.as_deref() == Some("stale_cursor")),
        "{response:?}"
    );
}

#[test]
fn native_display_order_and_original_references_survive_page_boundaries() {
    let d = test_daemon::start("placeholder");
    add_selected_session(
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
                "session:claude-code:s2",
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
        command_surface::Answer {
            outcome: command_surface::Outcome::Error(_),
            ..
        }
    ));
}

#[test]
fn hundred_thousand_message_pages_and_direct_jumps_keep_source_identity() {
    let d = test_daemon::start("placeholder");
    let path = d.home.join(".claude/projects/-p/s2.jsonl");
    {
        use std::io::Write;
        let mut source = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        for n in 0..100_000 {
            source
                .write_all(
                    polylogue_testkit::claude_user(
                        &format!("large-{n}"),
                        "s2",
                        &format!("source {n} λ🙂"),
                    )
                    .as_bytes(),
                )
                .unwrap();
        }
        source.flush().unwrap();
    }
    d.wait_for(|status| status["store"]["messages"] == 100_001);
    let selected = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .select_messages(
            &polylogue_model::SessionId::parse("claude-code:s2").unwrap(),
            false,
        )
        .unwrap()
        .len()
        .unwrap();
    assert_eq!(selected, 100_000);
    let first = page(
        &d.daemon,
        "session:claude-code:s2",
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
    let continuation = next(&first).unwrap();
    let second = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        1,
        Some(&continuation),
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

    let mut record: serde_json::Value = serde_json::from_str(&polylogue_testkit::claude_user(
        "issue-own",
        "issue-child",
        "available child body",
    ))
    .unwrap();
    record["forkedFrom"] =
        serde_json::json!({"sessionId":"missing-parent", "messageUuid":"missing-anchor"});
    std::fs::write(
        d.home.join(".claude/projects/-p/issue-child.jsonl"),
        format!("{record}\n"),
    )
    .unwrap();
    d.wait_for(|status| {
        status["store"]["messages"] == 2
            && status["store"]["search_pending"] == 0
            && status["busy"] == false
    });
    let target = "session:claude-code:issue-child";
    let whole = page(&d.daemon, target, true, View::Transcript, 4000, None);
    assert!(whole.document.footer.next.is_none());
    let boundary = whole
        .document
        .items
        .iter()
        .position(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
        .unwrap();
    assert!(boundary > 0);
    assert!(
        matches!(&whole.document.items[boundary-1].kind, ItemKind::LinkIssue { issue: polylogue_model::read::ParentIssue::MissingParent { child, parent } }
        if child.to_string() == "claude-code:issue-child" && parent.to_string() == "claude-code:missing-parent")
    );
    // Calibrate the payload cap through the public encoder and sole estimator,
    // so the next ordinary item begins exactly at the family boundary.
    let item_tokens = |item: &ReadItem| {
        let mut prefix = Vec::new();
        polylogue_render::Encoder::new(
            &whole.document.header,
            polylogue_model::read::Format::Json,
            Default::default(),
            &mut prefix,
        )
        .unwrap();
        let mut encoded = Vec::new();
        {
            let mut encoder = polylogue_render::Encoder::new(
                &whole.document.header,
                polylogue_model::read::Format::Json,
                Default::default(),
                &mut encoded,
            )
            .unwrap();
            encoder.push(item).unwrap();
        }
        assert!(encoded.starts_with(&prefix));
        let encoded = std::str::from_utf8(&encoded[prefix.len()..]).unwrap();
        polylogue_model::read::estimate_tokens(encoded)
    };
    let budget = whole.document.items[..boundary]
        .iter()
        .map(item_tokens)
        .sum();
    let first = page(&d.daemon, target, true, View::Transcript, budget, None);
    assert_eq!(first.document.items, whole.document.items[..boundary]);
    let second = page(
        &d.daemon,
        target,
        true,
        View::Transcript,
        budget,
        Some(&next(&first).unwrap()),
    );
    assert!(matches!(
        second.document.items.first().unwrap().kind,
        ItemKind::MessageHeader { .. }
    ));
    let prior = page(
        &d.daemon,
        target,
        true,
        View::Transcript,
        budget,
        Some(&call_cursor(
            second.document.footer.previous.as_ref().unwrap(),
        )),
    );
    assert_eq!(
        prior.document.items.last(),
        Some(&whole.document.items[boundary - 1])
    );
    assert!(!prior.document.items.iter().any(|item| matches!(
        item.kind,
        ItemKind::MessageHeader { .. } | ItemKind::Text { .. }
    )));
    // Footer reservation may trim the earlier session card. The fixed upper
    // content boundary must still reopen the exact next message page.
    let reopened = page(
        &d.daemon,
        target,
        true,
        View::Transcript,
        budget,
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

/// law: pl.read.page-append-stable
#[test]
fn a_cursor_on_a_live_session_continues_after_appended_messages() {
    let d = test_daemon::start("placeholder");
    let first: Vec<(String, String)> = (0..6)
        .map(|n| {
            (
                format!("live-{n}"),
                format!("message {n} before the append"),
            )
        })
        .collect();
    let borrowed: Vec<(&str, &str)> = first
        .iter()
        .map(|(id, text)| (id.as_str(), text.as_str()))
        .collect();
    add_selected_session(&d, &borrowed);
    let opening = page(
        &d.daemon,
        "session:claude-code:s2",
        false,
        View::Transcript,
        8,
        None,
    );
    let cursor = next(&opening).expect("a second page");
    let mut source = std::fs::read_to_string(d.home.join(".claude/projects/-p/s2.jsonl")).unwrap();
    for n in 6..9 {
        source.push_str(&polylogue_testkit::claude_user(
            &format!("live-{n}"),
            "s2",
            &format!("message {n} after the append"),
        ));
    }
    std::fs::write(d.home.join(".claude/projects/-p/s2.jsonl"), source).unwrap();
    d.wait_for(|status| {
        status["store"]["messages"] == 10 && status["store"]["search_pending"] == 0
    });
    let mut text = Text::default();
    text.page(&opening);
    let mut cursor = Some(cursor);
    while let Some(current) = cursor {
        let continued = page(
            &d.daemon,
            "session:claude-code:s2",
            false,
            View::Transcript,
            8,
            Some(&current),
        );
        text.page(&continued);
        cursor = next(&continued);
    }
    let mut expected = String::new();
    for n in 0..6 {
        expected.push_str(&format!("message {n} before the append"));
    }
    for n in 6..9 {
        expected.push_str(&format!("message {n} after the append"));
    }
    assert_eq!(text.value, expected);
}

/// law: pl.read.page-no-proof
#[test]
fn continuation_pages_write_nothing() {
    let d = test_daemon::start("a synthetic prompt long enough for several pages of reading");
    let first = page(
        &d.daemon,
        "session:claude-code:s1",
        false,
        View::Transcript,
        8,
        None,
    );
    // The first page's run record goes to the writer after the answer; a
    // blocking write behind it is a barrier.
    d.set("browser.actions", "on");
    let before = d.daemon.reads.reader().unwrap().change_position().unwrap();
    let mut cursor = next(&first);
    let mut pages = 0;
    while let Some(current) = cursor {
        let continued = page(
            &d.daemon,
            "session:claude-code:s1",
            false,
            View::Transcript,
            8,
            Some(&current),
        );
        pages += 1;
        cursor = next(&continued);
    }
    assert!(pages > 0);
    let after = d.daemon.reads.reader().unwrap().change_position().unwrap();
    assert_eq!(before, after);
}
