//! Physical-session display progress through the ordinary rows consumer.
use super::*;
use crate::commands::Surface;
use crate::test_daemon::{Call, TestDaemon};
use polylogue_model::{Reference, read::*, user::SeenMark};
use serde_json::json;

fn request(step: serde_json::Value) -> Call {
    Call::new(
        "rows",
        json!({"selection":"session:claude-code:s1", "steps":[step]}),
    )
}
fn page(d: &TestDaemon, step: serde_json::Value) -> ReadPage {
    crate::test_answers::read(request(step).answer(&d.daemon))
}
fn marks(d: &TestDaemon) -> Vec<SeenMark> {
    // A synchronous write drains the earlier acknowledgement, without polling.
    d.set("browser.actions", "on");
    let author = request(json!({"verb":"read"}))
        .authorship
        .author
        .to_string();
    d.daemon
        .reads
        .reader()
        .unwrap()
        .seen_marks(
            &author,
            &[Reference::parse("session:claude-code:s1").unwrap()],
        )
        .unwrap()
}
fn fixture() -> TestDaemon {
    test_daemon::start_with_sources_and_workers(
        "opening marker",
        &|home| {
            let text = (1..=8)
                .map(|n| {
                    polylogue_testkit::claude_user(
                        &format!("u{n}"),
                        "s1",
                        &format!("message {n} {}", "synthetic text ".repeat(150)),
                    )
                })
                .collect::<String>();
            std::fs::write(home.join(".claude/projects/-p/s1.jsonl"), text).unwrap();
        },
        7,
        2,
    )
}

#[test]
fn from_seen_keeps_its_accepted_bound_through_both_directions_and_reopen() {
    let mut d = fixture();
    let whole = page(
        &d,
        json!({"verb":"read", "from":"seen", "budget":100000, "hide":["annotations"]}),
    );
    assert!(whole.document.items.iter().any(|item| {
        item.source
            .message
            .as_ref()
            .is_some_and(|id| id.to_string() == "claude-code:u1")
    }));
    assert!(
        marks(&d).is_empty(),
        "a non-displayed CLI page must not mark"
    );

    // Display only the first message through the normal context selector.
    page(
        &d,
        json!({"verb":"read", "view":"context", "at":"claude-code:u1", "displayed":true, "budget":100000}),
    );
    let before = marks(&d);
    assert_eq!(before.len(), 1);
    assert_eq!(before[0].item.as_deref(), Some("block:claude-code:u1#0"));
    let first = page(
        &d,
        json!({"verb":"read", "from":"seen", "displayed":true, "budget":1000, "hide":["annotations"]}),
    );
    assert!(!first.document.items.iter().any(|item| {
        item.source
            .message
            .as_ref()
            .is_some_and(|id| id.to_string() == "claude-code:u1")
    }));
    let next = first
        .document
        .footer
        .next
        .clone()
        .expect("the fixture needs several pages");
    let advanced = marks(&d);
    assert_ne!(
        advanced[0].item, before[0].item,
        "the displayed page must advance progress"
    );
    let mut next_request = next.request.clone();
    next_request.arguments["steps"][0]["displayed"] = true.into();
    let second = crate::test_answers::read(Call::from_request(&next_request).answer(&d.daemon));
    let _ = marks(&d);
    let previous = second
        .document
        .footer
        .previous
        .as_ref()
        .expect("a second page has a previous page");
    let returned =
        crate::test_answers::read(Call::from_request(&previous.request).answer(&d.daemon));
    assert_eq!(
        returned.document.items, first.document.items,
        "newer marks cannot move a previous page's lower bound"
    );
    d.restart();
    let reopened = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
    assert_eq!(reopened.document.items, second.document.items);
    let previous = reopened.document.footer.previous.as_ref().unwrap();
    let returned =
        crate::test_answers::read(Call::from_request(&previous.request).answer(&d.daemon));
    assert_eq!(returned.document.items, first.document.items);

    let mut changed = next.request.clone();
    changed.arguments["steps"][0]
        .as_object_mut()
        .unwrap()
        .remove("from");
    assert_eq!(
        crate::test_answers::refusal(Call::from_request(&changed).answer(&d.daemon))
            .code
            .as_deref(),
        Some("invalid_query")
    );
    let mut forged = next.request;
    forged.arguments["steps"][0]["cursor"] = "forged".into();
    assert_eq!(
        crate::test_answers::refusal(Call::from_request(&forged).answer(&d.daemon))
            .code
            .as_deref(),
        Some("invalid_query")
    );
}

#[test]
fn displayed_progress_is_surface_bound_and_partial_entries_remain_unseen() {
    let d = test_daemon::start(&"unread remainder ".repeat(1000));
    for surface in [Surface::Cli, Surface::Http, Surface::Hook] {
        let displayed = surface != Surface::Cli;
        crate::test_answers::read(
            request(json!({"verb":"read", "displayed":displayed, "budget":100000}))
                .surface(surface)
                .answer(&d.daemon),
        );
        assert!(marks(&d).is_empty(), "{surface:?} cannot mark this page");
    }
    let mut next = Some(request(json!({"verb":"read", "budget":1})));
    let mut saw_fragment = false;
    // The lead-only page and an unfinished first body must not consume a message.
    while let Some(call) = next.take() {
        let page = crate::test_answers::read(call.surface(Surface::Mcp).answer(&d.daemon));
        saw_fragment |= page
            .document
            .items
            .iter()
            .any(|item| matches!(item.kind, ItemKind::Fragment { .. }));
        if saw_fragment {
            assert!(
                marks(&d).is_empty(),
                "an unfinished first entry has no completed progress"
            );
            break;
        }
        next = page
            .document
            .footer
            .next
            .as_ref()
            .map(|call| Call::from_request(&call.request));
    }
    assert!(saw_fragment);
    crate::test_answers::read(
        request(json!({"verb":"read", "budget":100000}))
            .surface(Surface::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(
        marks(&d).len(),
        1,
        "a returned MCP page marks its completed item"
    );
}
