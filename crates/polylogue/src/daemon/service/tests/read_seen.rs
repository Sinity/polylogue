//! Physical-session display progress through the ordinary rows consumer.
use super::*;
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
    marks_of(d, "session:claude-code:s1")
}
fn marks_of(d: &TestDaemon, target: &str) -> Vec<SeenMark> {
    // A synchronous write drains the earlier acknowledgement, without polling.
    d.set("browser.actions", "on");
    let author = request(json!({"verb":"read"}))
        .caller
        .authorship
        .author
        .to_string();
    d.daemon
        .reads
        .reader()
        .unwrap()
        .seen_marks(&author, &[Reference::parse(target).unwrap()])
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

    // The page holds the first message whole and stops inside the second.
    let displayed = page(
        &d,
        json!({"verb":"read", "displayed":true, "budget":800, "hide":["annotations"]}),
    );
    assert!(displayed.document.items.iter().any(
        |item| matches!(&item.kind, ItemKind::Text { text } if text.starts_with("message 1 "))
    ));
    assert!(!displayed.document.items.iter().any(
        |item| matches!(&item.kind, ItemKind::Text { text } if text.starts_with("message 3 "))
    ));
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
    let text = "unread remainder ".repeat(2000);
    assert!(
        text.len() > PageBudget::default().bytes.get(),
        "fixture must require transfer fragments"
    );
    let d = test_daemon::start(&text);
    for surface in [
        command_surface::Via::Cli,
        command_surface::Via::Web,
        command_surface::Via::Hook,
    ] {
        let displayed = surface != command_surface::Via::Cli;
        let mut next = Some(request(
            json!({"verb":"read", "displayed":displayed, "budget":100000}),
        ));
        let mut calls = std::collections::BTreeSet::new();
        while let Some(mut call) = next.take() {
            call.arguments["steps"][0]["displayed"] = displayed.into();
            let page = crate::test_answers::read(call.channel(surface).answer(&d.daemon));
            assert!(marks(&d).is_empty(), "{surface:?} cannot mark this page");
            if let Some(ready) = page.document.footer.next {
                assert!(calls.insert(serde_json::to_string(&ready.request).unwrap()));
                next = Some(Call::from_request(&ready.request));
            }
        }
    }
    let mut next = Some(request(json!({"verb":"read", "budget":1})));
    let mut saw_fragment = false;
    // The lead-only page and an unfinished first body must not consume a message.
    while let Some(call) = next.take() {
        let page =
            crate::test_answers::read(call.channel(command_surface::Via::Mcp).answer(&d.daemon));
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
    let mut next = Some(request(json!({"verb":"read", "budget":100000})));
    let mut calls = std::collections::BTreeSet::new();
    while let Some(call) = next.take() {
        let page =
            crate::test_answers::read(call.channel(command_surface::Via::Mcp).answer(&d.daemon));
        if let Some(ready) = &page.document.footer.next {
            assert!(
                calls.insert(serde_json::to_string(&ready.request).unwrap()),
                "continuation must advance"
            );
            assert!(
                marks(&d).is_empty(),
                "unfinished message remains unseen until final MCP continuation"
            );
            next = Some(Call::from_request(&ready.request));
        }
    }
    assert!(
        !calls.is_empty(),
        "the full-budget read still exceeds its byte transfer bound"
    );
    let completed = marks(&d);
    assert_eq!(
        completed.len(),
        1,
        "the final returned MCP continuation marks the completed block"
    );
    assert_eq!(completed[0].item.as_deref(), Some("block:claude-code:u1#0"));
}

#[test]
fn inline_child_last_keeps_the_latest_rendered_physical_parent_progress() {
    let d = test_daemon::start("placeholder");
    super::reading_shapes::family(&d);
    let target = "session:claude-code:root";
    let page = crate::test_answers::read(Call::new("rows", json!({
        "selection":target,
        "steps":[{"verb":"read", "show":["spawns:full", "harness:full"], "hide":["tools"], "budget":100000}],
    })).channel(command_surface::Via::Mcp).answer(&d.daemon));
    assert!(
        page.document.footer.next.is_none(),
        "the small family must fit one page"
    );
    let text: Vec<&str> = page
        .document
        .items
        .iter()
        .filter_map(|item| {
            if let ItemKind::Text { text } = &item.kind {
                Some(text.as_str())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        text,
        ["root question", "do the subtask", "the nested subtask"],
        "the last displayed body belongs to a nested child"
    );
    let marks = marks_of(&d, target);
    assert_eq!(
        marks.len(),
        1,
        "the eligible rendered parent still advances"
    );
    assert_eq!(marks[0].item.as_deref(), Some("block:claude-code:r1#0"));
    assert!(
        marks_of(&d, "session:claude-code:agent-x1").is_empty(),
        "a child is not marked as a separate read"
    );
}

fn composed_call(target: &str, step: serde_json::Value) -> Call {
    Call::new("rows", json!({"selection":target,"steps":[step]}))
}
fn body_text(page: &ReadPage) -> Vec<String> {
    page.document
        .items
        .iter()
        .filter(|item| item.source.message.is_some())
        .filter_map(|item| match &item.kind {
            ItemKind::Text { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn logical_seen_keeps_unread_root_when_its_continuation_was_read_first() {
    let d = test_daemon::start("placeholder");
    super::reading_shapes::family(&d);
    let step = json!({"verb":"read","from":"seen","budget":100000,"hide":["tools"]});
    let continuation = crate::test_answers::read(
        composed_call("session:claude-code:cont", step.clone())
            .channel(command_surface::Via::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(body_text(&continuation), ["continuation question"]);
    assert!(marks_of(&d, "session:claude-code:root").is_empty());
    let root = crate::test_answers::read(
        composed_call("root:claude-code:root", step.clone()).answer(&d.daemon),
    );
    assert_eq!(body_text(&root), ["root question"]);
    assert!(
        marks_of(&d, "session:claude-code:root").is_empty(),
        "piped CLI does not acknowledge the root"
    );
    let root = crate::test_answers::read(
        composed_call("root:claude-code:root", step.clone())
            .channel(command_surface::Via::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(body_text(&root), ["root question"]);
    assert_eq!(
        marks_of(&d, "session:claude-code:root")[0].item.as_deref(),
        Some("block:claude-code:r1#0")
    );
    assert_eq!(
        marks_of(&d, "session:claude-code:cont")[0].item.as_deref(),
        Some("block:claude-code:c1#0")
    );
    let empty =
        crate::test_answers::read(composed_call("root:claude-code:root", step).answer(&d.daemon));
    assert!(body_text(&empty).is_empty());
}

#[test]
fn logical_seen_keeps_copied_prefix_progress_on_its_physical_owner() {
    let d = test_daemon::start("placeholder");
    super::reading_shapes::family(&d);
    let step =
        json!({"verb":"read","from":"seen","budget":100000,"show":["inherited"],"hide":["tools"]});
    // The fork's reading, one item a page, up to the end of its inherited
    // prefix: each page is returned over MCP, so it is displayed.
    let mut page = crate::test_answers::read(
        composed_call(
            "session:claude-code:fork",
            json!({"verb":"read","budget":1,"show":["inherited"],"hide":["tools","annotations"]}),
        )
        .channel(command_surface::Via::Mcp)
        .answer(&d.daemon),
    );
    let mut shown = body_text(&page);
    while !shown.iter().any(|text| text == "root question") {
        let next = page.document.footer.next.clone().expect("a next page");
        page = crate::test_answers::read(
            Call::from_request(&next.request)
                .channel(command_surface::Via::Mcp)
                .answer(&d.daemon),
        );
        shown.extend(body_text(&page));
    }
    assert_eq!(shown, ["root question"]);
    assert_eq!(
        marks_of(&d, "session:claude-code:fork")[0].item.as_deref(),
        Some("block:claude-code:r1#0")
    );
    let fork = crate::test_answers::read(
        composed_call("root:claude-code:fork", step.clone())
            .channel(command_surface::Via::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(body_text(&fork), ["forked question"]);
    assert!(
        marks_of(&d, "session:claude-code:root").is_empty(),
        "an inherited prefix belongs to the accepted fork reading"
    );
    assert_eq!(
        marks_of(&d, "session:claude-code:fork")[0].item.as_deref(),
        Some("block:claude-code:f1#0")
    );
    let again =
        crate::test_answers::read(composed_call("root:claude-code:fork", step).answer(&d.daemon));
    assert!(body_text(&again).is_empty());
    let root = crate::test_answers::read(
        composed_call(
            "root:claude-code:root",
            json!({"verb":"read","from":"seen","budget":100000,"hide":["tools"]}),
        )
        .answer(&d.daemon),
    );
    assert_eq!(body_text(&root), ["root question", "continuation question"]);
}

#[test]
fn logical_seen_cursor_keeps_all_accepted_bounds_through_parent_append_and_restart() {
    use std::io::Write;
    let mut d = test_daemon::start_with_sources_and_workers(
        "placeholder",
        &|home| {
            let dir = home.join(".claude/projects/-p");
            let root = (1..=8)
                .map(|n| {
                    polylogue_testkit::claude_user(
                        &format!("r{n}"),
                        "root",
                        &format!("root {n} {}", "synthetic text ".repeat(150)),
                    )
                })
                .collect::<String>();
            std::fs::write(
                dir.join("root.jsonl"),
                format!(
                    "{root}{}\n",
                    json!({"type":"continued-in","sessionId":"root","continuedInSessionId":"cont"})
                ),
            )
            .unwrap();
            std::fs::write(
                dir.join("cont.jsonl"),
                polylogue_testkit::claude_user("c1", "cont", "continuation body"),
            )
            .unwrap();
        },
        9,
        2,
    );
    let first = crate::test_answers::read(
        composed_call(
            "root:claude-code:root",
            json!({"verb":"read","from":"seen","budget":1000,"hide":["annotations"]}),
        )
        .channel(command_surface::Via::Mcp)
        .answer(&d.daemon),
    );
    let next = first
        .document
        .footer
        .next
        .clone()
        .expect("large root crosses pages");
    // A newer display updates both durable marks after the first cursor captured
    // their absent bounds. That cursor must continue the original population.
    let mut newer = crate::test_answers::read(
        composed_call(
            "root:claude-code:root",
            json!({"verb":"read","budget":100000,"hide":["annotations"]}),
        )
        .channel(command_surface::Via::Mcp)
        .answer(&d.daemon),
    );
    let mut accepted = std::collections::BTreeSet::new();
    while let Some(next) = newer.document.footer.next {
        assert!(accepted.insert(serde_json::to_string(&next.request).unwrap()));
        newer = crate::test_answers::read(
            Call::from_request(&next.request)
                .channel(command_surface::Via::Mcp)
                .answer(&d.daemon),
        );
    }
    assert_eq!(
        marks_of(&d, "session:claude-code:cont")[0].item.as_deref(),
        Some("block:claude-code:c1#0")
    );
    let second = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
    let mut source = std::fs::OpenOptions::new()
        .append(true)
        .open(d.home.join(".claude/projects/-p/root.jsonl"))
        .unwrap();
    writeln!(source, "{}", json!({"type":"user","uuid":"r9","sessionId":"root","timestamp":"2026-12-01T00:00:00Z","message":{"role":"user","content":"parent appended after acceptance"}})).unwrap();
    drop(source);
    d.wait_for(|_| {
        d.daemon
            .reads
            .reader()
            .unwrap()
            .message_session(&polylogue_model::MessageId::parse("claude-code:r9").unwrap())
            .unwrap()
            .is_some()
    });
    d.restart();
    let reopened = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
    assert_eq!(reopened.document.items, second.document.items);
    let previous = reopened
        .document
        .footer
        .previous
        .as_ref()
        .expect("second page has a previous page");
    let back = crate::test_answers::read(Call::from_request(&previous.request).answer(&d.daemon));
    assert_eq!(back.document.items, first.document.items);
    let mut rest = body_text(&reopened);
    let mut page = reopened;
    let mut visited = std::collections::BTreeSet::new();
    while let Some(next) = page.document.footer.next {
        assert!(visited.insert(serde_json::to_string(&next.request).unwrap()));
        page = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
        rest.extend(body_text(&page));
    }
    assert_eq!(
        rest.iter()
            .filter(|text| text.as_str() == "parent appended after acceptance")
            .count(),
        1
    );
    assert_eq!(
        rest.last().map(String::as_str),
        Some("continuation body"),
        "parent appends retain composed part order"
    );
    let fresh = crate::test_answers::read(
        composed_call(
            "root:claude-code:root",
            json!({"verb":"read","from":"seen","budget":100000,"hide":["annotations"]}),
        )
        .answer(&d.daemon),
    );
    assert_eq!(
        body_text(&fresh),
        ["parent appended after acceptance"],
        "new root content remains available even though the continuation was already read"
    );
}

#[test]
fn logical_historical_context_continuation_retains_root_without_a_copied_prefix_claim() {
    let mut d = test_daemon::start("placeholder");
    super::reading_shapes::family(&d);
    let at = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .change_position()
        .unwrap()
        .to_string();
    let step = json!({"verb":"read","view":"context","at":at,"from":"seen","budget":12,"hide":["annotations","tools"]});
    let first =
        crate::test_answers::read(composed_call("root:claude-code:root", step).answer(&d.daemon));
    let whole = crate::test_answers::read(composed_call("root:claude-code:root", json!({"verb":"read","view":"context","at":at,"from":"seen","budget":100000,"hide":["annotations","tools"]})).answer(&d.daemon));
    assert!(whole.document.items.iter().any(|item| matches!(&item.kind,
        ItemKind::Text { text } if text == &format!("Annotations as of {at}; expiry uses this read's clock."))));
    assert_eq!(
        body_text(&whole),
        ["root question", "continuation question"]
    );
    let mut items = first.document.items.clone();
    let next = first
        .document
        .footer
        .next
        .clone()
        .expect("small context needs continuation");
    d.restart();
    let mut page = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
    items.extend(page.document.items.clone());
    let mut visited = std::collections::BTreeSet::new();
    while let Some(next) = page.document.footer.next {
        assert!(visited.insert(serde_json::to_string(&next.request).unwrap()));
        page = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
        items.extend(page.document.items.clone());
    }
    assert_eq!(items, whole.document.items);
    assert!(marks_of(&d, "session:claude-code:root").is_empty());
    assert!(marks_of(&d, "session:claude-code:cont").is_empty());
}

#[test]
fn logical_seen_matches_physical_trimming_after_an_inline_spawn_and_parent_tail() {
    use std::io::Write;
    let d = test_daemon::start("placeholder");
    super::reading_shapes::family(&d);
    let mut source = std::fs::OpenOptions::new()
        .append(true)
        .open(d.home.join(".claude/projects/-p/root.jsonl"))
        .unwrap();
    writeln!(source, "{}", json!({"type":"user","uuid":"parent-tail","sessionId":"root","timestamp":"2026-03-02T00:00:00Z","message":{"role":"user","content":"parent after the inline children"}})).unwrap();
    drop(source);
    d.wait_for(|_| {
        d.daemon
            .reads
            .reader()
            .unwrap()
            .message_session(&polylogue_model::MessageId::parse("claude-code:parent-tail").unwrap())
            .unwrap()
            .is_some()
    });
    let step = json!({"verb":"read","show":["spawns:full","harness:full"],"hide":["tools"],"budget":100000});
    let displayed = crate::test_answers::read(
        composed_call("session:claude-code:root", step.clone())
            .channel(command_surface::Via::Mcp)
            .answer(&d.daemon),
    );
    assert_eq!(
        body_text(&displayed),
        [
            "root question",
            "do the subtask",
            "the nested subtask",
            "parent after the inline children"
        ]
    );
    assert_eq!(
        marks_of(&d, "session:claude-code:root")[0].item.as_deref(),
        Some("block:claude-code:parent-tail#0")
    );
    let mut seen = step;
    seen["from"] = "seen".into();
    let physical = crate::test_answers::read(
        composed_call("session:claude-code:root", seen.clone()).answer(&d.daemon),
    );
    assert!(body_text(&physical).is_empty());
    let logical =
        crate::test_answers::read(composed_call("root:claude-code:root", seen).answer(&d.daemon));
    assert_eq!(
        body_text(&logical),
        ["continuation question"],
        "a logical reading must skip the same already-read inline prefix as its physical owner"
    );
    assert!(marks_of(&d, "session:claude-code:cont").is_empty());
    assert!(marks_of(&d, "session:claude-code:agent-x1").is_empty());
}
