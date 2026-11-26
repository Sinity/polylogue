//! Readings through the real request path: logical sessions, fork and
//! spawn cards, context differences, the links tree and following.
use super::*;
use polylogue_model::read::{Card, HeaderKind, ItemKind, ReadOptions, ReadPage, View};
use serde_json::json;

fn line(value: serde_json::Value) -> String {
    format!("{value}\n")
}
fn user(uuid: &str, session: &str, at: &str, text: &str) -> String {
    line(json!({
        "type": "user", "uuid": uuid, "sessionId": session, "timestamp": at,
        "message": {"role": "user", "content": text}
    }))
}

/// A root with a spawned agent call and a continuation, the spawned agent
/// with a nested spawn of its own, a fork of the root, the continuation
/// itself, a session with a compaction, and one with a microcompaction.
pub(super) fn family(d: &test_daemon::TestDaemon) {
    let dir = d.home.join(".claude/projects/-p");
    let root = [
        user("r1", "root", "2026-03-01T10:00:00Z", "root question"),
        line(json!({
            "type": "assistant", "uuid": "r2", "parentUuid": "r1", "sessionId": "root",
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": "msg_r2", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "call-1", "name": "Task",
                             "input": {"description": "check the parser", "prompt": "check it"}},
                            {"type": "tool_use", "id": "call-1b", "name": "Bash",
                             "input": {"command": "ls after the spawn"}}]}
        })),
        line(json!({
            "type": "user", "uuid": "r3", "parentUuid": "r2", "sessionId": "root",
            "timestamp": "2026-03-01T10:00:02Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "checked"}]},
            "toolUseResult": {"agentId": "x1"}
        })),
        line(json!({"type": "continued-in", "sessionId": "root", "continuedInSessionId": "cont"})),
    ]
    .concat();
    std::fs::write(dir.join("root.jsonl"), root).unwrap();
    let agents = dir.join("root/subagents");
    std::fs::create_dir_all(&agents).unwrap();
    let agent = [
        line(json!({
            "type": "user", "uuid": "a1", "sessionId": "root", "agentId": "x1", "isSidechain": true,
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"role": "user", "content": "do the subtask"}
        })),
        line(json!({
            "type": "assistant", "uuid": "a2", "parentUuid": "a1", "sessionId": "root",
            "agentId": "x1", "isSidechain": true, "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": "msg_a2", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "call-2", "name": "Task",
                             "input": {"description": "nested check", "prompt": "nested"}}]}
        })),
        line(json!({
            "type": "user", "uuid": "a3", "parentUuid": "a2", "sessionId": "root",
            "agentId": "x1", "isSidechain": true, "timestamp": "2026-03-01T10:00:01Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-2", "content": "nested done"}]},
            "toolUseResult": {"agentId": "x2"}
        })),
    ]
    .concat();
    std::fs::write(agents.join("agent-x1.jsonl"), agent).unwrap();
    std::fs::write(
        agents.join("agent-x2.jsonl"),
        line(json!({
            "type": "user", "uuid": "n1", "sessionId": "root", "agentId": "x2", "isSidechain": true,
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"role": "user", "content": "the nested subtask"}
        })),
    )
    .unwrap();
    std::fs::write(
        dir.join("cont.jsonl"),
        user(
            "c1",
            "cont",
            "2026-03-01T11:00:00Z",
            "continuation question",
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("fork.jsonl"),
        line(json!({
            "type": "user", "uuid": "f1", "sessionId": "fork", "timestamp": "2026-03-01T12:00:00Z",
            "message": {"role": "user", "content": "forked question"},
            "forkedFrom": {"sessionId": "root", "messageUuid": "r1"}
        })),
    )
    .unwrap();
    let compacted = [
        user("m1", "cmp", "2026-03-01T13:00:00Z", "before compaction"),
        line(json!({
            "type": "system", "subtype": "compact_boundary", "uuid": "b1", "sessionId": "cmp",
            "logicalParentUuid": "m1", "timestamp": "2026-03-01T13:00:01Z",
            "compactMetadata": {"trigger": "auto", "preTokens": 182000}
        })),
        line(json!({
            "type": "user", "uuid": "s1", "sessionId": "cmp", "isCompactSummary": true,
            "timestamp": "2026-03-01T13:00:02Z",
            "message": {"role": "user", "content": "summary of the earlier work"}
        })),
        user("m2", "cmp", "2026-03-01T13:00:03Z", "after compaction"),
    ]
    .concat();
    std::fs::write(dir.join("cmp.jsonl"), compacted).unwrap();
    let micro = [
        user(
            "u1",
            "micro",
            "2026-03-01T14:00:00Z",
            "before microcompaction",
        ),
        line(json!({
            "type": "assistant", "uuid": "u2", "parentUuid": "u1", "sessionId": "micro",
            "timestamp": "2026-03-01T14:00:01Z",
            "message": {"id": "msg_u2", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "call-m", "name": "Bash",
                             "input": {"command": "ls"}}]}
        })),
        line(json!({
            "type": "user", "uuid": "u3", "parentUuid": "u2", "sessionId": "micro",
            "timestamp": "2026-03-01T14:00:02Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-m", "content": "cleared listing"}]}
        })),
        line(json!({
            "type": "system", "subtype": "microcompact_boundary", "uuid": "u4", "parentUuid": "u3",
            "sessionId": "micro", "timestamp": "2026-03-01T14:00:03Z",
            "content": "Context microcompacted",
            "microcompactMetadata": {"trigger": "auto", "preTokens": 100, "tokensSaved": 50,
                "compactedToolIds": ["call-m"], "clearedAttachmentUUIDs": []}
        })),
        user(
            "u5",
            "micro",
            "2026-03-01T14:00:04Z",
            "after microcompaction",
        ),
    ]
    .concat();
    std::fs::write(dir.join("micro.jsonl"), micro).unwrap();
    d.wait_for(|status| status["store"]["sessions"] == 8 && status["store"]["search_pending"] == 0);
}

fn read(d: &test_daemon::TestDaemon, target: &str, options: ReadOptions) -> ReadPage {
    let response = (crate::test_daemon::read_call(polylogue_reading::ReadRequest {
        start: Default::default(),
        references: None,
        target: target.into(),
        options,
        delivery: polylogue_model::read::ReadDelivery::Page,
        format: polylogue_model::read::Format::Json,
        budget: polylogue_model::read::ReadBudget::Tokens(100_000),
        alternates: Vec::new(),
        cursor: None,
    }))
    .answer(&d.daemon);
    crate::test_answers::read(response)
}

fn texts(page: &ReadPage) -> Vec<String> {
    page.document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Text { text } => Some(text.clone()),
            ItemKind::Indicator { text, .. } => Some(text.clone()),
            ItemKind::Omission { why, .. } => Some(why.clone()),
            _ => None,
        })
        .collect()
}

fn cards(page: &ReadPage) -> Vec<&Card> {
    page.document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Card { card } => Some(card.as_ref()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_logical_read_follows_the_root_with_its_continuation() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let logical = texts(&read(&d, "root:claude-code:root", ReadOptions::default()));
    let root = logical
        .iter()
        .position(|text| text == "root question")
        .unwrap();
    let continues = logical
        .iter()
        .position(|text| text.starts_with("continues claude-code:root"))
        .expect("a continues line");
    let continued = logical
        .iter()
        .position(|text| text == "continuation question")
        .expect("the continuation's message");
    assert!(root < continues && continues < continued, "{logical:?}");
}

#[test]
fn forks_and_spawns_are_cards_at_their_branch_point_and_call() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let page = read(&d, "session:claude-code:root", ReadOptions::default());
    let shown = cards(&page);
    assert!(shown.iter().any(|card| matches!(card,
        Card::Delegation { card } if card.child == "session:claude-code:agent-x1"
            && card.objective.value().map(String::as_str) == Some("check the parser"))));
    assert!(shown.iter().any(|card| matches!(card,
        Card::Session { session, .. } if session.id.to_string() == "claude-code:fork")));
    let hidden = read(
        &d,
        "session:claude-code:root",
        ReadOptions {
            hide: vec!["spawns".into()],
            ..ReadOptions::default()
        },
    );
    assert!(
        !cards(&hidden)
            .iter()
            .any(|card| matches!(card, Card::Delegation { .. }))
    );
}

#[test]
fn the_context_view_and_its_difference_across_a_compaction() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let context = texts(&read(
        &d,
        "session:claude-code:cmp",
        ReadOptions {
            view: View::Context,
            ..ReadOptions::default()
        },
    ));
    assert!(
        context.iter().any(|text| text == "after compaction"),
        "{context:?}"
    );
    assert!(
        !context.iter().any(|text| text == "before compaction"),
        "{context:?}"
    );
    assert!(
        context.iter().any(|text| text == "removed from context"),
        "{context:?}"
    );
    let difference = texts(&read(
        &d,
        "session:claude-code:cmp",
        ReadOptions {
            view: View::Context,
            diff: Some("compaction".into()),
            ..ReadOptions::default()
        },
    ));
    assert!(
        difference
            .iter()
            .any(|text| text == "left the context between the two places"),
        "{difference:?}"
    );
    assert!(
        difference.iter().any(|text| text == "after compaction"),
        "{difference:?}"
    );
    assert!(
        !difference.iter().any(|text| text == "before compaction"),
        "{difference:?}"
    );
}

#[test]
fn the_links_view_is_a_tree_from_the_root_with_the_seed_marked() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let page = read(
        &d,
        "session:claude-code:cont",
        ReadOptions {
            view: View::Tree,
            ..ReadOptions::default()
        },
    );
    let nodes: Vec<(String, bool)> = cards(&page)
        .into_iter()
        .filter_map(|card| match card {
            Card::LinkNode {
                reference, asked, ..
            } => Some((reference.clone(), *asked)),
            _ => None,
        })
        .collect();
    assert_eq!(
        nodes.first().map(|(node, _)| node.as_str()),
        Some("session:claude-code:root")
    );
    for expected in [
        "session:claude-code:cont",
        "session:claude-code:fork",
        "session:claude-code:agent-x1",
    ] {
        assert_eq!(
            nodes.iter().filter(|(node, _)| node == expected).count(),
            1,
            "{expected} once in {nodes:?}"
        );
    }
    assert_eq!(
        nodes
            .iter()
            .filter(|(_, asked)| *asked)
            .map(|(node, _)| node.as_str())
            .collect::<Vec<_>>(),
        ["session:claude-code:cont"]
    );
}

#[test]
fn the_last_page_follows_what_the_session_appends() {
    let d = test_daemon::start("placeholder");
    family(&d);
    // One page holds the whole session, so it is the last.
    let last = read(&d, "session:claude-code:cont", ReadOptions::default());
    let follow = last.document.footer.follow.clone().expect("a follow call");
    let before = crate::test_daemon::Call::new("status", serde_json::json!({}))
        .answer(&d.daemon)
        .value()
        .unwrap()["store"]["messages"]
        .as_u64()
        .unwrap();
    let path = d.home.join(".claude/projects/-p/cont.jsonl");
    let mut source = std::fs::read_to_string(&path).unwrap();
    source.push_str(&user(
        "c2",
        "cont",
        "2026-03-01T11:00:05Z",
        "appended question",
    ));
    std::fs::write(&path, source).unwrap();
    d.wait_for(|status| {
        status["store"]["messages"]
            .as_u64()
            .is_some_and(|n| n > before)
            && status["store"]["search_pending"] == 0
    });
    let envelope: command_surface::RunnableRequest = follow.request;
    let call = crate::test_daemon::Call::from_request(&envelope);
    let response = call.answer(&d.daemon);
    let page = crate::test_answers::read(response);
    let texts = texts(&page);
    assert_eq!(
        texts
            .iter()
            .filter(|text| text.as_str() == "appended question")
            .count(),
        1,
        "{texts:?}"
    );
    assert!(!texts.iter().any(|text| text == "continuation question"));
}

#[test]
fn a_logical_read_has_the_logical_header_and_physical_reads_the_root_alone() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let logical = read(&d, "root:claude-code:root", ReadOptions::default());
    let HeaderKind::Logical {
        root,
        physical_sessions,
        own_messages,
        inherited_messages,
    } = &logical.document.header.kind
    else {
        panic!("{:?}", logical.document.header.kind)
    };
    assert_eq!(root.to_string(), "claude-code:root");
    assert_eq!(
        physical_sessions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["claude-code:root", "claude-code:cont"]
    );
    assert_eq!(own_messages.value(), Some(&4));
    assert_eq!(inherited_messages.value(), Some(&0));
    let physical = read(
        &d,
        "root:claude-code:root",
        ReadOptions {
            physical: true,
            ..ReadOptions::default()
        },
    );
    assert!(matches!(
        physical.document.header.kind,
        HeaderKind::Session { .. }
    ));
    let texts = texts(&physical);
    assert!(texts.iter().any(|text| text == "root question"));
    assert!(!texts.iter().any(|text| text == "continuation question"));
}

/// Spawned sessions inline at their call, each after its card, to the
/// asked depth; below it they stay cards.
#[test]
fn spawns_read_inline_at_their_call_to_the_asked_depth() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let carded = texts(&read(
        &d,
        "session:claude-code:root",
        ReadOptions::default(),
    ));
    assert!(!carded.iter().any(|text| text == "do the subtask"));
    let one = read(
        &d,
        "session:claude-code:root",
        ReadOptions {
            show: vec!["spawns:1".into()],
            ..ReadOptions::default()
        },
    );
    let one_texts = texts(&one);
    let question = one_texts
        .iter()
        .position(|text| text == "root question")
        .expect("the root's question");
    let subtask = one_texts
        .iter()
        .position(|text| text == "do the subtask")
        .expect("the spawned session inline");
    assert!(question < subtask, "{one_texts:?}");
    assert!(!one_texts.iter().any(|text| text == "the nested subtask"));
    let card_at = one
        .document
        .items
        .iter()
        .position(|item| matches!(&item.kind, ItemKind::Card { card }
            if matches!(card.as_ref(), Card::Delegation { card } if card.child == "session:claude-code:agent-x1")))
        .expect("the delegation card");
    let subtask_at = one
        .document
        .items
        .iter()
        .position(|item| matches!(&item.kind, ItemKind::Text { text } if text == "do the subtask"))
        .unwrap();
    assert!(card_at < subtask_at, "the card heads the inline reading");
    // The child sits at its call, inside the spawning message: before the
    // message's next call and before the result message.
    let position = |needle: &str| {
        one.document
            .items
            .iter()
            .position(|item| serde_json::to_string(item).unwrap().contains(needle))
            .unwrap_or_else(|| panic!("{needle} in the reading"))
    };
    assert!(subtask_at < position("ls after the spawn"));
    assert!(subtask_at < position("message:claude-code:r3"));
    // The split message is laid out once.
    let headers: Vec<&str> = one
        .document
        .items
        .iter()
        .filter(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
        .map(|item| item.source.reference.as_str())
        .collect();
    let mut distinct = headers.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(headers.len(), distinct.len(), "{headers:?}");
    let calls = one
        .document
        .items
        .iter()
        .filter(|item| {
            serde_json::to_string(item)
                .unwrap()
                .contains("ls after the spawn")
        })
        .count();
    assert_eq!(calls, 1);
    assert!(
        cards(&one).iter().any(|card| matches!(card,
            Card::Delegation { card } if card.child == "session:claude-code:agent-x2")),
        "a spawn below the depth stays a card"
    );
    let full = texts(&read(
        &d,
        "session:claude-code:root",
        ReadOptions {
            show: vec!["spawns:full".into()],
            ..ReadOptions::default()
        },
    ));
    let nested = full
        .iter()
        .position(|text| text == "the nested subtask")
        .expect("every level inline");
    let subtask = full
        .iter()
        .position(|text| text == "do the subtask")
        .unwrap();
    assert!(subtask < nested, "{full:?}");
}

/// The dialogue view hides the spawning call, so a cut after it leaves its
/// message header with no shown block before the child; the header stays.
#[test]
fn a_spawn_cut_keeps_its_header() {
    let d = test_daemon::start("placeholder");
    let dir = d.home.join(".claude/projects/-p");
    let root = [
        user("s1", "spawner", "2026-03-01T10:00:00Z", "spawner question"),
        line(json!({
            "type": "assistant", "uuid": "s2", "parentUuid": "s1", "sessionId": "spawner",
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": "msg_s2", "role": "assistant", "stop_reason": "tool_use",
                "content": [{"type": "tool_use", "id": "call-s", "name": "Task",
                             "input": {"description": "check", "prompt": "check it"}},
                            {"type": "text", "text": "spawned a checker"}]}
        })),
        line(json!({
            "type": "user", "uuid": "s3", "parentUuid": "s2", "sessionId": "spawner",
            "timestamp": "2026-03-01T10:00:02Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-s", "content": "checked"}]},
            "toolUseResult": {"agentId": "y1"}
        })),
    ]
    .concat();
    std::fs::write(dir.join("spawner.jsonl"), root).unwrap();
    let agents = dir.join("spawner/subagents");
    std::fs::create_dir_all(&agents).unwrap();
    std::fs::write(
        agents.join("agent-y1.jsonl"),
        line(json!({
            "type": "user", "uuid": "b1", "sessionId": "spawner", "agentId": "y1", "isSidechain": true,
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"role": "user", "content": "do the subtask"}
        })),
    )
    .unwrap();
    let spawner = polylogue_model::SessionId::parse("claude-code:spawner").unwrap();
    let inline = || ReadOptions {
        view: View::Dialogue,
        show: vec!["spawns:1".into()],
        ..ReadOptions::default()
    };
    d.wait_for(|status| {
        status["busy"] == false
            && d.daemon
                .reads
                .reader()
                .unwrap()
                .session(&spawner)
                .unwrap()
                .is_some()
            && texts(&read(&d, "session:claude-code:spawner", inline()))
                .iter()
                .any(|text| text == "do the subtask")
    });
    let page = read(&d, "session:claude-code:spawner", inline());
    let place = |wanted: &dyn Fn(&ItemKind) -> bool| {
        page.document
            .items
            .iter()
            .position(|item| wanted(&item.kind))
            .expect("item in the reading")
    };
    let spawner = page
        .document
        .items
        .iter()
        .position(|item| {
            matches!(item.kind, ItemKind::MessageHeader { .. })
                && item.source.reference == "message:claude-code:s2"
        })
        .expect("the spawning message keeps its header");
    let child = place(&|kind| matches!(kind, ItemKind::Text { text } if text == "do the subtask"));
    let after =
        place(&|kind| matches!(kind, ItemKind::Text { text } if text == "spawned a checker"));
    assert!(spawner < child && child < after);
}

#[test]
fn a_microcompaction_removes_its_results_from_the_context_view() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let shown = |view| {
        let page = read(
            &d,
            "session:claude-code:micro",
            ReadOptions {
                view,
                show: vec!["tools:100".into()],
                ..ReadOptions::default()
            },
        );
        serde_json::to_string(&page.document.items).unwrap()
    };
    let transcript = shown(View::Transcript);
    assert!(transcript.contains("cleared listing"), "{transcript}");
    let context = shown(View::Context);
    assert!(!context.contains("cleared listing"), "{context}");
    assert!(context.contains("after microcompaction"), "{context}");
    assert!(context.contains("before microcompaction"), "{context}");
}

/// Model replies without reported usage before and after a microcompaction
/// and a compaction: each one's context meter estimate must equal the sum
/// over what `context_at` finds in context there, computed from the whole
/// index.
#[test]
fn the_context_meter_estimates_from_stored_sums_what_context_at_computes() {
    let d = test_daemon::start("placeholder");
    let reply = |uuid: &str, parent: &str, at: &str, content: serde_json::Value| {
        line(json!({
            "type": "assistant", "uuid": uuid, "parentUuid": parent, "sessionId": "meter",
            "timestamp": at,
            "message": {"id": format!("msg_{uuid}"), "role": "assistant", "content": content}
        }))
    };
    let session = [
        user("p1", "meter", "2026-03-02T10:00:00Z", "first prompt"),
        reply(
            "a1",
            "p1",
            "2026-03-02T10:00:01Z",
            json!([{"type": "tool_use", "id": "call-x", "name": "Bash", "input": {"command": "ls"}}]),
        ),
        line(json!({
            "type": "user", "uuid": "r1", "parentUuid": "a1", "sessionId": "meter",
            "timestamp": "2026-03-02T10:00:02Z",
            "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-x", "content": "a long listing ".repeat(40)}]}
        })),
        reply("a2", "r1", "2026-03-02T10:00:03Z", json!([{"type": "text", "text": "seen the listing"}])),
        line(json!({
            "type": "system", "subtype": "microcompact_boundary", "uuid": "c1", "parentUuid": "a2",
            "sessionId": "meter", "timestamp": "2026-03-02T10:00:04Z",
            "content": "Context microcompacted",
            "microcompactMetadata": {"trigger": "auto", "preTokens": 100, "tokensSaved": 50,
                "compactedToolIds": ["call-x"], "clearedAttachmentUUIDs": []}
        })),
        reply("a3", "c1", "2026-03-02T10:00:05Z", json!([{"type": "text", "text": "after the clearing"}])),
        line(json!({
            "type": "system", "subtype": "compact_boundary", "uuid": "b1", "sessionId": "meter",
            "logicalParentUuid": "a3", "timestamp": "2026-03-02T10:00:06Z",
            "compactMetadata": {"trigger": "auto", "preTokens": 9000}
        })),
        line(json!({
            "type": "user", "uuid": "s1", "sessionId": "meter", "isCompactSummary": true,
            "timestamp": "2026-03-02T10:00:07Z",
            "message": {"role": "user", "content": "summary of the work so far"}
        })),
        reply("a4", "s1", "2026-03-02T10:00:08Z", json!([{"type": "text", "text": "after compaction"}])),
    ]
    .concat();
    std::fs::write(d.home.join(".claude/projects/-p/meter.jsonl"), session).unwrap();
    d.wait_for(|status| status["store"]["sessions"] == 2 && status["store"]["search_pending"] == 0);
    let page = read(
        &d,
        "session:claude-code:meter",
        ReadOptions {
            show: vec!["context".into()],
            ..ReadOptions::default()
        },
    );
    // The oracle: the whole index and the model's context function.
    let reader = d.daemon.reads.reader().unwrap();
    let session = polylogue_model::SessionId::parse("claude-code:meter").unwrap();
    let index = reader.item_index(&session).unwrap().unwrap();
    let parts = [polylogue_model::read::Part {
        session,
        start: 0,
        end: index.entries.len(),
        inherited: false,
        joined_by: None,
    }];
    let mut meters = Vec::new();
    for item in &page.document.items {
        let ItemKind::MessageHeader { header } = &item.kind else {
            continue;
        };
        let Some(context) = &header.context else {
            continue;
        };
        let message = polylogue_model::MessageId::parse(
            item.source.reference.strip_prefix("message:").unwrap(),
        )
        .unwrap();
        let at = index
            .entries
            .iter()
            .position(|entry| entry.key == polylogue_model::read::EntryKey::header(&message))
            .unwrap();
        let members = polylogue_model::read::context_at(
            &parts,
            |_| &index.entries,
            polylogue_model::read::Place { part: 0, entry: at },
        )
        .members;
        let expected: u64 = members
            .iter()
            .map(|place| index.entries[place.entry].tokens_full)
            .sum();
        assert_eq!(
            context,
            &polylogue_model::read::ValueState::Inferred { value: expected },
            "{message}"
        );
        meters.push(expected);
    }
    // Four replies; the clearing and the compaction each lower the estimate.
    assert_eq!(meters.len(), 4, "{meters:?}");
    assert!(meters[2] < meters[1] && meters[3] < meters[2], "{meters:?}");
}

#[test]
fn selected_message_context_keeps_its_prefix_and_existing_parent_branch_cut() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let position = d.daemon.reads.reader().unwrap().change_position().unwrap();
    let options = ReadOptions {
        view: View::Context,
        at: Some(polylogue_model::read::ReadAt::try_from(position.to_string()).unwrap()),
        ..ReadOptions::default()
    };
    let headers = |page: &ReadPage| {
        page.document
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::MessageHeader { .. } => Some(item.source.reference.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let context = read(&d, "message:claude-code:r2", options.clone());
    assert_eq!(
        headers(&context),
        ["message:claude-code:r1", "message:claude-code:r2"]
    );
    assert!(texts(&context).iter().any(|text| text == "root question"));
    for excluded in [
        "continuation question",
        "forked question",
        "do the subtask",
        "the nested subtask",
    ] {
        assert!(!texts(&context).iter().any(|text| text == excluded));
    }
    let fork = read(&d, "message:claude-code:f1", options);
    assert_eq!(
        headers(&fork),
        ["message:claude-code:r1", "message:claude-code:f1"]
    );
    let around = read(
        &d,
        "message:claude-code:r2",
        ReadOptions {
            view: View::Context,
            around: Some(1),
            ..ReadOptions::default()
        },
    );
    assert_eq!(
        headers(&around),
        [
            "message:claude-code:r1",
            "message:claude-code:r2",
            "message:claude-code:r3"
        ]
    );
    // Transcript message ranges keep their original one-message meaning.
    let transcript = read(&d, "message:claude-code:r2", ReadOptions::default());
    assert_eq!(headers(&transcript), ["message:claude-code:r2"]);
    assert!(
        !texts(&transcript)
            .iter()
            .any(|text| text == "root question")
    );
}

#[test]
fn finite_context_streams_use_the_same_lineage_compaction_and_tool_projection_as_pages() {
    let d = test_daemon::start("placeholder");
    family(&d);
    for (target, diff, around) in [
        ("root:claude-code:root", None, None),
        ("message:claude-code:r2", None, None),
        ("message:claude-code:f1", None, None),
        ("message:claude-code:r2", None, Some(1)),
        ("session:claude-code:cmp", None, None),
        ("session:claude-code:cmp", Some("compaction"), None),
        ("session:claude-code:micro", None, None),
    ] {
        let options = ReadOptions {
            view: View::Context,
            show: vec!["tools:100".into()],
            diff: diff.map(str::to_owned),
            around,
            ..Default::default()
        };
        let expected = read(&d, target, options.clone());
        let (items, originals, _) =
            super::read_annotations::context_stream(&d.daemon, target, options);
        assert_eq!(
            items, expected.document.items,
            "{target}, {diff:?}, {around:?}"
        );
        for original in &originals {
            if let polylogue_model::bundle::Record::Message(message) = original {
                message.validate().unwrap();
            }
        }
        let encoded = serde_json::to_string(&originals).unwrap();
        if target == "session:claude-code:micro" {
            assert!(
                !encoded.contains("cleared listing"),
                "evicted blocks must also be absent from original frames: {encoded}"
            );
        }
        if target == "session:claude-code:cmp" && diff.is_none() {
            assert!(!encoded.contains("before compaction"), "{encoded}");
        }
    }
}

#[test]
fn a_context_stream_crosses_entry_windows_without_duplicating_its_source_message() {
    let d = test_daemon::start_with_sources_and_workers(
        "placeholder",
        &|home| {
            let blocks: Vec<_> = (0..40)
                .map(|i| json!({"type":"text","text":format!("neutral block {i}")}))
                .collect();
            std::fs::write(home.join(".claude/projects/-p/wide.jsonl"), line(json!({
            "type":"assistant", "uuid":"wide", "sessionId":"wide", "timestamp":"2026-03-01T10:00:00Z",
            "message":{"id":"msg_wide", "role":"assistant", "stop_reason":"end_turn", "content":blocks}
        }))).unwrap();
        },
        1,
        1,
    );
    let options = ReadOptions {
        view: View::Context,
        ..Default::default()
    };
    let expected = read(&d, "session:claude-code:wide", options.clone());
    assert!(
        expected.document.items.len() > 32,
        "the fixture crosses two entry windows"
    );
    let (items, originals, complete) =
        super::read_annotations::context_stream(&d.daemon, "session:claude-code:wide", options);
    assert_eq!(items, expected.document.items);
    assert_eq!(complete.counts.messages, 1);
    let messages: Vec<_> = originals
        .iter()
        .filter_map(|record| match record {
            polylogue_model::bundle::Record::Message(message) => Some(message),
            _ => None,
        })
        .collect();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].block_count, 40);
    assert_eq!(messages[0].block_ordinals, (0..40).collect::<Vec<u64>>());
    messages[0].validate().unwrap();
}

#[test]
fn context_result_data_and_attachments_survive_entry_windows_and_page_boundaries_once() {
    use polylogue_model::read::Expansion;
    let d = test_daemon::start_with_sources_and_workers(
        "placeholder",
        &|home| {
            let mut blocks =
                vec![json!({"type":"tool_use","id":"capture-1","name":"capture","input":{}})];
            blocks.extend(
                (0..20).map(
                    |i| json!({"type":"text","text":format!("neutral intervening block {i}")}),
                ),
            );
            let call = line(json!({
                "type":"assistant","uuid":"capture-call","sessionId":"capture","timestamp":"2026-03-01T10:00:00Z",
                "message":{"id":"msg_capture","role":"assistant","stop_reason":"tool_use","content":blocks}
            }));
            let result = line(json!({
                "type":"user","uuid":"capture-result","parentUuid":"capture-call","sessionId":"capture","timestamp":"2026-03-01T10:00:01Z",
                "message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"capture-1","content":[
                    {"type":"text","text":"neutral paired output"},
                    {"type":"document","source":{"type":"base64","media_type":"application/octet-stream","data":"YWJj"}}
                ]}]}, "toolUseResult":{"status":"complete"}
            }));
            std::fs::write(
                home.join(".claude/projects/-p/capture.jsonl"),
                call + &result,
            )
            .unwrap();
        },
        2,
        1,
    );
    for tools in [Expansion::Full, Expansion::Collapsed, Expansion::None] {
        let (show, hide) = match tools {
            Expansion::Full => (vec!["tools:100".into()], vec![]),
            Expansion::Collapsed => (vec!["tools:line".into()], vec![]),
            Expansion::None => (vec![], vec!["tools".into()]),
        };
        let options = ReadOptions {
            view: View::Context,
            show,
            hide,
            ..Default::default()
        };
        let target = "session:claude-code:capture";
        let expected = read(&d, target, options.clone());
        let (items, _, _) =
            super::read_annotations::context_stream(&d.daemon, target, options.clone());
        assert_eq!(items, expected.document.items);
        assert_eq!(
            items
                .iter()
                .filter(|item| matches!(&item.kind, ItemKind::Call { .. }))
                .count(),
            usize::from(tools != Expansion::None)
        );
        let data: Vec<_> = items.iter().filter(|item| matches!(&item.kind, ItemKind::Other { type_name, .. } if type_name == "tool result data")).collect();
        let attachments: Vec<_> = items
            .iter()
            .filter(|item| matches!(&item.kind, ItemKind::Attachment { .. }))
            .collect();
        assert_eq!(data.len(), usize::from(tools == Expansion::Full));
        assert_eq!(attachments.len(), usize::from(tools == Expansion::Full));
        for item in data.into_iter().chain(attachments) {
            assert_eq!(item.source.reference, "block:claude-code:capture-result#0");
        }
        if tools == Expansion::Full {
            assert!(items.iter().any(|item| matches!(&item.kind, ItemKind::Call { call } if call.output.as_deref() == Some("neutral paired output"))));
            let request = polylogue_reading::ReadRequest {
                start: Default::default(),
                references: None,
                target: target.into(),
                options,
                delivery: polylogue_model::read::ReadDelivery::Page,
                format: polylogue_model::read::Format::Json,
                budget: polylogue_model::read::ReadBudget::Tokens(64),
                alternates: Vec::new(),
                cursor: None,
            };
            let mut page = crate::test_answers::read(
                crate::test_daemon::read_call(request.clone()).answer(&d.daemon),
            );
            let mut all = page.document.items;
            let mut seen = std::collections::BTreeSet::new();
            while let Some(next) = page.document.footer.next {
                assert!(
                    seen.insert(serde_json::to_string(&next.request).unwrap()),
                    "pages advance"
                );
                page = crate::test_answers::read(
                    crate::test_daemon::Call::from_request(&next.request).answer(&d.daemon),
                );
                all.extend(page.document.items);
            }
            assert!(!seen.is_empty(), "the fixture must cross page boundaries");
            assert_eq!(all, items);
        }
    }
}

#[test]
fn composed_context_streams_keep_distinct_physical_session_headers_once() {
    use polylogue_model::bundle::Record;
    let d = test_daemon::start("placeholder");
    family(&d);
    let cases: &[(&str, &[&str], &[&str])] = &[
        (
            "message:claude-code:f1",
            &["claude-code:fork", "claude-code:root"],
            &["root question", "forked question"],
        ),
        // A bare continued-in marker has no branch point or copied-prefix
        // claim. A physical read must not invent its parent's context.
        (
            "session:claude-code:cont",
            &["claude-code:cont"],
            &["continuation question"],
        ),
        // The logical root deliberately follows that recorded continuation.
        (
            "root:claude-code:root",
            &["claude-code:root", "claude-code:cont"],
            &["root question", "continuation question"],
        ),
    ];
    for &(target, expected, expected_text) in cases {
        let options = ReadOptions {
            view: View::Context,
            ..Default::default()
        };
        let page = read(&d, target, options.clone());
        let (items, originals, complete) =
            super::read_annotations::context_stream(&d.daemon, target, options);
        assert_eq!(items, page.document.items);
        let body_text: Vec<_> = items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        for text in expected_text {
            assert!(
                body_text.contains(text),
                "{target} must retain {text}: {body_text:?}"
            );
        }
        if target == "session:claude-code:cont" {
            assert!(
                !body_text.contains(&"root question"),
                "a missing branch point is not a copied prefix"
            );
        }
        let headers: Vec<_> = originals
            .iter()
            .filter_map(|record| match record {
                Record::Session(row) => Some(row),
                _ => None,
            })
            .collect();
        assert_eq!(
            headers
                .iter()
                .map(|row| row.session.id.to_string())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(complete.counts.sessions, expected.len() as u64);
        let reader = d.daemon.reads.reader().unwrap();
        for header in &headers {
            assert_eq!(header.reference, format!("session:{}", header.session.id));
            assert_eq!(
                header.session,
                reader.session(&header.session.id).unwrap().unwrap()
            );
            assert_eq!(
                Some(header.commit),
                reader.session_commit(&header.session.id).unwrap()
            );
            assert_eq!(
                header.copied_prefixes,
                reader.copied_prefixes(&header.session.id).unwrap()
            );
        }
        for record in &originals {
            if let Record::Message(row) = record {
                assert!(
                    headers
                        .iter()
                        .any(|header| header.session.id == row.message.session),
                    "the source of {} needs its own physical header",
                    row.reference
                );
            }
        }
    }
}
