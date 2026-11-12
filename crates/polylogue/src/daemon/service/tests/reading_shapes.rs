//! Readings through the real request path: logical sessions, fork and
//! spawn cards, context differences, the links tree and following.
use super::*;
use polylogue_model::read::{Card, HeaderKind, ItemKind, ReadOptions, ReadPage, SpawnFold, View};
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
fn family(d: &test_daemon::TestDaemon) {
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
    let response = (crate::test_daemon::read(crate::daemon::service::read::ReadRequest {
        target: target.into(),
        options,
        delivery: polylogue_model::read::ReadDelivery::Page,
        format: "json".into(),
        budget: Some(100_000),
        with_blobs: false,
        cursor: None,
    }))
    .answer(&d.daemon);
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    serde_json::from_value(data).unwrap()
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
    let physical = texts(&read(
        &d,
        "session:claude-code:root",
        ReadOptions::default(),
    ));
    assert!(!physical.iter().any(|text| text == "continuation question"));
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
            spawns: SpawnFold::Hidden,
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
            view: View::Links,
            ..ReadOptions::default()
        },
    );
    let nodes: Vec<(String, bool)> = cards(&page)
        .into_iter()
        .filter_map(|card| match card {
            Card::LinkNode {
                reference, seed, ..
            } => Some((reference.clone(), *seed)),
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
            .filter(|(_, seed)| *seed)
            .map(|(node, _)| node.as_str())
            .collect::<Vec<_>>(),
        ["session:claude-code:cont"]
    );
}

#[test]
fn the_last_page_follows_what_the_session_appends() {
    let d = test_daemon::start("placeholder");
    family(&d);
    let last = read(
        &d,
        "session:claude-code:cont",
        ReadOptions {
            latest: true,
            ..ReadOptions::default()
        },
    );
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
    let native = crate::test_daemon::Call::from_request(&envelope);
    let response = (native).answer(&d.daemon);
    let page: ReadPage = serde_json::from_value(response.value().cloned().unwrap()).unwrap();
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
            spawns: SpawnFold::Inline { depth: Some(1) },
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
            spawns: SpawnFold::Inline { depth: None },
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
                tools: Some(polylogue_model::read::Expansion::Full),
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
            context_meter: true,
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
