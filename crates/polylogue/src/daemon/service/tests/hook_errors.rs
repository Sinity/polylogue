use super::*;
use polylogue_model::read::{
    Expansion, ItemKind, ReadDelivery, ReadItem, ReadOptions, ReadPage, View,
};
use serde_json::json;

#[derive(Default)]
struct Items {
    pending: String,
    values: Vec<ReadItem>,
}
impl Items {
    fn push(&mut self, page: &ReadPage) {
        for item in &page.document.items {
            if let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            {
                assert_eq!(position.character, self.pending.chars().count() as u64);
                assert!(*until > position.character && *until <= *total);
                self.pending.push_str(text);
                assert_eq!(*until, self.pending.chars().count() as u64);
                if until == total {
                    assert_eq!(
                        polylogue_model::BlobHash::of(self.pending.as_bytes()),
                        position.item_hash
                    );
                    let complete: ReadItem = serde_json::from_str(&self.pending).unwrap();
                    assert!(!matches!(complete.kind, ItemKind::Fragment { .. }));
                    self.values.push(complete);
                    self.pending.clear();
                }
            } else {
                assert!(
                    self.pending.is_empty(),
                    "fragment interrupted before its final byte"
                );
                self.values.push(item.clone());
            }
        }
    }
    fn text(&self) -> String {
        assert!(self.pending.is_empty());
        self.values
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
}
fn message_id(item: &ReadItem) -> polylogue_model::MessageId {
    let polylogue_model::Reference::Message(id) =
        polylogue_model::Reference::parse(&item.source.reference).unwrap()
    else {
        panic!("message header must name its actual canonical message")
    };
    // A header's reference is its message; only block sources name a containing message.
    assert_eq!(item.source.message, None, "{item:?}");
    id
}
fn mcp_page(archive: &Archive, arguments: serde_json::Value) -> ReadPage {
    let answer =
        crate::mcp::tests::call(archive, "read", arguments.as_object().unwrap().clone()).unwrap();
    serde_json::from_str(&answer).unwrap()
}

#[test]
fn native_hook_failures_reach_ordinary_mcp_reads_and_budgeted_pages() {
    let daemon = test_daemon::start("human prompt");
    let source = daemon.home.join(".claude/projects/-p/s1.jsonl");
    let summary = json!({"type":"system", "subtype":"stop_hook_summary", "uuid":"hook1",
        "sessionId":"s1", "parentUuid":"u1", "hookErrors":["  synthetic hook failed λ\n"],
        "hookAdditionalContext":[], "preventedContinuation":true, "stopReason":"check blocked"});
    let mut original = std::fs::read(&source).unwrap();
    original.extend(serde_json::to_vec(&summary).unwrap());
    original.push(b'\n');
    std::fs::write(&source, &original).unwrap();
    let id = polylogue_model::SessionId::parse("claude-code:s1").unwrap();
    daemon.wait_for(|_| {
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .events(&id)
            .unwrap()
            .iter()
            .any(|event| event.data == summary)
    });
    let native_messages = daemon
        .daemon
        .reads
        .reader()
        .unwrap()
        .messages(&id, false)
        .unwrap();
    assert_eq!(native_messages.len(), 2);
    assert_eq!(
        native_messages[1].prompt_kind,
        Some(polylogue_model::PromptKind::Harness)
    );
    let ordinary = (crate::test_daemon::read(crate::daemon::service::read::ReadRequest {
        target: "session:claude-code:s1".into(),
        options: ReadOptions {
            view: View::Transcript,
            harness: Some(Expansion::Full),
            ..Default::default()
        },
        delivery: ReadDelivery::Page,
        format: "json".into(),
        budget: None,
        with_blobs: false,
        cursor: None,
    }))
    .answer(&daemon.daemon);
    let typed = typed(ordinary);
    let headers: Vec<_> = typed
        .document
        .items
        .iter()
        .filter(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
        .collect();
    assert_eq!(headers.len(), 2);
    assert_eq!(message_id(headers[1]), native_messages[1].id);
    let archive = Archive::at(&daemon.daemon.archive.root);
    let full = mcp_page(
        &archive,
        json!({"target":"session:claude-code:s1", "show":["harness:full"], "page_tokens":10000}),
    );
    assert!(full.document.footer.next.is_none());
    let mut expected = Items::default();
    expected.push(&full);
    let expected_text = expected.text();
    assert!(
        expected_text.contains("synthetic hook failed λ"),
        "{expected_text}"
    );
    assert!(expected_text.contains("Stop hook prevented continuation."));
    assert!(expected_text.contains("check blocked"));
    let mut page = mcp_page(
        &archive,
        json!({"target":"session:claude-code:s1", "show":["harness:full"], "page_tokens":1}),
    );
    assert_eq!(page.snapshot.query, full.snapshot.query);
    let question = page.snapshot.query.clone();
    let mut items = Items::default();
    let mut seen = std::collections::BTreeSet::new();
    let mut root = None;
    loop {
        assert_eq!(page.snapshot.query, question);
        assert!(serde_json::to_vec(&page).unwrap().len() <= 25_000);
        items.push(&page);
        let Some(next) = page.document.footer.next else {
            break;
        };
        let envelope: command_surface::RunnableRequest =
            serde_json::from_value(next.http.arguments.clone()).unwrap();
        assert_eq!(envelope.operation, "read");
        let arguments: crate::protocol::read::Arguments =
            serde_json::from_value(envelope.arguments.clone()).unwrap();
        if let Some(root) = &root {
            assert_eq!(&arguments.target, root);
        } else {
            root = Some(arguments.target.clone());
        }
        assert!(
            seen.insert(
                envelope
                    .cursor
                    .clone()
                    .expect("a continuation has a cursor")
            ),
            "continuation must advance"
        );
        page = crate::mcp::tests::read_pages::execute_ready(&archive, &next).unwrap();
    }
    let joined = items.text();
    let messages: Vec<_> = items
        .values
        .iter()
        .filter(|item| matches!(item.kind, ItemKind::MessageHeader { .. }))
        .map(message_id)
        .collect();
    assert_eq!(
        messages,
        native_messages
            .iter()
            .map(|message| message.id.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(joined, expected_text);
    assert!(joined.contains("synthetic hook failed λ"));
    assert_eq!(std::fs::read(source).unwrap(), original);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .events(&id)
            .unwrap()
            .iter()
            .any(|event| event.data == summary)
    );
}
