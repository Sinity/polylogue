use super::*;
use polylogue_model::read::{ItemKind, ReadItem, ReadPage, ReadyCall, SelectionLocator};
use polylogue_model::{BlobHash, RunId};
use std::collections::{BTreeMap, BTreeSet};

type ItemKey = (SelectionLocator, u64, BlobHash);
#[derive(Default)]
struct PartialItem {
    text: String,
    characters: u64,
    total: u64,
}
#[derive(Default)]
pub(super) struct Items {
    pending: BTreeMap<ItemKey, PartialItem>,
    completed: BTreeSet<ItemKey>,
    pub(super) values: Vec<ReadItem>,
    bytes: usize,
}
impl Items {
    fn push(&mut self, page: &ReadPage) {
        let before = self.bytes;
        for item in &page.document.items {
            let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            else {
                assert!(
                    !self.values.contains(item),
                    "duplicate complete layout item"
                );
                self.bytes += serde_json::to_vec(item).unwrap().len();
                self.values.push(item.clone());
                continue;
            };
            assert_eq!(
                item.source.reference, page.snapshot.query,
                "fragment source must be the acknowledged question"
            );
            let key = (position.selection, position.item, position.item_hash);
            assert!(
                !self.completed.contains(&key),
                "a completed item was repeated"
            );
            let pending = self.pending.entry(key).or_insert_with(|| PartialItem {
                total: *total,
                ..Default::default()
            });
            assert_eq!(pending.total, *total, "fragment total changed");
            assert_eq!(
                position.character, pending.characters,
                "fragment gap or repeated prefix"
            );
            assert!(*until >= position.character);
            assert_eq!(*until - position.character, text.chars().count() as u64);
            assert!(*until <= *total);
            assert!(*until > position.character, "empty fragment cannot advance");
            pending.text.push_str(text);
            pending.characters = *until;
            self.bytes += text.len();
            if until == total {
                let complete = self.pending.remove(&key).unwrap();
                assert_eq!(
                    BlobHash::of(complete.text.as_bytes()),
                    position.item_hash,
                    "complete encoded item digest"
                );
                let item: ReadItem = serde_json::from_str(&complete.text).unwrap();
                assert!(
                    !matches!(item.kind, ItemKind::Fragment { .. }),
                    "nested delivery fragment"
                );
                assert!(
                    !self.values.contains(&item),
                    "duplicate reconstructed layout item"
                );
                self.values.push(item);
                self.completed.insert(key);
            }
        }
        assert!(self.bytes > before, "a page must advance the actual layout");
    }
    pub(super) fn text(&self) -> String {
        self.values
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }
    fn finish(&self) {
        assert!(
            self.pending.is_empty(),
            "delivery ended inside an encoded layout item"
        );
    }
}

pub(super) fn body(page: &ReadPage) -> String {
    let mut items = Items::default();
    items.push(page);
    items.text()
}

/// Acts as a client: execute the actual bounded continuation, then join encoded
/// fragments by their typed coordinate and item digest before decoding content.
pub(super) fn collect(
    daemon: &test_daemon::TestDaemon,
    mut page: ReadPage,
    budget: Option<u64>,
    max_wire_bytes: Option<usize>,
    expected_deadline: command_surface::Deadline,
) -> (Items, usize) {
    let archive = Archive::at(&daemon.daemon.archive.root);
    let query = page.snapshot.query.clone();
    let format = page.format;
    let mut root: Option<(RunId, BlobHash)> = None;
    let mut cursors = BTreeSet::new();
    let mut requests = BTreeSet::new();
    let mut items = Items::default();
    let mut pages = 0;
    loop {
        assert_eq!(page.snapshot.query, query);
        assert_eq!(
            page.format, format,
            "continuation changed the original format"
        );
        assert!(
            page.records.is_empty(),
            "bounded layout must not duplicate canonical records"
        );
        if let Some(cap) = max_wire_bytes {
            assert!(serde_json::to_vec(&page).unwrap().len() < cap);
        }
        items.push(&page);
        pages += 1;
        assert!(pages < 100, "continuation failed to advance");
        let Some(mut ready) = page.document.footer.next.clone() else {
            break;
        };
        let envelope = &ready.mcp["arguments"];
        let child: crate::protocol::operations::McpArguments<Value> =
            serde_json::from_value(envelope.clone()).unwrap();
        assert_eq!(
            child.deadline, expected_deadline,
            "generated child changed the deadline"
        );
        let request_id = envelope["request_id"]
            .as_str()
            .expect("complete accepted child envelope");
        assert!(
            requests.insert(request_id.to_owned()),
            "next page reused an invocation identity"
        );
        let cursor = envelope["arguments"]["cursor"]
            .as_str()
            .expect("opaque continuation");
        assert!(
            cursors.insert(cursor.to_owned()),
            "continuation repeated a position"
        );
        let crate::protocol::read::Arguments::Resume(arguments) =
            serde_json::from_value(envelope["arguments"].clone()).unwrap()
        else {
            panic!("next call must use the closed compact resume form");
        };
        let reader = daemon.daemon.reads.reader().unwrap();
        let acknowledged = reader
            .query_run(&arguments.root_run, None)
            .unwrap()
            .expect("root Begin is acknowledged before exposing continuation")
            .value;
        let original = acknowledged
            .binding
            .original()
            .expect("continuation must point one hop to the original run");
        let original: command_surface::RunnableRequest = serde_json::from_value(
            original
                .accepted_request
                .clone()
                .expect("actual accepted root envelope"),
        )
        .unwrap();
        assert_eq!(
            original.deadline, expected_deadline,
            "root deadline was not retained"
        );
        assert_eq!(query, format!("query:{}", acknowledged.question));
        let hash = acknowledged.binding_hash().unwrap();
        let identity = (arguments.root_run, hash);
        if let Some(root) = &root {
            assert_eq!(&identity, root, "original execution binding changed");
        } else {
            root = Some(identity);
        }
        drop(reader);
        if let Some(budget) = budget {
            ready.mcp["arguments"]["arguments"]["budget"] = json!(budget);
        }
        page = execute_ready(&archive, &ready).unwrap();
    }
    items.finish();
    (items, pages)
}

/// Whole-byte reconstruction uses the documented lifted deadline at the root.
/// Children retain that accepted choice; the client never edits their deadlines.
pub(super) fn read_without_deadline(archive: &Archive, arguments: Value) -> Result<String, String> {
    let command = crate::commands::command("read").unwrap();
    let envelope = serde_json::to_value(crate::protocol::operations::McpArguments {
        arguments,
        request_id: uuid::Uuid::now_v7().to_string(),
        deadline: command_surface::Deadline::None,
    })
    .unwrap();
    assert_eq!(envelope.get("deadline"), Some(&Value::Null));
    let (text, _, failed) = call_with_limit(
        archive,
        &command.mcp_name(),
        envelope.as_object().unwrap().clone(),
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )?;
    if failed { Err(text) } else { Ok(text) }
}

pub(super) fn execute_ready(archive: &Archive, ready: &ReadyCall) -> Result<ReadPage, String> {
    let name = ready.mcp["name"].as_str().expect("generated MCP tool name");
    let arguments = ready.mcp["arguments"]
        .as_object()
        .expect("generated complete envelope")
        .clone();
    let (text, _, failed) = call_with_limit(
        archive,
        name,
        arguments,
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )?;
    if failed {
        Err(text)
    } else {
        serde_json::from_str(&text).map_err(|error| error.to_string())
    }
}

#[test]
fn mcp_read_pages_execute_the_generated_continuation_without_losing_unicode() {
    let source = "synthetic MCP prompt λ🙂 boundary ".repeat(1000);
    let daemon = test_daemon::start(&source);
    let archive = Archive::at(&daemon.daemon.archive.root);
    let first = read_without_deadline(
        &archive,
        json!({"target":"session:claude-code:s1", "budget":256}),
    )
    .unwrap();
    let page: ReadPage = serde_json::from_str(&first).unwrap();
    let first_next = page
        .document
        .footer
        .next
        .clone()
        .expect("bounded first page has continuation");
    let mut invalid = first_next.clone();
    invalid.mcp["arguments"]["arguments"]["view"] = json!("dialogue");
    let error = execute_ready(&archive, &invalid).unwrap_err();
    assert!(error.contains("bad_request"), "{error}");
    let (items, pages) = collect(
        &daemon,
        page,
        Some(512),
        None,
        command_surface::Deadline::None,
    );
    assert!(pages > 1);
    assert_eq!(items.text(), source);
}

#[test]
fn budgeted_mcp_output_contains_one_bounded_layout_and_no_duplicate_source_records() {
    let source = "synthetic word ".repeat(10000);
    let daemon = test_daemon::start(&source);
    let archive = Archive::at(&daemon.daemon.archive.root);
    let output = call(
        &archive,
        "read",
        args(json!({"target":"session:claude-code:s1","budget":2048})),
    )
    .unwrap();
    let page: ReadPage = serde_json::from_str(&output).unwrap();
    assert!(page.document.footer.next.is_some());
    // A bounded page may finish at a header or inside encoded item JSON.
    // The client follows the actual continuation until the complete body decodes.
    let (items, pages) = collect(
        &daemon,
        page,
        None,
        Some(22_000),
        command_surface::Deadline::Default,
    );
    assert!(pages > 1);
    assert!(!items.text().is_empty());
    assert_eq!(items.text(), source);
    let error = call(
        &archive,
        "read",
        args(json!({"target":"session:claude-code:s1","slices":true})),
    )
    .unwrap_err();
    assert!(error.contains("bad_request"), "{error}");
}
