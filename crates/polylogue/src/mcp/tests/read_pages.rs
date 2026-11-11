use super::*;
use polylogue_model::BlobHash;
use polylogue_model::read::{ItemKind, ReadItem, ReadPage, ReadyCall, SelectionLocator};
use std::collections::{BTreeMap, BTreeSet};

type ItemKey = (SelectionLocator, u64, BlobHash);

// The public encoder owns the exact item JSON, including inert control escaping.
fn item_json(header: &polylogue_model::read::ReadHeader, item: &ReadItem) -> String {
    let mut prefix = Vec::new();
    {
        let _encoder = polylogue_render::Encoder::new(
            header,
            polylogue_model::read::Format::Json,
            Default::default(),
            &mut prefix,
        )
        .unwrap();
    }
    let mut output = Vec::new();
    {
        let mut encoder = polylogue_render::Encoder::new(
            header,
            polylogue_model::read::Format::Json,
            Default::default(),
            &mut output,
        )
        .unwrap();
        encoder.push(item).unwrap();
    }
    assert!(output.starts_with(&prefix));
    String::from_utf8(output.split_off(prefix.len())).unwrap()
}

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
                self.bytes += item_json(&page.document.header, item).len();
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

struct ItemUniverse {
    items: Vec<ReadItem>,
    encoded: BTreeMap<BlobHash, (String, Vec<usize>)>,
    bytes: usize,
}
impl ItemUniverse {
    fn from_root(daemon: &test_daemon::TestDaemon, next: &ReadyCall) -> Self {
        let child: crate::mcp::ToolCall =
            serde_json::from_value(next.mcp["arguments"].clone()).unwrap();
        let arguments: crate::protocol::read::Arguments =
            serde_json::from_value(child.arguments).unwrap();
        let mut request =
            crate::daemon::service::read::ReadRequest::from_arguments(arguments, None).unwrap();
        request.delivery = polylogue_model::read::ReadDelivery::Stream;
        request.budget = None;
        request.with_blobs = false;
        let operation = daemon
            .daemon
            .operation_context(&crate::operation::Context::worker_authorship(
                "polylogue-page-oracle",
            ))
            .unwrap();
        let mut bytes = Vec::new();
        crate::daemon::service::read::produce(&request, &daemon.daemon, &operation, &mut bytes)
            .unwrap()
            .unwrap();
        let mut items = Vec::new();
        let mut began = false;
        let mut header = None;
        let mut completed = false;
        crate::client::stream::receive(bytes.as_slice(), |frame| {
            match frame.event.as_str() {
                "begin" => {
                    assert!(!began);
                    began = true;
                    let begin: polylogue_model::read::ReadBegin =
                        serde_json::from_str(&frame.data).unwrap();
                    header = Some(begin.header);
                }
                "record" => {
                    assert!(began && !completed);
                    let record: polylogue_model::read::ReadRecord =
                        serde_json::from_str(&frame.data).unwrap();
                    items.extend(record.parts().iter().filter_map(|part| match part {
                        polylogue_model::read::ReadPart::Item(item) => Some(item.clone()),
                        _ => None,
                    }));
                }
                "complete" => {
                    assert!(began && !completed);
                    completed = true;
                }
                other => panic!("unexpected read oracle frame {other}"),
            }
            Ok(completed)
        })
        .unwrap();
        assert!(completed);
        let header = header.expect("actual stream header");
        let mut encoded = BTreeMap::new();
        let mut bytes = 0;
        for item in &items {
            let text = item_json(&header, item);
            bytes += text.len();
            let boundaries = text
                .char_indices()
                .map(|(at, _)| at)
                .chain(std::iter::once(text.len()))
                .collect();
            assert!(
                encoded
                    .insert(BlobHash::of(text.as_bytes()), (text, boundaries))
                    .is_none(),
                "duplicate source item"
            );
        }
        assert!(bytes > 0);
        Self {
            items,
            encoded,
            bytes,
        }
    }
    fn check(&self, page: &ReadPage) {
        for item in &page.document.items {
            if let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            {
                let (source, boundaries) = self
                    .encoded
                    .get(&position.item_hash)
                    .expect("fragment belongs to actual streamed source");
                assert_eq!(
                    *total,
                    (boundaries.len() - 1) as u64,
                    "actual encoded source total"
                );
                let from = usize::try_from(position.character).unwrap();
                let until = usize::try_from(*until).unwrap();
                assert!(from < until && until < boundaries.len());
                assert_eq!(
                    text,
                    &source[boundaries[from]..boundaries[until]],
                    "exact independently streamed fragment bytes"
                );
            } else {
                let text = item_json(&page.document.header, item);
                let (source, _) = self
                    .encoded
                    .get(&BlobHash::of(text.as_bytes()))
                    .expect("complete item belongs to actual streamed source");
                assert_eq!(&text, source);
            }
        }
    }
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
    let universe = page
        .document
        .footer
        .next
        .as_ref()
        .map(|next| ItemUniverse::from_root(daemon, next));
    let archive = Archive::at(&daemon.daemon.archive.root);
    let query = page.snapshot.query.clone();
    let format = page.format;
    let mut root: Option<(String, polylogue_model::read::View)> = None;
    let mut cursors = BTreeSet::new();
    let mut requests = BTreeSet::new();
    let mut items = Items::default();
    let mut pages = 0;
    let mut footer_total = 0;
    let mut wire_total = 0;
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
        let wire = serde_json::to_vec(&page).unwrap();
        let footer_bytes = serde_json::to_vec(&page.document.footer).unwrap().len();
        let payload_tokens = page
            .document
            .items
            .iter()
            .map(|item| {
                let text = match &item.kind {
                    ItemKind::Fragment { text, .. } => serde_json::to_string(text).unwrap(),
                    _ => item_json(&page.document.header, item),
                };
                polylogue_model::read::estimate_tokens(&text)
            })
            .sum::<usize>();
        let before = items.bytes;
        if let Some(cap) = max_wire_bytes {
            assert!(wire.len() < cap);
        }
        if let Some(universe) = &universe {
            universe.check(&page);
        }
        items.push(&page);
        pages += 1;
        if let Some(universe) = &universe {
            assert!(
                items.bytes <= universe.bytes,
                "strict byte progress cannot exceed the actual finite source"
            );
            assert!(
                pages <= universe.bytes,
                "each page delivers at least one source byte"
            );
        }
        let interval = |item: &ReadItem| match &item.kind {
            ItemKind::Fragment {
                position,
                until,
                total,
                ..
            } => Some((*position, *until, *total)),
            _ => None,
        };
        eprintln!(
            "page {pages}: wire={} footer={footer_bytes} payload={} payload_tokens={payload_tokens} wire_tokens={} first={:?} last={:?}",
            wire.len(),
            items.bytes - before,
            polylogue_model::read::estimate_tokens(std::str::from_utf8(&wire).unwrap()),
            page.document.items.first().and_then(interval),
            page.document.items.last().and_then(interval)
        );
        footer_total += footer_bytes;
        wire_total += wire.len();
        let Some(mut ready) = page.document.footer.next.clone() else {
            break;
        };
        let envelope = &ready.mcp["arguments"];
        let child: crate::mcp::ToolCall = serde_json::from_value(envelope.clone()).unwrap();
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
        let cursor = envelope["cursor"].as_str().expect("opaque continuation");
        assert!(
            cursors.insert(cursor.to_owned()),
            "continuation repeated a position"
        );
        let arguments: crate::protocol::read::Arguments =
            serde_json::from_value(envelope["arguments"].clone()).unwrap();
        // A continuation reads the same target with the same options.
        let identity = (arguments.target.clone(), arguments.view);
        if let Some(root) = &root {
            assert_eq!(&identity, root, "the continuation changed its reading");
        } else {
            root = Some(identity);
        }
        if let Some(budget) = budget {
            ready.mcp["arguments"]["arguments"]["page_tokens"] = json!(budget);
        }
        page = execute_ready(&archive, &ready).unwrap();
    }
    items.finish();
    if let Some(universe) = &universe {
        assert_eq!(
            items.bytes, universe.bytes,
            "every actual source byte delivered exactly once"
        );
        assert_eq!(
            items.values, universe.items,
            "complete streamed item universe in source order"
        );
    }
    eprintln!(
        "delivery: pages={pages} payload={} wire={wire_total} footer={footer_total}",
        items.bytes
    );
    (items, pages)
}

/// Whole-byte reconstruction uses the documented lifted deadline at the root.
/// Children retain that accepted choice; the client never edits their deadlines.
pub(super) fn structured_read_without_deadline(
    archive: &Archive,
    arguments: Value,
) -> Result<ReadPage, String> {
    let envelope = serde_json::to_value(crate::mcp::ToolCall {
        operation: Some("read".into()),
        arguments,
        budget_tokens: Default::default(),
        cursor: None,
        request_id: uuid::Uuid::now_v7().to_string(),
        deadline: command_surface::Deadline::None,
    })
    .unwrap();
    assert_eq!(envelope.get("deadline"), Some(&Value::Null));
    let (text, structured, failed) = call_with_limit(
        archive,
        Tool::Read.name(),
        envelope.as_object().unwrap().clone(),
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )?;
    decode_page(text, structured, failed)
}

fn decode_page(text: String, structured: Value, failed: bool) -> Result<ReadPage, String> {
    if failed {
        return Err(text);
    }
    let answer: command_surface::Answer<ReadPage> =
        serde_json::from_value(structured).map_err(|error| error.to_string())?;
    answer.value().cloned().ok_or(text)
}

pub(crate) fn execute_ready(archive: &Archive, ready: &ReadyCall) -> Result<ReadPage, String> {
    let name = ready.mcp["name"].as_str().expect("generated MCP tool name");
    let arguments = ready.mcp["arguments"]
        .as_object()
        .expect("generated complete envelope")
        .clone();
    let (text, structured, failed) = call_with_limit(
        archive,
        name,
        arguments,
        None,
        &tokio_util::sync::CancellationToken::new(),
        None,
    )?;
    decode_page(text, structured, failed)
}

#[test]
fn mcp_read_pages_execute_the_generated_continuation_without_losing_unicode() {
    let source = "synthetic MCP prompt λ🙂 boundary ".repeat(1000);
    let daemon = test_daemon::start(&source);
    let archive = Archive::at(&daemon.daemon.archive.root);
    let first = structured_read_without_deadline(
        &archive,
        json!({"target":"session:claude-code:s1", "page_tokens":256}),
    )
    .unwrap();
    let page = first;
    let first_next = page
        .document
        .footer
        .next
        .clone()
        .expect("bounded first page has continuation");
    let mut invalid = first_next.clone();
    invalid.mcp["arguments"]["arguments"]["view"] = json!("dialogue");
    let error = execute_ready(&archive, &invalid).unwrap_err();
    // docs/query.md: "a cursor from another question is `invalid_query`";
    // a read continuation with other arguments is refused the same way.
    assert!(error.contains("invalid_query"), "{error}");
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
    let output = structured_read_without_deadline(
        &archive,
        json!({"target":"session:claude-code:s1","page_tokens":2048}),
    )
    .unwrap();
    let page = output;
    assert!(page.document.footer.next.is_some());
    // A bounded page may finish at a header or inside encoded item JSON.
    // The client follows the actual continuation until the complete body decodes.
    let (items, pages) = collect(
        &daemon,
        page,
        None,
        Some(22_000),
        command_surface::Deadline::None,
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

#[test]
fn an_omitted_read_deadline_stays_default_in_the_next_call() {
    let daemon = test_daemon::start(&"small default read fragment ".repeat(100));
    let archive = Archive::at(&daemon.daemon.archive.root);
    let answer = structured_call_as(
        &archive,
        "read",
        args(json!({"target":"session:claude-code:s1","page_tokens":256})),
        &uuid::Uuid::now_v7().to_string(),
    );
    let page: ReadPage = serde_json::from_value(answer.value().unwrap().clone()).unwrap();
    let next = page
        .document
        .footer
        .next
        .expect("small bounded root has a next call");
    let envelope = &next.mcp["arguments"];
    assert!(
        envelope.get("deadline").is_none(),
        "default is represented by omission"
    );
    let child: crate::mcp::ToolCall = serde_json::from_value(envelope.clone()).unwrap();
    assert_eq!(child.deadline, command_surface::Deadline::Default);
    assert!(child.cursor.is_some());
    let continued = execute_ready(&archive, &next).unwrap();
    assert_eq!(continued.snapshot.query, page.snapshot.query);
}
