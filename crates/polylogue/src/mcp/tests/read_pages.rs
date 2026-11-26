use super::*;
use polylogue_model::BlobHash;
use polylogue_model::read::{ItemKind, ReadItem, ReadPage, ReadyCall, SelectionLocator};
use std::collections::{BTreeMap, BTreeSet};

type ItemKey = (SelectionLocator, u64, BlobHash);

// The public encoder owns the item JSON, including inert control escaping.
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
        assert!(self.bytes > before, "a page must advance the layout");
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

struct AllItems {
    items: Vec<ReadItem>,
    encoded: BTreeMap<BlobHash, (String, Vec<usize>)>,
    bytes: usize,
}
impl AllItems {
    fn from_root(daemon: &test_daemon::TestDaemon, next: &ReadyCall) -> Self {
        let call = crate::commands::declarations()
            .mcp_call(&next.request)
            .unwrap();
        let child = crate::commands::declarations()
            .parse_mcp_call(call["name"].as_str().unwrap(), call["arguments"].clone())
            .unwrap();
        let rows: crate::protocol::line::Rows = serde_json::from_value(child.arguments).unwrap();
        let [crate::protocol::line::Step::Read(read)] = rows.steps.as_slice() else {
            panic!("one read continuation")
        };
        let target = rows
            .selection
            .clone()
            .unwrap_or_else(|| rows.references.join(" OR "));
        let arguments = crate::daemon::verbs::read::arguments(read, target).unwrap();
        let mut request = crate::daemon::service::read::request_of(arguments, None).unwrap();
        request.delivery = polylogue_model::read::ReadDelivery::Stream;
        request.budget = polylogue_model::read::ReadBudget::Default;
        request.with_blobs = false;
        if !rows.references.is_empty() {
            request.references = Some(
                rows.references
                    .iter()
                    .map(|reference| polylogue_model::Reference::parse(reference).unwrap())
                    .collect(),
            );
        }
        let operation = daemon
            .daemon
            .operation_context(&polylogue_reading::operation::Context::worker_authorship(
                "polylogue-page-oracle",
            ))
            .unwrap();
        let mut bytes = Vec::new();
        let reader = daemon.daemon.reads.reader().unwrap();
        polylogue_reading::stream(
            &reader,
            daemon.daemon.as_ref(),
            &request,
            &operation,
            &mut bytes,
        )
        .unwrap();
        let mut items = Vec::new();
        let mut began = false;
        let mut header = None;
        let mut completed = false;
        crate::client::receive(&mut bytes.as_slice(), None, |frame| {
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
        let header = header.expect("stream header");
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
                    .expect("fragment belongs to streamed source");
                assert_eq!(
                    *total,
                    (boundaries.len() - 1) as u64,
                    "encoded source total"
                );
                let from = usize::try_from(position.character).unwrap();
                let until = usize::try_from(*until).unwrap();
                assert!(from < until && until < boundaries.len());
                assert_eq!(
                    text,
                    &source[boundaries[from]..boundaries[until]],
                    "independently streamed fragment bytes"
                );
            } else {
                let text = item_json(&page.document.header, item);
                let (source, _) = self
                    .encoded
                    .get(&BlobHash::of(text.as_bytes()))
                    .expect("complete item belongs to streamed source");
                assert_eq!(&text, source);
            }
        }
    }
}

/// Acts as a client: execute the bounded continuation, then join encoded
/// fragments by their typed coordinate and item digest before decoding content.
pub(super) fn collect(
    daemon: &test_daemon::TestDaemon,
    mut page: ReadPage,
    budget: Option<u64>,
    max_wire_bytes: Option<usize>,
    expected_deadline: command_surface::Deadline,
) -> (Items, usize) {
    let all_items = page
        .document
        .footer
        .next
        .as_ref()
        .map(|next| AllItems::from_root(daemon, next));
    let archive = Archive::at(&daemon.daemon.archive.root);
    let query = page.snapshot.query.clone();
    let format = page.format;
    let mut root = None;
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
            "bounded layout must not duplicate source records"
        );
        let wire = serde_json::to_vec(&page).unwrap();
        let footer_wire = serde_json::to_string(&page.document.footer).unwrap();
        let footer_bytes = footer_wire.len();
        for call in [&page.document.footer.next, &page.document.footer.previous]
            .into_iter()
            .flatten()
        {
            let cursor = call.request.arguments["steps"][0]["cursor"]
                .as_str()
                .expect("paged step has its cursor");
            assert_eq!(
                footer_wire.matches(cursor).count(),
                1,
                "one stored cursor per direction"
            );
        }
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
        if let Some(all_items) = &all_items {
            all_items.check(&page);
        }
        items.push(&page);
        pages += 1;
        if let Some(all_items) = &all_items {
            assert!(
                items.bytes <= all_items.bytes,
                "strict byte progress cannot exceed the finite source"
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
        let call = crate::commands::declarations()
            .mcp_call(&ready.request)
            .unwrap();
        let envelope = &call["arguments"];
        let child = crate::commands::declarations()
            .parse_mcp_call(call["name"].as_str().unwrap(), envelope.clone())
            .unwrap();
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
        let cursor = envelope["arguments"]["steps"][0]["cursor"]
            .as_str()
            .expect("opaque continuation");
        assert!(
            cursors.insert(cursor.to_owned()),
            "continuation repeated a position"
        );
        let rows: crate::protocol::line::Rows =
            serde_json::from_value(envelope["arguments"].clone()).unwrap();
        let [crate::protocol::line::Step::Read(read)] = rows.steps.as_slice() else {
            panic!("one read continuation")
        };
        // A continuation reads the same target with the same options.
        let identity = (rows.selection.clone(), rows.references.clone(), read.view);
        if let Some(root) = &root {
            assert_eq!(&identity, root, "the continuation changed its reading");
        } else {
            root = Some(identity);
        }
        if let Some(budget) = budget {
            ready.request.arguments["steps"][0]["budget"] = json!(budget);
        }
        page = execute_ready(&archive, &ready).unwrap();
    }
    items.finish();
    if let Some(all_items) = &all_items {
        assert_eq!(
            items.bytes, all_items.bytes,
            "every source byte delivered exactly once"
        );
        assert_eq!(
            items.values, all_items.items,
            "complete streamed items in source order"
        );
    }
    eprintln!(
        "delivery: pages={pages} payload={} wire={wire_total} footer={footer_total}",
        items.bytes
    );
    (items, pages)
}

/// Whole-byte reconstruction uses the documented lifted deadline at the root.
/// Children keep that accepted choice; the client never edits their deadlines.
pub(super) fn structured_read_without_deadline(
    archive: &Archive,
    arguments: Value,
) -> Result<ReadPage, String> {
    let mut request =
        command_surface::RunnableRequest::new("rows", arguments, uuid::Uuid::now_v7().to_string());
    request.deadline = command_surface::Deadline::None;
    let call = crate::commands::declarations().mcp_call(&request).unwrap();
    let envelope = &call["arguments"];
    assert_eq!(envelope.get("deadline"), Some(&Value::Null));
    let (text, structured, failed) = call_with_limit(
        archive,
        Tool::Read.name(),
        envelope.as_object().unwrap().clone(),
        None,
        &CancellationToken::new(),
        None,
    )?;
    decode_page(text, structured, failed)
}

fn decode_page(text: String, structured: Value, failed: bool) -> Result<ReadPage, String> {
    if failed {
        return Err(text);
    }
    let answer: command_surface::Answer<crate::protocol::line::LineAnswer> =
        serde_json::from_value(structured).map_err(|error| error.to_string())?;
    let Some(crate::protocol::line::LineAnswer::Steps(steps)) = answer.value() else {
        return Err(text);
    };
    let [step] = steps.steps.as_slice() else {
        return Err(text);
    };
    match &step.outcome {
        crate::protocol::line::StepOutcome::Done {
            reply: crate::protocol::line::StepAnswer::Read(page),
            ..
        } => Ok(*page.clone()),
        crate::protocol::line::StepOutcome::Failed { error } => {
            Err(format!("{:?}: {}", error.kind, error.message))
        }
        _ => Err(text),
    }
}

pub(crate) fn execute_ready(archive: &Archive, ready: &ReadyCall) -> Result<ReadPage, String> {
    let projected = crate::commands::declarations()
        .mcp_call(&ready.request)
        .unwrap();
    let name = projected["name"].as_str().expect("generated MCP tool name");
    let arguments = projected["arguments"]
        .as_object()
        .expect("generated complete envelope")
        .clone();
    let (text, structured, failed) = call_with_limit(
        archive,
        name,
        arguments,
        None,
        &CancellationToken::new(),
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
        json!({"selection":"session:claude-code:s1", "steps":[{"verb":"read", "budget":256}]}),
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
    invalid.request.arguments["steps"][0]["view"] = json!("dialogue");
    let error = execute_ready(&archive, &invalid).unwrap_err();
    // docs/query.md: "a cursor from another question is `invalid_query`";
    // a read continuation with other arguments is refused the same way.
    assert!(
        error.starts_with("error (invalid_query): the read cursor belongs to another reading: the reading options differ"),
        "{error}"
    );
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
        json!({"selection":"session:claude-code:s1", "steps":[{"verb":"read", "budget":2048}]}),
    )
    .unwrap();
    let page = output;
    assert!(page.document.footer.next.is_some());
    // A bounded page may finish at a header or inside encoded item JSON.
    // The client follows the continuation until the complete body decodes.
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
        "rows",
        args(
            json!({"selection":"session:claude-code:s1", "steps":[{"verb":"read", "slices":true}]}),
        ),
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
        "rows",
        args(
            json!({"selection":"session:claude-code:s1", "steps":[{"verb":"read", "budget":256}]}),
        ),
        &uuid::Uuid::now_v7().to_string(),
    );
    let page = decode_page(String::new(), serde_json::to_value(answer).unwrap(), false).unwrap();
    let next = page
        .document
        .footer
        .next
        .expect("small bounded root has a next call");
    let call = crate::commands::declarations()
        .mcp_call(&next.request)
        .unwrap();
    let envelope = &call["arguments"];
    assert!(
        envelope.get("deadline").is_none(),
        "default is represented by omission"
    );
    let child = crate::commands::declarations()
        .parse_mcp_call(call["name"].as_str().unwrap(), envelope.clone())
        .unwrap();
    assert_eq!(child.deadline, command_surface::Deadline::Default);
    assert!(
        child.cursor.is_none(),
        "the read step owns its continuation"
    );
    let rows: crate::protocol::line::Rows = serde_json::from_value(child.arguments).unwrap();
    let [crate::protocol::line::Step::Read(read)] = rows.steps.as_slice() else {
        panic!("one continued read")
    };
    assert!(read.cursor.is_some());
    let continued = execute_ready(&archive, &next).unwrap();
    assert_eq!(continued.snapshot.query, page.snapshot.query);
}
