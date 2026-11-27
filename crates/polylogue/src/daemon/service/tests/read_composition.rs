//! A stream of a reading and the pages of the same reading come from one
//! composition (docs/reading.md, Paging and budgets), over the Morrow archive.
use crate::daemon::service::{ArchiveMode, Daemon, Runtime};
use crate::test_daemon::{Call, read_call};
use polylogue_model::Reference;
use polylogue_model::read::*;
use polylogue_testkit::{Morrow, Set, session};

fn morrow() -> (Morrow, Runtime) {
    let archive = Morrow::build(Set::Main);
    let runtime =
        crate::test_daemon::start_runtime(&archive.root, archive.config(), ArchiveMode::Standalone)
            .unwrap();
    (archive, runtime)
}

fn request(
    target: &str,
    view: View,
    delivery: ReadDelivery,
    budget: ReadBudget,
) -> polylogue_reading::ReadRequest {
    polylogue_reading::ReadRequest {
        delivery,
        budget,
        format: if delivery == ReadDelivery::Page {
            Format::Json
        } else {
            Format::Jsonl
        },
        ..polylogue_reading::ReadRequest::new(
            target,
            ReadOptions {
                view,
                ..ReadOptions::default()
            },
        )
    }
}

/// A page's items with every item that was cut into fragments joined back
/// into the item its fragments encode. `pending` holds the encoding of an
/// item a page left unfinished.
fn joined(items: Vec<ReadItem>, pending: &mut String, out: &mut Vec<ReadItem>) {
    for item in items {
        match item.kind {
            ItemKind::Fragment {
                text, until, total, ..
            } => {
                pending.push_str(&text);
                if until == total {
                    out.push(serde_json::from_str(pending).unwrap());
                    pending.clear();
                }
            }
            _ => {
                assert!(pending.is_empty(), "an item interrupted a fragmented item");
                out.push(item);
            }
        }
    }
}

fn page(d: &Daemon, call: Call) -> ReadPage {
    crate::test_answers::read(call.answer(d))
}

/// The items of every page reached by following `next` from the first page.
fn paged(d: &Daemon, target: &str, view: View, budget: u64) -> Vec<ReadItem> {
    let mut call = read_call(request(
        target,
        view,
        ReadDelivery::Page,
        ReadBudget::Tokens(budget),
    ));
    let (mut items, mut pending) = (Vec::new(), String::new());
    loop {
        let page = page(d, call);
        joined(page.document.items, &mut pending, &mut items);
        let Some(next) = page.document.footer.next else {
            break;
        };
        call = Call::from_request(&next.request);
    }
    assert!(pending.is_empty(), "the last page left an item unfinished");
    items
}

/// The items a stream of the reading sends, in order.
fn streamed(d: &Daemon, target: &str, view: View) -> Vec<ReadItem> {
    let (bytes, answer) = read_call(request(
        target,
        view,
        ReadDelivery::Stream,
        ReadBudget::Default,
    ))
    .answer_with_events(d);
    assert!(!answer.is_error(), "{answer:?}");
    let frames = command_surface::SseDecoder::new().push(&bytes).unwrap();
    let events: Vec<&str> = frames.iter().map(|frame| frame.event.as_str()).collect();
    assert_eq!(events.first(), Some(&"begin"), "{target} {view}");
    assert_eq!(
        events.iter().filter(|event| **event == "complete").count(),
        1
    );
    let mut items = Vec::new();
    for frame in frames.iter().filter(|frame| frame.event == "record") {
        let record: ReadRecord = serde_json::from_str(&frame.data).unwrap();
        if let ReadRecord::Presentation { parts } = record {
            for part in parts {
                let ReadPart::Item(item) = part else {
                    panic!("a streamed record holds an item")
                };
                items.push(item);
            }
        }
    }
    items
}

#[test]
fn stream_equals_concatenated_pages() {
    let (_archive, runtime) = morrow();
    let d = &runtime.daemon;
    for n in 1..=8 {
        let target = Reference::Session(session(n)).to_string();
        for view in [View::Transcript, View::Dialogue, View::Context, View::Tree] {
            let stream = streamed(d, &target, view);
            assert!(!stream.is_empty(), "{target} {view}");
            // Small pages cut the reading in many places and items into
            // fragments; the default page is the common case.
            for budget in [150, 4000] {
                assert_eq!(
                    paged(d, &target, view, budget),
                    stream,
                    "{target} in the {view} view, pages of {budget} tokens"
                );
            }
        }
    }
}

#[test]
fn backward_pages_number_from_the_end() {
    let (_archive, runtime) = morrow();
    let d = &runtime.daemon;
    let first = || {
        read_call(request(
            "sessions",
            View::Transcript,
            ReadDelivery::Page,
            ReadBudget::Tokens(150),
        ))
    };
    // Forward to the last page.
    let (mut forward, mut pending) = (Vec::new(), String::new());
    let mut call = first();
    let last = loop {
        let page = page(d, call);
        joined(page.document.items.clone(), &mut pending, &mut forward);
        match &page.document.footer.next {
            Some(next) => call = Call::from_request(&next.request),
            None => break page,
        }
    };
    assert!(pending.is_empty());
    // Back from the last page to the first, through `previous`.
    let mut backward = vec![last];
    while let Some(previous) = &backward.last().unwrap().document.footer.previous {
        let page = page(d, Call::from_request(&previous.request));
        backward.push(page);
    }
    assert!(backward.len() > 2, "the reading spans several pages");
    backward.reverse();
    let mut walked = Vec::new();
    for page in &backward {
        // A page numbers its rows from its first row, whichever way it was
        // read: the first fragment on a page is in its row 0, and rows only
        // grow through the page.
        let rows: Vec<u64> = page
            .document
            .items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::Fragment { position, .. } => Some(position.selection.row),
                _ => None,
            })
            .collect();
        if let Some(first_row) = rows.first()
            && matches!(
                page.document.items.first().map(|item| &item.kind),
                Some(ItemKind::Fragment { .. })
            )
        {
            assert_eq!(*first_row, 0);
        }
        assert!(rows.windows(2).all(|pair| pair[0] <= pair[1]), "{rows:?}");
        joined(page.document.items.clone(), &mut pending, &mut walked);
    }
    assert!(pending.is_empty());
    assert_eq!(walked, forward);
}
