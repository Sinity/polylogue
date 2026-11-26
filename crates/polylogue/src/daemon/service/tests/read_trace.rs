use super::*;
use crate::test_daemon::Call;
use polylogue_model::{BlobHash, Reference, read::*};
use serde_json::json;

fn read(target: &str) -> Call {
    Call::new(
        "rows",
        json!({"selection":target,"steps":[{"verb":"read","view":"trace","budget":100_000}]}),
    )
}
fn card(page: &ReadPage) -> &TraceCard {
    let [
        ReadItem {
            kind: ItemKind::Card { card },
            ..
        },
    ] = page.document.items.as_slice()
    else {
        panic!("trace card: {:?}", page.document)
    };
    let Card::Trace(trace) = card.as_ref() else {
        panic!("trace card")
    };
    trace
}

#[test]
fn trace_shows_record_chain_and_names_unbuilt_sources() {
    let text = "neutral trace text";
    let original = polylogue_testkit::claude_user("u1", "s1", text).into_bytes();
    let mut d = test_daemon::start(text);
    d.pause_signals();
    let first = crate::test_answers::read(read("message:claude-code:u1").answer(&d.daemon));
    let trace = card(&first);
    assert_eq!(
        trace.item,
        Reference::parse("message:claude-code:u1").unwrap()
    );
    assert_eq!(first.document.header.view, View::Trace);
    assert_eq!(first.document.header.target, "message:claude-code:u1");
    assert_eq!(
        trace.missing,
        ["quote origins", "paste origins", "file-line lineage"]
    );
    let (input, chain, start, end) = trace
        .chain
        .iter()
        .find_map(|step| match step {
            ChainStep::InputRecord {
                input,
                chain,
                start,
                end,
                route: polylogue_model::Route::Live,
                shown: true,
            } => Some((input.clone(), *chain, *start, *end)),
            _ => None,
        })
        .unwrap();
    assert_eq!((start, end), (0, original.len() as u64 - 1));
    assert!(trace.chain.iter().any(|step|matches!(step,ChainStep::StoredBytes{blob,size,offset:0,blob_offset:0,..} if *blob==BlobHash::of(&original) && *size==original.len() as u64)));
    let reader = d.daemon.reads.reader().unwrap();
    assert_eq!(
        reader
            .stored_range(
                &input,
                chain,
                start,
                end,
                &d.daemon.archive.blob_reader().unwrap()
            )
            .unwrap(),
        original[..original.len() - 1]
    );
    drop(reader);
    for target in [
        "block:claude-code:u1#0",
        "span:message:claude-code:u1#c=0-7",
    ] {
        let page = crate::test_answers::read(read(target).answer(&d.daemon));
        assert_eq!(card(&page).item, Reference::parse(target).unwrap());
        assert_eq!(
            card(&page)
                .chain
                .iter()
                .filter(|step| matches!(step, ChainStep::StoredBytes { .. }))
                .count(),
            1
        );
    }
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let text = encoded(&first, format);
        assert!(
            !text.contains("neutral trace text"),
            "trace is provenance, not transcript"
        );
        assert!(text.contains("Stored bytes:"));
        assert!(text.contains("quote origins: not built yet"));
    }
    d.restart();
    d.pause_signals();
    let repeated = crate::test_answers::read(read("message:claude-code:u1").answer(&d.daemon));
    assert_eq!(repeated.document.items, first.document.items);
}

#[test]
fn trace_of_file_line_and_other_rows_is_explicitly_not_built() {
    let d = test_daemon::start("trace target");
    for target in ["file:src/main.rs#L12", "session:claude-code:s1"] {
        let failure = crate::test_answers::refusal(read(target).answer(&d.daemon));
        assert_eq!(
            failure.code.as_deref(),
            Some("not_built"),
            "{target}: {failure:?}"
        );
    }
    let failure =
        crate::test_answers::refusal(read("message:claude-code:missing").answer(&d.daemon));
    assert_eq!(failure.code.as_deref(), Some("not_found"));
}

#[test]
fn trace_pages_keep_every_source_after_restart() {
    let text = "neutral trace copies";
    let seed = |home: &std::path::Path| {
        let copies = home.join(".claude/projects/-copies");
        std::fs::create_dir_all(&copies).unwrap();
        for n in 0..8 {
            std::fs::write(
                copies.join(format!("copy-{n}.jsonl")),
                polylogue_testkit::claude_user("u1", "s1", text),
            )
            .unwrap();
        }
    };
    let mut d = test_daemon::start_with_sources_and_workers(text, &seed, 0, 2);
    d.wait_for(|_| {
        d.daemon
            .reads
            .reader()
            .unwrap()
            .known_inputs("synthetic-machine")
            .unwrap()
            .iter()
            .filter(|input| {
                input.format == "claude-code"
                    && input.state.parsed_to() == input.stored
                    && input.stored > 0
            })
            .count()
            == 9
    });
    d.pause_signals();
    let mut request = read("message:claude-code:u1");
    request.arguments["steps"][0]["budget"] = json!(512);
    let mut page = crate::test_answers::read(request.answer(&d.daemon));
    assert!(
        page.document.footer.next.is_some(),
        "all source routes need several pages"
    );
    d.restart();
    d.pause_signals();
    let mut combined = String::new();
    let mut pages = 0;
    loop {
        pages += 1;
        assert!(pages <= 64, "trace must finish");
        for item in &page.document.items {
            let ItemKind::Fragment {
                text,
                position,
                until,
                ..
            } = &item.kind
            else {
                panic!("one paged trace item: {item:?}")
            };
            assert_eq!(position.character, combined.chars().count() as u64);
            combined.push_str(text);
            assert_eq!(*until, combined.chars().count() as u64);
        }
        let Some(next) = page.document.footer.next else {
            break;
        };
        page = crate::test_answers::read(Call::from_request(&next.request).answer(&d.daemon));
    }
    let restored: ReadItem = serde_json::from_str(&combined).unwrap();
    let ItemKind::Card { card } = restored.kind else {
        panic!("original card")
    };
    let Card::Trace(trace) = *card else {
        panic!("trace card")
    };
    assert_eq!(
        trace.item,
        Reference::parse("message:claude-code:u1").unwrap()
    );
    assert_eq!(
        trace
            .chain
            .iter()
            .filter(|step| matches!(step, ChainStep::StoredBytes { .. }))
            .count(),
        9
    );
}
