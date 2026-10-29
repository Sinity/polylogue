use super::*;
use polylogue_model::read::{FragmentEncoding, ItemKind, ReadPage, ReadyCall};

pub(super) fn body(page: &ReadPage) -> String {
    page.document
        .items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::Text { text }
            | ItemKind::Fragment {
                encoding: FragmentEncoding::Text,
                text,
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect()
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
    let first = call(
        &archive,
        "read",
        args(json!({"target":"session:claude-code:s1", "budget":256})),
    )
    .unwrap();
    let mut page: ReadPage = serde_json::from_str(&first).unwrap();
    let first_next = page
        .document
        .footer
        .next
        .clone()
        .expect("bounded first page has continuation");
    let mut invalid = first_next.clone();
    invalid.mcp["arguments"]["arguments"]["view"] = json!("dialogue");
    let error = execute_ready(&archive, &invalid).unwrap_err();
    assert!(error.contains("stale_cursor"), "{error}");
    let mut reconstructed = body(&page);
    let mut pages = 1;
    while let Some(mut ready) = page.document.footer.next.clone() {
        // A page budget may change while the selected question and view stay bound.
        ready.mcp["arguments"]["arguments"]["budget"] = json!(512);
        page = execute_ready(&archive, &ready).unwrap();
        reconstructed.push_str(&body(&page));
        pages += 1;
        assert!(pages < 100, "continuation failed to advance");
    }
    assert!(pages > 1);
    assert_eq!(reconstructed, source);
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
    assert!(page.records.is_empty());
    assert!(!body(&page).is_empty());
    assert!(source.starts_with(&body(&page)));
    assert!(page.document.footer.next.is_some());
    assert!(serde_json::to_vec(&page).unwrap().len() < 22_000);
    let error = call(
        &archive,
        "read",
        args(json!({"target":"session:claude-code:s1","slices":true})),
    )
    .unwrap_err();
    assert!(error.contains("bad_request"), "{error}");
}
