//! Fixtures copied from the Python Polylogue, checked against the facts its
//! tests assert (see docs/rewrite/python-port.md). The old parser is not run.
use polylogue_model::BlockContent;
use polylogue_sources::{
    codex::keyed::{KeyedRecords, MemoryReadView, Resume},
    frame::drive_jsonl,
};

#[test]
fn codex_result_status_comes_from_provider_headers_not_output_text() {
    let input = include_bytes!("fixtures/python/tool-result-headers.jsonl");
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(KeyedRecords::new(&memory, Resume::default()), input, 0);
    assert!(framed.counts.balanced());
    assert!(framed.counts.unrecognized.is_empty());
    let delta = framed.output.unwrap();
    let count = delta.resume.message_count();
    let mut memory = memory;
    memory.apply(&delta);
    let results: std::collections::BTreeMap<_, _> = (0..count)
        .flat_map(|i| memory.message(i).unwrap().unwrap().blocks)
        .filter_map(|b| match b.content {
            BlockContent::ToolResult(r) => Some((
                r.call_id.unwrap().as_str().to_owned(),
                (r.outcome.as_str().to_owned(), r.exit_code),
            )),
            _ => None,
        })
        .collect();
    for (call, outcome, exit) in [
        ("call_command_ok", "ok", Some(0)),
        ("call_command_fail", "error", Some(2)),
        ("call_command_multiline", "unknown", None),
        ("call_stdin_running", "unknown", None),
        ("call_patch_fail", "error", Some(1)),
        ("call_shell_ok", "ok", Some(0)),
        // Python ignored the exec status item; the spec now reads it.
        ("call_script_ok", "ok", None),
        ("call_script_fail", "error", None),
        ("call_prose_decoy", "unknown", None),
        ("call_script_decoy", "unknown", None),
    ] {
        assert_eq!(
            results.get(call),
            Some(&(outcome.to_owned(), exit)),
            "{call}"
        );
    }
    assert_eq!(results.len(), 10);
}

#[test]
fn claude_fixture_accepts_records_without_ids_and_keeps_each_contribution() {
    use polylogue_sources::claude_code::{FileCtx, KeyedRecords, keyed::MemoryReadView};
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("claude-normalization-main"), &memory).unwrap(),
        include_bytes!("fixtures/python/claude-normalization-main.jsonl"),
        0,
    );
    assert!(framed.counts.balanced());
    assert!(
        framed.counts.unrecognized.is_empty(),
        "{:?}",
        framed.counts.unrecognized
    );
    let output = framed.output.unwrap();
    let provisional: Vec<_> = output
        .delta
        .responses
        .iter()
        .filter(|(_, _, r)| r.message.id_kind == polylogue_model::IdKind::Provisional)
        .collect();
    assert_eq!(provisional.len(), 2);
    assert_ne!(provisional[0].2.message.id, provisional[1].2.message.id);
    assert!(
        output
            .delta
            .contributions
            .iter()
            .any(|(_, _, c)| c.blocks.iter().any(
                |b| matches!(b, BlockContent::Text{text} if text == "[Request interrupted by user]")
            ))
    );
    assert!(output.delta.contributions.iter().any(|(_, _, c)| {
        c.blocks
            .iter()
            .any(|b| matches!(b, BlockContent::Text{text} if text == "Normalization complete."))
    }));
    assert_eq!(output.delta.occurrences.len(), 2);
    assert!(
        output
            .delta
            .contributions
            .iter()
            .all(|(_, _, c)| c.projection.unresolved.is_empty()),
        "{:?}",
        output
            .delta
            .contributions
            .iter()
            .map(|(_, _, c)| &c.projection.unresolved)
            .collect::<Vec<_>>()
    );
    let background = output
        .delta
        .contributions
        .iter()
        .flat_map(|(_, _, c)| &c.blocks)
        .find_map(|b| match b {
            BlockContent::ToolResult(result)
                if result
                    .call_id
                    .as_ref()
                    .is_some_and(|id| id.as_str() == "tool-bg") =>
            {
                Some(result)
            }
            _ => None,
        })
        .unwrap();
    assert_eq!(background.outcome.as_str(), "error");
    assert_eq!(background.exit_code, Some(7));
    assert_eq!(
        background.data.as_ref().unwrap()["background_completion_status"].as_str(),
        Some("failed")
    );
    let assistant = output
        .delta
        .responses
        .iter()
        .find(|(_, _, r)| {
            r.message.model.as_deref() == Some("claude-sonnet-4-5-20250929")
                && r.message.usage.is_some()
        })
        .unwrap();
    assert_eq!(assistant.2.message.duration_ms, Some(1500));
    assert_eq!(
        assistant
            .2
            .message
            .usage
            .as_ref()
            .unwrap()
            .provider_cost_nusd,
        Some(polylogue_model::Usd(125_000_000))
    );

    assert!(output.delta.contributions.iter().any(|(_,_,c)| c.blocks.iter().any(|b| matches!(b, BlockContent::ToolCall(call) if call.call_id.as_ref().is_some_and(|id| id.as_str() == "tool-bg")))));
}

#[test]
fn identical_idless_claude_records_get_distinct_stable_occurrence_identity() {
    use polylogue_sources::claude_code::{
        FileCtx, KeyedRecords,
        keyed::{MemoryReadView, ReadView},
    };
    let record = b"{\"type\":\"user\",\"sessionId\":\"s\",\"message\":{\"content\":\"same\"}}\n";
    let mut memory = MemoryReadView::default();
    let mut ids = Vec::new();
    for _ in 0..2 {
        let start = memory.header().unwrap().parsed_upto;
        let framed = drive_jsonl(
            KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
            record,
            start,
        );
        let mut output = framed.output.unwrap();
        ids.push(output.delta.responses[0].2.message.id.clone());
        output.header.parsed_upto = framed.parsed_upto;
        memory.apply(output.header, output.delta);
    }
    assert_ne!(ids[0], ids[1]);
    let memory = MemoryReadView::default();
    let whole = [record.as_slice(), record.as_slice()].concat();
    let output = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        &whole,
        0,
    )
    .output
    .unwrap();
    assert_eq!(
        ids,
        output
            .delta
            .responses
            .iter()
            .map(|(_, _, r)| r.message.id.clone())
            .collect::<Vec<_>>()
    );
}

#[test]
fn legacy_codex_keeps_reasoning_calls_outputs_and_metadata_outcomes() {
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(&memory, Resume::default()),
        include_bytes!("fixtures/python/legacy-response-records.jsonl"),
        0,
    );
    assert!(framed.counts.balanced());
    assert!(framed.counts.unrecognized.is_empty());
    let delta = framed.output.unwrap();
    let count = delta.resume.message_count();
    let mut memory = memory;
    memory.apply(&delta);
    assert_eq!(
        memory
            .message(0)
            .unwrap()
            .unwrap()
            .at
            .unwrap()
            .at()
            .unwrap()
            .0,
        1754042400000
    );
    assert!((1..count).all(|i| memory.message(i).unwrap().unwrap().at.is_none()));

    let blocks: Vec<_> = (0..count)
        .flat_map(|i| memory.message(i).unwrap().unwrap().blocks)
        .collect();
    for (call, outcome, exit) in [
        ("call_sample_ok", "ok", 0),
        ("call_sample_fail", "error", 1),
    ] {
        assert!(blocks.iter().any(|b| matches!(&b.content, BlockContent::ToolCall(c) if c.call_id.as_ref().is_some_and(|id| id.as_str()==call))));
        assert!(blocks.iter().any(|b| matches!(&b.content, BlockContent::ToolResult(r) if r.call_id.as_ref().is_some_and(|id| id.as_str()==call) && r.outcome.as_str()==outcome && r.exit_code==Some(exit) && r.duration_ms==Some(100))));
    }
    assert!(blocks.iter().any(|b| matches!(&b.content, BlockContent::Thinking(t) if t.text.as_deref()==Some("Listing the directory first."))));
    assert!(blocks.iter().any(|b| matches!(&b.content, BlockContent::Text{text} if text=="The directory holds alpha.txt and beta.txt.")));
}

#[test]
fn named_background_completion_survives_a_later_metadata_only_uuid_supplement() {
    use polylogue_sources::claude_code::{
        FileCtx, KeyedRecords,
        keyed::{MemoryReadView, ReadView},
    };
    let mut memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("claude-normalization-main"), &memory).unwrap(),
        include_bytes!("fixtures/python/claude-normalization-main.jsonl"),
        0,
    );
    let mut output = framed.output.unwrap();
    output.header.parsed_upto = framed.parsed_upto;
    memory.apply(output.header, output.delta);
    let supplement = b"{\"type\":\"user\",\"uuid\":\"main-bg-start\",\"message\":{},\"timestamp\":\"2026-07-01T10:00:11Z\"}\n";
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("claude-normalization-main"), &memory).unwrap(),
        supplement,
        memory.header().unwrap().parsed_upto,
    );
    let output = framed.output.unwrap();
    let contribution = &output.delta.contributions[0].2;
    assert!(contribution.completion.is_some());
    assert!(
        matches!(&contribution.blocks[0], BlockContent::ToolResult(result) if result.outcome.as_str()=="error" && result.exit_code==Some(7))
    );
    assert!(contribution.projection.facts.iter().any(|f| matches!(
        f.selection.as_ref().map(|s| &s.transform),
        Some(polylogue_sources::claude_code::selection::Transform::BackgroundV1 { .. })
    )));
}
