use super::*;
use crate::parse::run;
use serde_json::json;

/// The rule as a file found at a location meets it: its first bytes alone.
fn recognize(path: &std::path::Path, head: &[u8]) -> bool {
    super::recognize(path, &crate::document::Head::new(head))
}

fn line(row: Json) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(
        &json!({"t":"conversation_summaries", "k":[row["conversation_id"]], "r":row}),
    )
    .unwrap();
    bytes.push(b'\n');
    bytes
}

#[test]
fn antigravity_summary_offers_title_workspaces_parent_and_source_time() {
    let row = json!({"conversation_id":"child", "title":"Check the fixture", "parent_conversation_id":"parent", "workspace_uris":"[\"file:///work/one\",\"file:///work/two\",\"https://example.invalid/remote\"]", "last_modified_time":"2026-03-05 05:21:34.123456789+01:00", "preview":"summary only"});
    let bytes = line(row.clone());
    let step = run(&Summaries, &mut NoState, &bytes, 50).unwrap();
    let session = &step.output.sessions[0];
    assert_eq!(session.id.to_string(), "antigravity:child");
    assert_eq!(session.kind, SessionKind::Subagent);
    assert_eq!(session.title.as_ref().unwrap().source, TitleSource::Summary);
    assert_eq!(session.title.as_ref().unwrap().text, "Check the fixture");
    assert_eq!(session.cwd, ["/work/one", "/work/two"]);
    assert!(
        session.started.is_none(),
        "a last-modified time is not a start"
    );
    assert!(
        step.output.messages.is_empty(),
        "a preview is not a message"
    );
    let link = &step.output.links[0];
    assert_eq!(link.kind, LinkKind::Spawn);
    assert_eq!(link.parent.to_string(), "antigravity:parent");
    assert_eq!(link.evidence, "parent_conversation_id");
    let event = &step.output.events[0];
    assert_eq!(event.at.unwrap().unix_ns(), 1_772_684_494_123_456_789);
    assert_eq!(
        event.at_precision,
        Some(polylogue_model::Precision::Instant)
    );
    assert_eq!(event.range, Some((50, 50 + bytes.len() as u64 - 1)));
    assert_eq!(event.data, row);
}

#[test]
fn missing_or_local_summary_times_stay_unknown_and_invalid_rows_are_accounted() {
    let bytes = [
        line(json!({"conversation_id":"no-time", "workspace_uris":"not json", "last_modified_time":"2026-03-05 04:21:34"})),
        line(json!({"conversation_id":"", "title":"No ID"})),
        b"{\"t\":\"future_table\",\"r\":{\"unknown\":1}}\n".to_vec(),
        b"{\"t\":\"conversation_summaries\",\"k\":[\"gone\"],\"gone\":true}\n".to_vec(),
    ].concat();
    let step = run(&Summaries, &mut NoState, &bytes, 0).unwrap();
    assert_eq!(step.output.sessions.len(), 1);
    assert!(step.output.sessions[0].cwd.is_empty());
    assert!(step.output.events[0].at.is_none());
    assert_eq!(
        step.output.events[0].at_quality,
        polylogue_model::TimeQuality::Unknown
    );
    assert_eq!(step.counts.records, 4);
    assert_eq!(step.counts.used.values().sum::<u64>(), 1);
    assert_eq!(step.counts.unreadable.values().sum::<u64>(), 1);
    assert_eq!(step.counts.skipped.values().sum::<u64>(), 2);
    assert!(
        step.counts
            .wire_shapes
            .values()
            .any(|shape| shape.unknown_table_count == 1)
    );
}

#[test]
fn summary_origin_does_not_claim_binary_trajectory_databases() {
    let summary = Path::new("/home/example/.gemini/antigravity-cli/conversation_summaries.db");
    assert!(is_database(summary));
    assert!(open(summary).is_some());
    assert!(recognize(
        Path::new("conversation_summaries.db.jsonl"),
        b"{\"schema\":{}}"
    ));
    for path in [
        "conversations/native.db",
        "conversation_summaries.db-wal",
        "state.vscdb",
    ] {
        assert!(!is_row_dump(Path::new(path)), "{path}");
        assert!(open(Path::new(path)).is_none());
    }
}
