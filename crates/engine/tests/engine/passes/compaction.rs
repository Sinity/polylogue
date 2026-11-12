//! Native retained context crosses parsing, writing and the stored item index.
use super::*;
use polylogue_model::read::{EntryKind, Part, Place, context_at};

#[test]
fn native_compaction_kept_ids_and_ranges_reach_stored_context() {
    for (kept, expected, evicted) in [
        (
            serde_json::json!({"preservedMessages":{"uuids":["third","first"],"anchorUuid":"summary"}}),
            vec!["summary", "third", "first", "tail"],
            vec!["second"],
        ),
        (
            serde_json::json!({"preservedSegment":{"headUuid":"second","tailUuid":"third","anchorUuid":"summary"}}),
            vec!["summary", "second", "third", "tail"],
            vec!["first"],
        ),
    ] {
        let home = ClaudeHome::new();
        let boundary = serde_json::json!({"type":"system","subtype":"compact_boundary","uuid":"boundary","sessionId":"s","logicalParentUuid":"third","compactMetadata":kept});
        let summary = serde_json::json!({"type":"user","uuid":"summary","sessionId":"s","parentUuid":"boundary","isCompactSummary":true,"message":{"role":"user","content":"Retained synthetic context."}});
        home.append(
            "s.jsonl",
            &(user("first", "s", "first")
                + &user("second", "s", "second")
                + &user("third", "s", "third")
                + &format!("{boundary}\n{summary}\n")
                + &user("tail", "s", "tail")),
        );
        let (mut engine, _lock) = home.engine(64 << 20);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        let pool = home.reads();
        let reader = pool.reader().unwrap();
        let session = SessionId::parse("claude-code:s").unwrap();
        let index = reader.item_index(&session).unwrap().unwrap();
        let entries = index.entries;
        let parts = [Part {
            session,
            start: 0,
            end: entries.len(),
            inherited: false,
            joined_by: None,
        }];
        let context = context_at(
            &parts,
            |_| &entries,
            Place {
                part: 0,
                entry: entries.len() - 1,
            },
        );
        assert!(context.complete, "{context:?}");
        let headers = |places: &[Place]| {
            places
                .iter()
                .filter_map(|p| {
                    let entry = &entries[p.entry];
                    (entry.kind == EntryKind::Header)
                        .then(|| entry.key.message().unwrap().native().as_str().to_owned())
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(headers(&context.members), expected);
        assert_eq!(headers(&context.evicted), evicted);
    }
}
