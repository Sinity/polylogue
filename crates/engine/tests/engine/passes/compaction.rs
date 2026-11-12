//! Native retained context crosses parsing, writing and the stored item index.
use super::*;
use polylogue_model::read::{EntryKind, Part, Place, context_at};

fn matched(home: &ClaudeHome, text: &str) -> u64 {
    let options = polylogue_store::query::Options {
        clock: query_language::time::Clock::new(
            "2026-10-29T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::time(0, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        page_size: Some(20),
        random_seed: Some(7),
        embeddings: false,
        self_session: None,
        saved: Default::default(),
        fields: vec!["id".into()],
        kept: Default::default(),
        chat_pages: vec![],
    };
    let program =
        polylogue_store::query::compile(&format!("messages where {text}"), &options).unwrap();
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    reader.query_count(&program).unwrap().unwrap().0
}

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
        assert_eq!(matched(&home, "\"Retained synthetic context\""), 1);
        assert!(
            reader
                .prose_texts(0, 20, Some(&["claude-code:s".to_owned()]))
                .unwrap()
                .iter()
                .any(|text| text.text == "Retained synthetic context.")
        );
    }
}

#[test]
fn unknown_compaction_replays_keep_native_uuid_ownership_in_both_arrival_orders() {
    for subagent in [false, true] {
        for carrier_first in [false, true] {
            let home = ClaudeHome::new();
            let directory = home.file("root/subagents");
            fs::create_dir_all(&directory).unwrap();
            let known = if subagent {
                serde_json::json!({"type":"user","uuid":"shared-native","sessionId":"root","agentId":"worker","isSidechain":true,"message":{"role":"user","content":"known original prose"}})
            } else {
                serde_json::from_str::<serde_json::Value>(&user(
                    "shared-native",
                    "known",
                    "known original prose",
                ))
                .unwrap()
            };
            let known_path = if subagent {
                directory.join("agent-worker.jsonl")
            } else {
                home.file("known.jsonl")
            };
            let carrier_path = directory.join("agent-acompact-one.jsonl");
            let carrier = serde_json::json!({"type":"user","uuid":"shared-native","sessionId":"root-carrier","agentId":"acompact-one","isSidechain":true,"message":{"role":"user","content":"copied alternative prose"}});
            let unique = serde_json::json!({"type":"user","uuid":"unique-carrier","sessionId":"root-carrier","agentId":"acompact-one","isSidechain":true,"message":{"role":"user","content":"uniquecarrierword"}});
            let (mut engine, _lock) = home.engine(64 << 20);
            for put_carrier in [carrier_first, !carrier_first] {
                if put_carrier {
                    fs::write(&carrier_path, format!("{carrier}\n{unique}\n")).unwrap();
                } else {
                    fs::write(&known_path, format!("{known}\n")).unwrap();
                }
                let pass = engine.run_pass().unwrap();
                assert!(pass.errors.is_empty(), "{pass:?}");
            }
            let pool = home.reads();
            let reader = pool.reader().unwrap();
            let id = polylogue_model::MessageId::parse("claude-code:shared-native").unwrap();
            let message = reader.message_with_projection(&id).unwrap().unwrap().0;
            let owner = if subagent {
                "claude-code:agent-worker"
            } else {
                "claude-code:known"
            };
            assert_eq!(message.session.to_string(), owner);
            assert_eq!(message.text(), "known original prose");
            assert_eq!(reader.query("SELECT count(*) AS n FROM message_source s JOIN message m ON m.pk=s.message_pk WHERE m.id='claude-code:shared-native'",&[]).unwrap()[0]["n"],2);
            assert_eq!(matched(&home, "uniquecarrierword"), 1);
        }
    }
}
