use super::*;
use polylogue_model::parser_context::{Handle, Memory};
use serde_json::json;

fn records(many_sessions: bool) -> Vec<Vec<u8>> {
    let mut pieces = Vec::new();
    let record = |uuid: String, session: String| {
        format!(
            "{}\n",
            json!({
        "type":"user","uuid":uuid,"sessionId":session,"cwd":"/synthetic/project",
        "gitBranch":"main","version":"1.0","timestamp":"2026-03-01T10:00:00Z",
        "message":{"role":"user","content":"synthetic prompt"}})
        )
    };
    let mut first = record("prompt".into(), "keyed".into());
    first.push_str(&keyed_fragment("first", "response-a", "begin"));
    pieces.push(first.into_bytes());
    let mut middle = String::new();
    for n in 0..128 {
        middle.push_str(&record(
            format!("user-{n}"),
            if many_sessions {
                format!("session-{n}")
            } else {
                "keyed".into()
            },
        ));
        middle.push_str(&keyed_fragment(
            &format!("fragment-{n}"),
            &format!("response-{n}"),
            "intervening",
        ));
    }
    pieces.push(middle.into_bytes());
    pieces.push(keyed_fragment("late", "response-a", "finish").into_bytes());
    pieces
}

#[test]
fn real_parser_state_buffer_matches_independent_memory_at_every_cut() {
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let parser = (origin.open)(std::path::Path::new("/synthetic/keyed.jsonl")).unwrap();
    let expected = Handle::new(Memory::default());
    let overlay = polylogue_store::parser_context::Overlay::temporary().unwrap();
    let actual = Handle::new(overlay.clone());
    let mut expected_state = None;
    let mut actual_state = None;
    let mut offset = 0;
    for bytes in records(true) {
        let (wanted, wanted_state) = parser
            .steps(expected_state.as_ref(), &bytes, offset, false, &expected)
            .unwrap();
        let (got, got_state) = parser
            .steps(actual_state.as_ref(), &bytes, offset, false, &actual)
            .unwrap();
        expected.check().unwrap();
        actual.check().unwrap();
        assert_eq!(
            serde_json::to_value(&got).unwrap(),
            serde_json::to_value(&wanted).unwrap()
        );
        assert_eq!(got_state, wanted_state);
        offset += bytes.len() as u64;
        let delta = overlay.checkpoint_delta(offset).unwrap();
        assert_eq!(delta.through, offset);
        expected_state = Some(wanted_state);
        actual_state = Some(got_state);
    }
}

#[test]
fn state_buffer_reopen_resumes_match_one_whole_normal_parse_and_citations() {
    let pieces = records(false);
    let full = ClaudeHome::new();
    let resumed = ClaudeHome::new();
    for bytes in &pieces {
        full.append("keyed.jsonl", std::str::from_utf8(bytes).unwrap());
    }
    {
        let (mut engine, _lock) = crate::import::tests::engine(&full, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    }
    for bytes in &pieces {
        resumed.append("keyed.jsonl", std::str::from_utf8(bytes).unwrap());
        let (mut engine, _lock) = crate::import::tests::engine(&resumed, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    }
    let snapshot = |home: &ClaudeHome| {
        let (engine, _lock) = crate::import::tests::engine(home, 4096);
        let reader = engine.reads.reader().unwrap();
        let id = polylogue_model::SessionId::parse("claude-code:keyed").unwrap();
        let messages = reader.messages(&id, false).unwrap();
        let response = messages
            .iter()
            .find(|message| message.id.native().as_str() == "first")
            .unwrap();
        assert_eq!(response.text(), "begin\n\nfinish");
        let messages = serde_json::to_value(messages).unwrap();
        let index = rusqlite::Connection::open(engine.archive.index_db()).unwrap();
        let citations: Vec<(String, i64, i64)> = index
            .prepare("SELECT id,source_start,source_end FROM message ORDER BY id")
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        (messages, citations)
    };
    assert_eq!(snapshot(&resumed), snapshot(&full));
}
