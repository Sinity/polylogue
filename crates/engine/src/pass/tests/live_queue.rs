use super::*;
use polylogue_model::{Block, SessionId, capture_log};
use polylogue_store::fault::{self, Action};
use serde_json::json;

#[test]
fn live_refresh_keeps_each_queued_partial_first_plan_and_finishes_its_bytes() {
    let home = ClaudeHome::new();
    let chunk = 256;
    let (mut engine, _lock) = crate::import::tests::engine(&home, chunk);
    engine.config.workers = 1;
    engine.config.live_every = Duration::ZERO;
    let mut files = Vec::new();
    for name in ["a", "b"] {
        let native = format!("partial-queue-{name}");
        let text = format!("synthetic queued body {name} ").repeat(40);
        let payload = serde_json::to_vec(&json!({
            "polylogue_capture_kind":"browser_llm_session", "schema_version":1,
            "provenance":{"captured_at":"2026-10-01T00:00:00Z"},
            "session":{"provider":"gemini", "provider_session_id":native,
                "turns":[{"provider_turn_id":"m1", "role":"user", "text":text}]}
        }))
        .unwrap();
        let mut bytes = capture_log::MAGIC.to_vec();
        bytes.extend(capture_log::Header::of(&payload).encode());
        bytes.extend(payload);
        assert!(bytes.len() > 2 * chunk as usize && bytes.len() <= 8 * chunk as usize);
        let path = engine
            .config
            .spool_dir("browser-capture")
            .join(format!("gemini/{native}.json"));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let facts = FileFacts::of(&std::fs::metadata(&path).unwrap());
        files.push((path, native, text, bytes, facts));
    }
    // Each frame fits in four geometrically growing reads, eight in total.
    // This existing fault only stops an old scheduler that keeps replacing
    // those partial plans; a correct pass finishes before its seventeenth read.
    fault::clear();
    fault::arm("parse", "partial-queue-", Action::Fail, 16, u32::MAX);
    let result = engine.run_pass();
    fault::clear();
    let pass = result.unwrap();
    let reader = engine.reads.reader().unwrap();
    let firsts = reader.query(
        "SELECT input,count(*) AS n FROM a.revision WHERE produced='first' GROUP BY input ORDER BY input",
        &[],
    ).unwrap();
    assert_eq!(firsts.len(), files.len());
    assert!(
        firsts.iter().all(|r| r["n"] == 1),
        "live refresh replaced queued continuations with new first chains: {firsts:?}"
    );
    let total: u64 = files
        .iter()
        .map(|(_, _, _, bytes, _)| bytes.len() as u64)
        .sum();
    assert_eq!(pass.bytes_stored, total, "each source byte is stored once");
    assert!(
        pass.bytes_read <= total * 3,
        "geometric frame read-ahead must stay bounded: {pass:?}"
    );
    assert!(
        pass.errors.is_empty(),
        "correct work must finish before the escape fault: {pass:?}"
    );
    for (path, native, text, expected, before) in files {
        assert_eq!(FileFacts::of(&std::fs::metadata(&path).unwrap()), before);
        let session = SessionId::parse(&format!("gemini-app:{native}")).unwrap();
        let messages = reader.messages(&session, false).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(
            messages[0]
                .blocks
                .iter()
                .filter_map(Block::text)
                .collect::<String>(),
            text
        );
        let known = reader
            .known_input_at("test-machine", &path.to_string_lossy())
            .unwrap()
            .unwrap();
        assert_eq!(known.stored, expected.len() as u64);
        assert_eq!(known.parsed_upto, known.stored);
        let chain = reader
            .stored_chain(&known.input_id, known.chain.unwrap(), known.stored)
            .unwrap();
        let mut actual = Vec::new();
        chain
            .reader(Arc::clone(&engine.blobs))
            .read_to_end(&mut actual)
            .unwrap();
        assert_eq!(actual, expected);
    }
}
