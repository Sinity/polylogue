use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

fn fragment(uuid: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"assistant","uuid":uuid,"sessionId":"continuation","requestId":"same-request","timestamp":"2026-03-01T10:00:01Z","message":{"id":"same-response","role":"assistant","model":"fixture-model","content":[{"type":"text","text":text}]}})
    )
}

#[test]
fn ordinary_restart_resumes_response_assembly_without_rereading_provider_prefix() {
    let home = ClaudeHome::new();
    home.append(
        "continuation.jsonl",
        &claude_user("prompt", "continuation", &"prefix ".repeat(32768)),
    );
    home.append("continuation.jsonl", &fragment("first", "First fragment."));
    let prefix = std::fs::metadata(home.file("continuation.jsonl"))
        .unwrap()
        .len();
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        let state = engine.inputs.get(&home.file("continuation.jsonl")).unwrap();
        let reader = engine.reads.reader().unwrap();
        let progress = reader.input_progress(&state.input_id).unwrap().unwrap();
        let stored = reader
            .parser_checkpoint(
                &state.input_id,
                progress.chain,
                &state.origin.reading_version(),
            )
            .unwrap()
            .unwrap();
        let checkpoint = stored.checkpoint.unwrap();
        assert_eq!(checkpoint.parsed_upto, prefix);
        assert!(
            checkpoint.parser_state.unwrap()["sessions"]
                .get("continuation")
                .is_some()
        );
    }
    home.append("continuation.jsonl", &fragment("last", "Final fragment."));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(
        pass.bytes_read < prefix,
        "restart read the provider prefix: {pass:?}"
    );
    let session = polylogue_model::SessionId::parse("claude-code:continuation").unwrap();
    let reader = engine.reads.reader().unwrap();
    let messages = reader.messages(&session, false).unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[1].blocks.len(), 2);
    assert_eq!(messages[1].blocks[0].text(), Some("First fragment."));
    assert_eq!(messages[1].blocks[1].text(), Some("Final fragment."));
}

fn job(
    engine: &Engine,
    path: &std::path::Path,
    origin: &'static polylogue_sources::OriginSpec,
    size: u64,
) -> crate::pool::Job {
    crate::pool::Job {
        acquisition: None,
        path: path.to_path_buf(),
        origin,
        input: polylogue_store::InputKey {
            id: engine.location_id(path),
            format: origin.name.into(),
            arrival_rank: origin.arrival.rank(),
        },
        change: polylogue_store::Change::First,
        observation: Some(engine.observation(
            path,
            Some(polylogue_store::FileFacts::of(
                &std::fs::metadata(path).unwrap(),
            )),
        )),
        continues: false,
        start: 0,
        store_from: Some(0),
        end: size,
        at_end: true,
        length: size,
        parser_state: None,
        parser_deferred: false,
        priority: crate::pool::Priority::Bulk,
        source: crate::pool::Bytes::File,
    }
}

#[test]
fn missing_same_version_state_defers_parse_while_every_raw_suffix_chunk_commits() {
    let home = ClaudeHome::new();
    home.append(
        "missing.jsonl",
        &claude_user("first", "missing", &"prefix ".repeat(4096)),
    );
    let path = home.file("missing.jsonl");
    let prefix = std::fs::metadata(&path).unwrap().len();
    let input;
    {
        let (engine, _lock) = crate::import::tests::engine(&home, 1024);
        let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
        let first = job(&engine, &path, origin, prefix);
        input = first.input.id.clone();
        let (batch, done) = crate::pool::run(first, &engine.archive, &engine.blobs, true);
        assert!(done.error.is_none());
        let mut batch = batch.unwrap();
        // A retained source reading whose parser control is unavailable.
        batch.batch.parse.last_mut().unwrap().checkpoint = None;
        let result = engine
            .writer
            .sender()
            .unwrap()
            .store_group(vec![batch], std::sync::Arc::clone(&engine.blobs))
            .unwrap();
        assert!(result.into_iter().all(|r| r.is_ok()));
    }
    let mut suffix = String::new();
    for index in 0..20 {
        suffix.push_str(&claude_user(
            &format!("later-{index}"),
            "missing",
            &"suffix ".repeat(80),
        ));
    }
    home.append("missing.jsonl", &suffix);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
    let pass = engine.run_pass().unwrap();
    assert_eq!(pass.errors.len(), 1, "{pass:?}");
    assert!(
        pass.errors[0].contains("parser continuation unavailable"),
        "{pass:?}"
    );
    assert_eq!(pass.bytes_stored, suffix.len() as u64);
    let reader = engine.reads.reader().unwrap();
    let progress = reader.input_progress(&input).unwrap().unwrap();
    assert_eq!(progress.length, prefix + suffix.len() as u64);
    assert_eq!(progress.parsed_upto, prefix);
    assert_eq!(
        reader
            .stored_range(
                &input,
                progress.chain,
                prefix,
                progress.length,
                &engine.blobs
            )
            .unwrap(),
        suffix.as_bytes()
    );
    assert_eq!(home.texts("missing").len(), 1);
}

#[test]
fn one_call_multiple_capture_steps_only_checkpoints_its_final_cursor() {
    let home = ClaudeHome::new();
    let path = home.dir.path().join("captures.json");
    let payload = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../examples/sources/browser-capture/python-chatgpt-capture/input.json"
    ));
    let mut bytes = polylogue_model::capture_log::MAGIC.to_vec();
    for _ in 0..2 {
        bytes.extend(polylogue_model::capture_log::Header::of(payload).encode());
        bytes.extend(payload);
    }
    std::fs::write(&path, &bytes).unwrap();
    let (engine, _lock) = crate::import::tests::engine(&home, 1024);
    let origin = polylogue_sources::origin::by_name("browser-capture").unwrap();
    let (batch, done) = crate::pool::run(
        job(&engine, &path, origin, bytes.len() as u64),
        &engine.archive,
        &engine.blobs,
        false,
    );
    assert!(done.error.is_none(), "{done:?}");
    let steps = batch.unwrap().batch.parse;
    assert_eq!(steps.len(), 2);
    assert!(steps[0].checkpoint.is_none());
    let last = steps[1].checkpoint.as_ref().unwrap();
    assert_eq!(last.parsed_upto, bytes.len() as u64);
    assert!(last.matches(&origin.reading_version(), bytes.len() as u64));
}

#[test]
fn raw_only_capture_keeps_bytes_without_inventing_a_parser_checkpoint_or_replaying_them() {
    let home = ClaudeHome::new();
    let dir = home.home.join(".pi/agent/sessions");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("native.jsonl");
    let bytes = b"{\"id\":\"synthetic-native\",\"content\":\"captured raw\"}\n";
    std::fs::write(&path, bytes).unwrap();
    for attempt in 0..3 {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 1024);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        if attempt > 0 {
            assert_eq!(pass.bytes_read, 0, "{pass:?}");
            assert_eq!(pass.bytes_stored, 0, "{pass:?}");
        }
        let input = engine.location_id(&path);
        let reader = engine.reads.reader().unwrap();
        let progress = reader.input_progress(&input).unwrap().unwrap();
        assert_eq!(progress.length, bytes.len() as u64);
        assert_eq!(progress.parsed_upto, 0);
        assert!(!progress.parsed);
        assert!(
            reader
                .parser_checkpoint(&input, progress.chain, "")
                .unwrap()
                .is_none()
        );
        let rows = reader.query("SELECT count(*) AS n FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1", &[polylogue_store::read::SqlValue::Text(input)]).unwrap();
        assert_eq!(rows[0]["n"], 0);
        let completeness = reader.completeness(&crate::reading_contracts()).unwrap();
        assert!(
            matches!(completeness.parsing, polylogue_store::CompletenessDimension::Incomplete(ref reasons) if reasons.contains(&polylogue_store::CompletenessReason::KnownRawOnly))
        );
        assert_eq!(completeness.raw_only_chains, 1);
        assert!(!completeness.caught_up());
    }
}

#[test]
fn declared_registry_keeps_absent_raw_only_origins_valid_and_real_row_dump_versions() {
    use polylogue_store::{CompletenessDimension, ReadingContract};
    let home = ClaudeHome::new();
    let (engine, _lock) = crate::import::tests::engine(&home, 1024);
    let contracts = crate::reading_contracts();
    let report = engine
        .reads
        .reader()
        .unwrap()
        .completeness(&contracts)
        .unwrap();
    assert_eq!(report.parsing, CompletenessDimension::Complete);
    assert_eq!(report.raw_only_chains, 0);
    assert_eq!(report.kept_chains, 0);
    for name in ["pi", "tool-context"] {
        assert!(matches!(
            contracts.iter().find(|contract| contract.origin() == name),
            Some(ReadingContract::RawOnly { .. })
        ));
    }
    for name in ["llm", "tool-database"] {
        let origin = polylogue_sources::origin::by_name(name).unwrap();
        assert!(
            matches!(contracts.iter().find(|contract| contract.origin()==name), Some(ReadingContract::Expected { version, .. }) if !version.is_empty() && version==&origin.reading_version())
        );
    }
    assert!(matches!(
        contracts
            .iter()
            .find(|contract| contract.origin() == "export-file"),
        Some(ReadingContract::Kept { .. })
    ));
}
