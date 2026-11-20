use super::*;
use polylogue_model::{BlobHash, MessageId, SessionId};
use polylogue_store::fault::{self, Action};

fn fragment(uuid: &str, key: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"assistant","uuid":uuid,"sessionId":"import-context","message":{"id":key,"role":"assistant","content":[{"type":"text","text":text}]}})
    )
}

fn input() -> String {
    let mut bytes = fragment("first", "response-a", "First café fragment.");
    // The native parser keeps only four recent responses. The final record
    // must reopen an evicted response from a previous acknowledged range.
    for n in 0..24 {
        bytes.push_str(&fragment(
            &format!("part-{n}"),
            &format!("response-{n}"),
            &format!("Independent fragment {n}."),
        ));
    }
    bytes.push_str(&fragment("last", "response-a", "Final fragment."));
    bytes
}

fn import(engine: &mut Engine, path: &std::path::Path) -> ImportReport {
    engine
        .import_with_options(
            &[path.to_owned()],
            &ImportOptions {
                format: Some("claude-code".into()),
                ..Default::default()
            },
        )
        .unwrap()
}

fn populated(home: &ClaudeHome) -> (Engine, polylogue_store::ArchiveLock) {
    home.append(
        "unrelated.jsonl",
        &claude_user("seed", "seed", "Already indexed."),
    );
    let (mut engine, lock) = engine(home, 256);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(home.texts("seed"), ["Already indexed."]);
    (engine, lock)
}

fn assert_reading(engine: &Engine, bytes: &str) {
    let reader = engine.reads.reader().unwrap();
    let messages = reader
        .messages(
            &SessionId::parse("claude-code:import-context").unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(messages.len(), 25);
    assert_eq!(
        messages
            .iter()
            .find(|m| m.id.native().as_str() == "first")
            .unwrap()
            .text(),
        "First café fragment.\n\nFinal fragment."
    );
    let late = MessageId::parse("claude-code:last").unwrap();
    let resolved = reader
        .message_with_block_selection(&late)
        .unwrap()
        .unwrap()
        .0;
    assert_eq!(resolved.id.native().as_str(), "first");
    let input = BlobHash::of(bytes.as_bytes()).to_string();
    let progress = reader.input_progress(&input).unwrap().unwrap();
    assert!(progress.parsed);
    assert_eq!(progress.parsed_upto, bytes.len() as u64);
    assert_eq!(
        reader
            .stored_range(&input, progress.chain, 0, bytes.len() as u64, &engine.blobs)
            .unwrap(),
        bytes.as_bytes()
    );
    let cited = reader
        .stored_record(
            polylogue_store::items::RecordTable::Message,
            &resolved.id.to_string(),
        )
        .unwrap()
        .unwrap();
    assert_eq!(cited.input, input);
    assert_eq!(cited.bytes, bytes.as_bytes());
}

#[test]
fn an_immutable_import_keeps_its_context_without_opening_the_populated_index() {
    fault::clear();
    let home = ClaudeHome::new();
    let path = home.dir.path().join("import-context.jsonl");
    let bytes = input();
    std::fs::write(&path, &bytes).unwrap();
    let input = BlobHash::of(bytes.as_bytes()).to_string();
    let (mut engine, _lock) = populated(&home);
    fault::arm(
        "parser_context.open_committed",
        &input,
        Action::Fail,
        0,
        1000,
    );
    let report = import(&mut engine, &path);
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert!(report.pass.bytes_read >= bytes.len() as u64);
    assert_reading(&engine, &bytes);
    // The same armed boundary must still reject a real committed-context
    // open, proving the successful import did not silently consume it.
    let progress = engine
        .reads
        .reader()
        .unwrap()
        .input_progress(&input)
        .unwrap()
        .unwrap();
    let origin = polylogue_sources::origin::by_name("claude-code").unwrap();
    let rejected = polylogue_store::parser_context::Overlay::open(
        &engine.archive,
        &input,
        Some(progress.chain),
        &origin.reading_version(),
        progress.parsed_upto,
    );
    assert!(
        matches!(rejected, Err(ref error) if error.to_string().contains("parser_context.open_committed"))
    );
    fault::clear();
    let revisions = home.count("SELECT count(*) AS n FROM a.revision");
    let repeated = import(&mut engine, &path);
    assert!(repeated.pass.errors.is_empty(), "{repeated:?}");
    assert_eq!(repeated.pass.bytes_read, 0);
    assert_eq!(repeated.pass.bytes_stored, 0);
    assert_eq!(
        home.count("SELECT count(*) AS n FROM a.revision"),
        revisions
    );
    assert_reading(&engine, &bytes);
}

#[test]
fn a_rejected_import_range_resumes_from_its_committed_context() {
    fault::clear();
    let home = ClaudeHome::new();
    let path = home.dir.path().join("interrupted-context.jsonl");
    let bytes = input();
    std::fs::write(&path, &bytes).unwrap();
    let input = BlobHash::of(bytes.as_bytes()).to_string();
    let (mut engine, _lock) = populated(&home);
    let index = rusqlite::Connection::open(home.index_db()).unwrap();
    index.execute_batch("CREATE TRIGGER reject_context_range BEFORE INSERT ON message WHEN NEW.id='claude-code:part-12' BEGIN SELECT RAISE(ABORT,'synthetic range refusal'); END;").unwrap();
    let failed = import(&mut engine, &path);
    assert!(!failed.pass.errors.is_empty(), "{failed:?}");
    let before = engine
        .reads
        .reader()
        .unwrap()
        .input_progress(&input)
        .unwrap()
        .unwrap();
    assert!(before.parsed_upto > 0 && before.parsed_upto < bytes.len() as u64);
    index
        .execute_batch("DROP TRIGGER reject_context_range")
        .unwrap();
    drop(index);
    // A later attempt must reopen the committed context. The previous
    // private scratch view cannot stand for a failed or restarted reading.
    fault::arm(
        "parser_context.open_committed",
        &input,
        Action::Fail,
        0,
        1000,
    );
    let unavailable = import(&mut engine, &path);
    assert!(
        unavailable
            .pass
            .errors
            .iter()
            .any(|error| error.contains("parser_context.open_committed")),
        "{unavailable:?}"
    );
    let after = engine
        .reads
        .reader()
        .unwrap()
        .input_progress(&input)
        .unwrap()
        .unwrap();
    assert_eq!(after.parsed_upto, before.parsed_upto);
    fault::clear();
    let retry = import(&mut engine, &path);
    assert!(retry.pass.errors.is_empty(), "{retry:?}");
    assert!(retry.pass.bytes_read < bytes.len() as u64, "{retry:?}");
    assert_eq!(retry.pass.bytes_stored, 0);
    assert_reading(&engine, &bytes);
    assert_eq!(home.texts("seed"), ["Already indexed."]);
}
