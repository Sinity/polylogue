mod appends;

use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

fn fragment(uuid: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"assistant","uuid":uuid,"sessionId":"continuation","requestId":"same-request","timestamp":"2026-03-01T10:00:01Z","message":{"id":"same-response","role":"assistant","model":"fixture-model","content":[{"type":"text","text":text}]}})
    )
}

#[test]
fn restart_resumes_response_assembly_after_checking_the_stored_prefix() {
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
        let progress = reader.latest_chain(&state.input_id).unwrap().unwrap();
        let stored = reader.chain_state(&state.input_id, progress.chain).unwrap();
        assert_eq!(stored.parsed_to(), prefix);
        assert_eq!(
            stored.version(),
            Some(state.origin.parse_version().as_str())
        );
        assert!(stored.resume().is_some());
    }
    let suffix = fragment("last", "Final fragment.");
    home.append("continuation.jsonl", &suffix);
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_read, prefix + suffix.len() as u64, "{pass:?}");
    assert_eq!(pass.bytes_stored, suffix.len() as u64, "{pass:?}");
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
        source_read: None,
        frozen_file: None,
        path: path.to_path_buf(),
        origin,
        input: polylogue_store::InputKey {
            id: engine.location_id(path),
            format: origin.name.into(),
            route: origin.route,
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
        finished: false,
        length: size,
        parser_state: None,
        parser_chain: None,
        store_only: false,
        priority: crate::pool::Priority::Bulk,
        source: crate::pool::Bytes::File,
    }
}

#[test]
fn missing_parser_state_reads_the_chain_again_from_its_start() {
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
        let (batch, done) = crate::pool::run(first, &engine.archive, &engine.blobs, true, None);
        assert!(done.error.is_none());
        let mut batch = batch.unwrap();
        // A stored source reading whose parser control is unavailable.
        batch.batch.parse.last_mut().unwrap().resume = None;
        let result = engine.writer.store_group(vec![batch]).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_ok());
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
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_stored, suffix.len() as u64);
    let reader = engine.reads.reader().unwrap();
    let progress = reader.latest_chain(&input).unwrap().unwrap();
    assert_eq!(progress.stored, prefix + suffix.len() as u64);
    // docs/archive.md, Parse versions: without its carried state the
    // chain is read from its start, which gives the same reading.
    assert_eq!(progress.state.parsed_to(), prefix + suffix.len() as u64);
    assert_eq!(
        reader
            .stored_range(
                &input,
                progress.chain,
                prefix,
                progress.stored,
                &engine.blobs
            )
            .unwrap(),
        suffix.as_bytes()
    );
    assert_eq!(home.texts("missing").len(), 21);
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
        None,
    );
    assert!(done.error.is_none(), "{done:?}");
    let steps = batch.unwrap().batch.parse;
    assert_eq!(steps.len(), 2);
    assert!(steps[0].resume.is_none());
    assert!(steps[1].resume.is_some());
    assert_eq!(steps[1].parsed_upto, bytes.len() as u64);
    assert_eq!(steps[1].parse_version, origin.parse_version());
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
        let progress = reader.latest_chain(&input).unwrap().unwrap();
        assert_eq!(progress.stored, bytes.len() as u64);
        assert_eq!(progress.state.parsed_to(), 0);
        assert!((progress.state == polylogue_model::chain::ChainState::Unparsed));
        let rows = reader.query("SELECT count(*) AS n FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk WHERE i.id=?1", &[polylogue_store::read::SqlValue::Text(input)]).unwrap();
        assert_eq!(rows[0]["n"], 0);
        let completeness = reader.completeness(&crate::reading_contracts()).unwrap();
        assert_eq!(
            completeness.parsing,
            polylogue_model::freshness::CompletenessDimension::Complete
        );
        assert_eq!(completeness.raw_only_chains, Some(1));
        assert!(completeness.caught_up());
    }
}

#[test]
fn declared_registry_keeps_absent_raw_only_origins_valid_and_real_row_dump_versions() {
    use polylogue_model::freshness::CompletenessDimension;
    use polylogue_store::ReadingContract;
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
    assert_eq!(report.raw_only_chains, Some(0));
    assert_eq!(report.kept_chains, Some(0));
    for name in ["pi", "tool-context"] {
        assert!(matches!(
            contracts.iter().find(|contract| contract.origin() == name),
            Some(ReadingContract::RawOnly { .. })
        ));
    }
    for name in ["llm", "tool-database"] {
        // The shared modules and the row-dump format, in declared order.
        let expected = "parse=3,lines=1,records=1,fields=3,document=1,time=2,command=2,crate::row_dump::VERSION=3";
        assert!(
            matches!(contracts.iter().find(|contract| contract.origin()==name), Some(ReadingContract::Expected { version, .. }) if version == expected)
        );
    }
    assert!(matches!(
        contracts
            .iter()
            .find(|contract| contract.origin() == "export-file"),
        Some(ReadingContract::Kept { .. })
    ));
}

fn keyed_fragment(uuid: &str, key: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({"type":"assistant","uuid":uuid,"sessionId":"keyed","message":{"id":key,"role":"assistant","content":[{"type":"text","text":text}]}})
    )
}

#[test]
fn a_response_growing_across_appends_is_joined_from_the_chains_rows() {
    let home = ClaudeHome::new();
    let mut length = 0;
    for (uuid, text) in [("first", "one"), ("second", "two"), ("third", "three")] {
        let suffix = keyed_fragment(uuid, "response-a", text);
        home.append("keyed.jsonl", &suffix);
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        if length > 0 {
            assert_eq!(pass.bytes_read, length + suffix.len() as u64, "{pass:?}");
            assert_eq!(pass.bytes_stored, suffix.len() as u64, "{pass:?}");
        }
        length += suffix.len() as u64;
    }
    let (engine, _lock) = crate::import::tests::engine(&home, 4096);
    let reader = engine.reads.reader().unwrap();
    let messages = reader
        .messages(
            &polylogue_model::SessionId::parse("claude-code:keyed").unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "one\n\ntwo\n\nthree");
    drop(reader);
    let index = rusqlite::Connection::open(engine.archive.index_db()).unwrap();
    let column = |sql: &str| -> Vec<(String, String)> {
        index
            .prepare(sql)
            .unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    // The chain's rows keep the response key as written and an alias row
    // for each later record (docs/sessions.md, Names and references).
    assert_eq!(
        column(
            "SELECT s.response_key,s.message_id FROM message_parse s WHERE s.response_key IS NOT NULL"
        ),
        [("response-a".to_owned(), "claude-code:first".to_owned())]
    );
    assert_eq!(
        column("SELECT name,target FROM message_alias ORDER BY name"),
        [
            (
                "claude-code:second".to_owned(),
                "claude-code:first".to_owned()
            ),
            (
                "claude-code:third".to_owned(),
                "claude-code:first".to_owned()
            ),
        ]
    );
    // The earlier readings of the growing response leave.
    assert_eq!(
        index
            .query_row(
                "SELECT count(*) FROM text t
                 WHERE NOT EXISTS(SELECT 1 FROM parse_text r WHERE r.text_id=t.id)",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
        0
    );
}

#[test]
fn an_append_continues_an_earlier_response_and_resolves_its_alias() {
    let home = ClaudeHome::new();
    home.append(
        "keyed.jsonl",
        &keyed_fragment("first", "response-a", "first"),
    );
    for key in ["b", "c", "d", "e", "f"] {
        home.append("keyed.jsonl", &keyed_fragment(key, key, key));
    }
    let prefix = std::fs::metadata(home.file("keyed.jsonl")).unwrap().len();
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    }
    let suffix = keyed_fragment("late", "response-a", "last");
    home.append("keyed.jsonl", &suffix);
    // The append joins a response stated before it from the chain's rows.
    let chunk = 128;
    assert!(suffix.len() as u64 > chunk && suffix.len() as u64 <= 2 * chunk);
    let (mut engine, _lock) = crate::import::tests::engine(&home, chunk);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(
        pass.bytes_read,
        prefix + chunk + suffix.len() as u64,
        "one prefix comparison and two reads of the split record"
    );
    assert_eq!(pass.bytes_stored, suffix.len() as u64, "{pass:?}");
    let reader = engine.reads.reader().unwrap();
    let messages = reader
        .messages(
            &polylogue_model::SessionId::parse("claude-code:keyed").unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(messages.len(), 6);
    assert_eq!(
        messages
            .iter()
            .find(|m| m.id.native().as_str() == "first")
            .unwrap()
            .text(),
        "first\n\nlast"
    );
    let message_reference = polylogue_model::Reference::Message(
        polylogue_model::MessageId::parse("claude-code:late").unwrap(),
    );
    let polylogue_model::Reference::Message(id) = &message_reference else {
        unreachable!()
    };
    let resolved = reader.message_with_block_selection(id).unwrap().unwrap().0;
    assert_eq!(resolved.id.native().as_str(), "first");
    assert_eq!(message_reference.to_string(), "message:claude-code:late");
    let options = polylogue_store::query::Options {
        clock: query_language::time::Clock::new(
            "2026-11-08T12:00:00Z".parse().unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                jiff::civil::Time::midnight(),
            )
            .unwrap(),
        ),
        origins: vec!["claude-code".into()],
        page_size: None,
        random_seed: Some(1),
        embeddings: false,
        self_session: None,
        here: None,
        seen_marks_of: None,
        saved: Default::default(),
        kept: Default::default(),
        fields: vec!["id".into()],
        provider_catalog: None,
    };
    // Select the message grain so the row's id names the resolved message.
    let program =
        polylogue_store::query::compile(&format!("messages where {message_reference}"), &options)
            .unwrap();
    let rows = reader.query_rows(&program).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["id"], "claude-code:first");
    drop(reader);
    engine
        .writer
        .archive()
        .user(polylogue_model::user::UserRequest {
            id: "0199d8a0-0000-7000-8000-000000000991".into(),
            machine: "synthetic-machine".into(),
            at: polylogue_model::Timestamp::from_unix_ms(1000).unwrap(),
            authorship: polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            },
            write: polylogue_model::user::UserWrite::Assert {
                annotations: vec![polylogue_model::annotation::NewAnnotation {
                    kind: polylogue_model::annotation::AnnotationKind::Note,
                    key: None,
                    value: Some(serde_json::json!("native-record note")),
                    valid_until: None,
                    links: vec![polylogue_model::annotation::NewAnnotationLink {
                        role: polylogue_model::annotation::LinkRole::About,
                        reference: message_reference.clone(),
                    }],
                }],
            },
        })
        .unwrap();
    let mut before = Vec::new();
    engine
        .reads
        .reader()
        .unwrap()
        .visit_authored(&mut |request| {
            before.push(request.clone());
            Ok(())
        })
        .unwrap();
    polylogue_store::fixture::older_reading(&engine.archive, "claude-code").unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let reader = engine.reads.reader().unwrap();
    let mut after = Vec::new();
    reader
        .visit_authored(&mut |request| {
            after.push(request.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        before, after,
        "reparse changed an authored request or its message reference"
    );
    let rows = reader.query_rows(&program).unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["id"], "claude-code:first");
    assert_eq!(
        reader
            .message_with_block_selection(id)
            .unwrap()
            .unwrap()
            .0
            .id
            .native()
            .as_str(),
        "first"
    );
}

#[test]
fn lost_parser_state_reads_again_keeping_message_sessions() {
    let home = ClaudeHome::new();
    home.append(
        "keyed.jsonl",
        &keyed_fragment("first", "response-a", "first"),
    );
    for key in ["b", "c", "d", "e", "f"] {
        home.append("keyed.jsonl", &keyed_fragment(key, key, key));
    }
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 512);
        assert!(engine.run_pass().unwrap().errors.is_empty());
        let index = rusqlite::Connection::open(engine.archive.index_db()).unwrap();
        index
            .execute_batch("UPDATE parse_state SET resume=NULL")
            .unwrap();
    }
    home.append("keyed.jsonl", &keyed_fragment("late", "response-a", "last"));
    let (mut engine, _lock) = crate::import::tests::engine(&home, 512);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let reader = engine.reads.reader().unwrap();
    let messages = reader
        .messages(
            &polylogue_model::SessionId::parse("claude-code:keyed").unwrap(),
            false,
        )
        .unwrap();
    assert_eq!(messages.len(), 6);
    assert_eq!(
        messages
            .iter()
            .find(|m| m.id.native().as_str() == "first")
            .unwrap()
            .text(),
        "first\n\nlast"
    );
}

/// docs/archive.md, Readings: equal content-named messages are numbered in
/// chain order, and an append numbers its messages after the committed ones.
#[test]
fn an_append_numbers_equal_content_named_messages_after_the_committed_ones() {
    let home = ClaudeHome::new();
    let unnamed = |text: &str| {
        format!(
            "{}\n",
            serde_json::json!({"type":"user","sessionId":"unnamed","timestamp":"2026-03-01T10:00:00Z",
                "message":{"role":"user","content":text}})
        )
    };
    home.append("unnamed.jsonl", &unnamed("same words"));
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    }
    home.append(
        "unnamed.jsonl",
        &(unnamed("other words") + &unnamed("same words")),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let reader = engine.reads.reader().unwrap();
    let ids: Vec<String> = reader
        .messages(
            &polylogue_model::SessionId::parse("claude-code:unnamed").unwrap(),
            false,
        )
        .unwrap()
        .into_iter()
        .map(|m| m.id.native().as_str().to_owned())
        .collect();
    assert_eq!(ids.len(), 3);
    let (first, number) = ids[0].rsplit_once('-').unwrap();
    assert_eq!(number, "0");
    assert_eq!(ids[2], format!("{first}-1"));
    assert!(ids[1].ends_with("-0") && !ids[1].starts_with(first));
}

/// A file history snapshot appended after a later record of a resumed
/// session's response follows the latest message that session opened before
/// it, found from the chain's committed rows: the snapshot names no session
/// and is stated for the file's own.
#[test]
fn an_appended_sessionless_event_follows_the_resumed_sessions_response() {
    let home = ClaudeHome::new();
    home.append("own.jsonl", &claude_user("u1", "own", "start"));
    let resumed = |uuid: &str, key: &str, text: &str| {
        format!(
            "{}\n",
            serde_json::json!({"type":"assistant","uuid":uuid,"sessionId":"resumed","message":{"id":key,"role":"assistant","content":[{"type":"text","text":text}]}})
        )
    };
    home.append("own.jsonl", &resumed("r1", "response-r", "first"));
    home.append("own.jsonl", &resumed("q1", "response-q", "other"));
    home.append("own.jsonl", &resumed("r2", "response-r", "second"));
    {
        let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
    }
    home.append(
        "own.jsonl",
        &format!(
            "{}\n",
            serde_json::json!({"type":"file-history-snapshot","messageId":"r2","snapshot":{}})
        ),
    );
    let (mut engine, _lock) = crate::import::tests::engine(&home, 4096);
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    let reader = engine.reads.reader().unwrap();
    let placed = reader
        .query(
            "SELECT m.id FROM event e JOIN message m ON m.id=e.after_message_id WHERE e.kind='file_history'",
            &[],
        )
        .unwrap();
    assert_eq!(placed.len(), 1, "{placed:?}");
    assert_eq!(placed[0]["id"], "claude-code:q1");
}
