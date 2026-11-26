use super::*;
use crate::fixture::{
    Fixture,
    writer::{WriterArchive, prose, step},
};

/// Runs `sql` on `database` in one transaction that advances its write
/// counter, as every writer transaction does: a read position names what
/// it reads.
fn edit(database: &std::path::Path, sql: &str) {
    let mut conn = Connection::open(database).unwrap();
    let tx = conn.transaction().unwrap();
    tx.execute_batch(sql).unwrap();
    crate::archive::bump_counter(&tx).unwrap();
    tx.commit().unwrap();
}

fn expected(origin: &str, version: &str) -> ReadingContract {
    ReadingContract::Expected {
        origin: origin.into(),
        version: version.into(),
    }
}

#[test]
fn derived_writes_count_checkpoints_across_counter_domains() {
    let mut stored = WriterArchive::new();
    stored.write("current", prose("current", 1, 1, "stored source text"));
    let contracts = [expected("claude-code", "synthetic=1")];
    let pool = ReadPool::open(&stored.archive).unwrap();
    let pinned = pool.reader().unwrap();
    let before = pinned.change_position().unwrap();
    assert!(
        before.index > before.archive,
        "derivation advances only the index domain"
    );
    assert!(pinned.completeness(&contracts).unwrap().caught_up());

    // Real new source bytes have no new parsed checkpoint. The old reader still
    // sees its own snapshot as complete; a fresh reader must see the lag.
    let mut raw = step(
        "input-current",
        "claude-code",
        "/p/current",
        Default::default(),
        stored.offset(),
        1,
    );
    raw.parse.clear();
    let body = stored.archive.prepare_bytes(&b"r\n"[..]).unwrap();
    stored.writer.store_bytes(body, raw).unwrap();
    assert!(pinned.completeness(&contracts).unwrap().caught_up());
    drop(pinned);
    let fresh = pool.reader().unwrap();
    let completeness = fresh.completeness(&contracts).unwrap();
    assert!(!completeness.caught_up());
    assert!(
        matches!(completeness.parsing, CompletenessDimension::Incomplete(ref reasons)
        if reasons.contains(&CompletenessReason::ParserBehind))
    );
}

#[test]
fn waiting_length_and_explicit_kept_contracts_have_distinct_meanings() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    edit(
        &fixture.archive.archive_db(),
        "INSERT INTO input VALUES('waiting','synthetic');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES('waiting',1,'first',1,10,1);",
    );
    edit(
        &fixture.archive.index_db(),
        "INSERT INTO input_ref(id,format) VALUES('waiting','synthetic');
        INSERT INTO parse_state(input_pk,chain,parser_version,state,parsed_to,waiting)
        SELECT pk,1,'v1','parsed',4,6 FROM input_ref WHERE id='waiting';",
    );
    let pool = fixture.reads().unwrap();
    let matching = pool
        .reader()
        .unwrap()
        .completeness(&[expected("synthetic", "v1")])
        .unwrap();
    assert!(matching.caught_up());
    let unknown = pool.reader().unwrap().completeness(&[]).unwrap();
    assert!(matches!(unknown.parsing, CompletenessDimension::Unknown(_)));
    let kept = pool
        .reader()
        .unwrap()
        .completeness(&[ReadingContract::Kept {
            origin: "synthetic".into(),
        }])
        .unwrap();
    assert!(kept.caught_up());
    assert_eq!(kept.kept_chains, 1);
    assert_eq!(kept.raw_only_chains, 0);
    // The parser saw 9 bytes of the 10 stored.
    edit(
        &fixture.archive.index_db(),
        "UPDATE parse_state SET waiting=5 WHERE parser_version='v1'",
    );
    let behind = pool
        .reader()
        .unwrap()
        .completeness(&[expected("synthetic", "v1")])
        .unwrap();
    assert_eq!(
        behind.parsing,
        CompletenessDimension::Incomplete(vec![CompletenessReason::ParserBehind])
    );
    assert!(!behind.caught_up());
    // A failure at the stored length waits for new bytes; it is shown, and
    // the index is not complete while it stands.
    edit(
        &fixture.archive.index_db(),
        "UPDATE parse_state SET state='failed',waiting=6,failure_kind='refused',failure_detail='not a synthetic file'",
    );
    let refused = pool
        .reader()
        .unwrap()
        .completeness(&[expected("synthetic", "v1")])
        .unwrap();
    assert_eq!(
        refused.parsing,
        CompletenessDimension::Incomplete(vec![CompletenessReason::Refused])
    );
    edit(
        &fixture.archive.index_db(),
        "UPDATE parse_state SET failure_kind='malformed'",
    );
    let failed = pool
        .reader()
        .unwrap()
        .completeness(&[expected("synthetic", "v1")])
        .unwrap();
    assert_eq!(
        failed.parsing,
        CompletenessDimension::Incomplete(vec![CompletenessReason::ParseFailed])
    );
}

#[test]
fn bytes_only_declaration_needs_no_parser_or_parsed_coverage() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    let contracts = [ReadingContract::RawOnly {
        origin: "unparsed-provider".into(),
    }];
    let pool = fixture.reads().unwrap();
    let empty = pool.reader().unwrap().completeness(&contracts).unwrap();
    assert_eq!(empty.parsing, CompletenessDimension::Complete);
    assert_eq!(empty.raw_only_chains, 0);
    edit(
        &fixture.archive.archive_db(),
        "INSERT INTO input VALUES('raw-input','unparsed-provider'); INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES('raw-input',1,'first',1,12,1);",
    );
    // Bytes kept with no parser are counted and leave the index complete.
    let present = pool.reader().unwrap().completeness(&contracts).unwrap();
    assert_eq!(present.parsing, CompletenessDimension::Complete);
    assert_eq!(present.raw_only_chains, 1);
    assert_eq!(present.kept_chains, 0);
    assert!(present.caught_up());
    assert!(matches!(
        pool.reader()
            .unwrap()
            .completeness(&[expected("unparsed-provider", "")]),
        Err(crate::StoreError::Internal(_))
    ));
}

/// The parsing answer of a walk over every chain of the source joined with
/// its parse state, written from the rules of `docs/archive.md` (parse
/// versions) rather than through `model::chain`: the independent oracle.
fn walked(reader: &Reader<'_>, contracts: &[ReadingContract]) -> (CompletenessDimension, u64, u64) {
    let declared: BTreeMap<&str, &ReadingContract> =
        contracts.iter().map(|c| (c.origin(), c)).collect();
    let conn = reader.conn().unwrap();
    let mut statement = conn
        .prepare(
            "WITH chains AS (
                SELECT input, chain, max(length) AS length FROM source_revision GROUP BY input, chain
            ), formats AS (
                SELECT o.input, json_extract(o.details_json, '$.import_format') AS format,
                    row_number() OVER (PARTITION BY o.input
                        ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                                 o.commit_seq DESC, o.id DESC) AS choice
                FROM source_observation o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT COALESCE(f.format, i.origin), c.length, p.state, p.parser_version,
                p.parsed_to + p.waiting, p.failure_kind
            FROM chains c JOIN source_input i ON i.id = c.input
            LEFT JOIN formats f ON f.input = c.input AND f.choice = 1
            LEFT JOIN input_ref ir ON ir.id = c.input
            LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain",
        )
        .unwrap();
    let mut rows = statement.query([]).unwrap();
    let (mut missing, mut unknown, mut kept, mut raw_only) = (BTreeSet::new(), false, 0, 0);
    while let Some(row) = rows.next().unwrap() {
        let origin: String = row.get(0).unwrap();
        let length: i64 = row.get(1).unwrap();
        let state: Option<String> = row.get(2).unwrap();
        let version: Option<String> = row.get(3).unwrap();
        let seen_to: Option<i64> = row.get(4).unwrap();
        let failure: Option<String> = row.get(5).unwrap();
        match declared.get(origin.as_str()) {
            None => {
                unknown = true;
                missing.insert(CompletenessReason::MissingDeclaration);
            }
            Some(ReadingContract::Kept { .. }) => kept += 1,
            Some(ReadingContract::RawOnly { .. }) => raw_only += 1,
            Some(ReadingContract::Expected {
                version: wanted, ..
            }) => {
                if state.is_none() {
                    if length > 0 {
                        missing.insert(CompletenessReason::MissingCheckpoint);
                    }
                } else if version.as_ref() != Some(wanted) {
                    missing.insert(CompletenessReason::ParseVersionChanged);
                } else if seen_to != Some(length) {
                    missing.insert(CompletenessReason::ParserBehind);
                } else if failure.as_deref() == Some("refused") {
                    missing.insert(CompletenessReason::Refused);
                } else if failure.is_some() {
                    missing.insert(CompletenessReason::ParseFailed);
                }
            }
        }
    }
    let parsing = if missing.is_empty() {
        CompletenessDimension::Complete
    } else if unknown {
        CompletenessDimension::Unknown(missing.into_iter().collect())
    } else {
        CompletenessDimension::Incomplete(missing.into_iter().collect())
    };
    (parsing, kept, raw_only)
}

/// Stores bytes of an export member that the import reads as `format`.
fn import(stored: &WriterArchive, body: &[u8], format: &str) -> crate::staging::StoredInput {
    let prepared = stored.archive.prepare_bytes(body).unwrap();
    let path = format!("/synthetic/{format}.json");
    let batch = crate::writer::Batch {
        input: crate::writer::InputKey {
            id: prepared.hash().to_string(),
            format: "export-file".into(),
            route: polylogue_model::Route::Export,
        },
        change: crate::Change::First,
        observation: Some(crate::Observation {
            machine: "synthetic-machine".into(),
            host: "synthetic-host".into(),
            path: path.clone(),
            path_bytes: path.into_bytes(),
            facts: None,
            seen_at_ms: 1,
            container: None,
            database: None,
            import: Some(crate::ImportMember {
                version: 1,
                import_id: format!("import-{format}"),
                member: format!("{format}.json"),
                format: Some(format.into()),
                authorship: None,
            }),
        }),
        continues: false,
        new_blob: None,
        blobs: vec![],
        length: prepared.len(),
        parse: vec![],
        capture_read: None,
        release: Default::default(),
    };
    stored.writer.store_bytes(prepared, batch).unwrap()
}

#[test]
fn caught_up_equals_a_full_walk() {
    let mut stored = WriterArchive::new();
    let pool = ReadPool::open(&stored.archive).unwrap();
    let contract_sets = [
        vec![
            expected("claude-code", "synthetic=1"),
            ReadingContract::RawOnly {
                origin: "unparsed-provider".into(),
            },
            ReadingContract::Kept {
                origin: "chatgpt-export".into(),
            },
        ],
        vec![
            expected("claude-code", "synthetic=2"),
            expected("chatgpt-export", "synthetic=1"),
        ],
        vec![ReadingContract::Kept {
            origin: "claude-code".into(),
        }],
        vec![],
    ];
    let mut answers = Vec::new();
    let mut agree = |stage: &str| {
        let reader = pool.reader().unwrap();
        for contracts in &contract_sets {
            let kept = reader.completeness(contracts).unwrap();
            let answer = (kept.parsing, kept.kept_chains, kept.raw_only_chains);
            assert_eq!(answer, walked(&reader, contracts), "{stage}: {contracts:?}");
            answers.push(answer);
        }
    };
    agree("empty archive");

    // Live appends, each parsed to its stored length.
    stored.write("a", prose("a", 1, 1, "first prompt"));
    stored.write("b", prose("b", 1, 2, "another session"));
    agree("parsed inputs");
    stored.write("a", prose("a", 2, 3, "appended prompt"));
    agree("parsed append");

    // Appended bytes stored without their reading: the chain is behind.
    let mut raw = step(
        "input-a",
        "claude-code",
        "/p/a",
        Default::default(),
        stored.offset(),
        1,
    );
    raw.parse.clear();
    let body = stored.archive.prepare_bytes(&b"r\n"[..]).unwrap();
    let appended = stored.writer.store_bytes(body, raw).unwrap();
    agree("unparsed append");

    // Bytes of an origin that has no parser, and an import read as another format.
    let mut unparsed = step(
        "raw-input",
        "unparsed-provider",
        "/p/raw",
        Default::default(),
        0,
        3,
    );
    unparsed.parse.clear();
    let body = stored.archive.prepare_bytes(&b"rrr\n"[..]).unwrap();
    stored.writer.store_bytes(body, unparsed).unwrap();
    let export = import(&stored, b"synthetic export member", "chatgpt-export");
    agree("raw-only store and import");

    // The appended chain re-parsed at a new parse version, then the import parsed.
    let mut reparse = step(
        "input-a",
        "claude-code",
        "/p/a",
        Default::default(),
        0,
        appended.length - 1,
    );
    reparse.change = crate::Change::Reparse {
        chain: appended.chain,
    };
    reparse.observation = None;
    reparse.parse[0].parse_version = "synthetic=2".into();
    stored.writer.send(reparse).unwrap();
    assert!(stored.writer.flush().unwrap().errors.is_empty());
    agree("re-parse at a new parse version");
    let mut parsed = step(
        &export.input,
        "chatgpt-export",
        "/synthetic/chatgpt-export.json",
        Default::default(),
        0,
        export.length - 1,
    );
    parsed.input.route = polylogue_model::Route::Export;
    parsed.change = crate::Change::Reparse {
        chain: export.chain,
    };
    parsed.observation = None;
    stored.writer.send(parsed).unwrap();
    assert!(stored.writer.flush().unwrap().errors.is_empty());
    agree("import parsed");

    // A new writer on the same archive reads the same states.
    stored.restart();
    agree("writer restarted");
    // The stages differ, so agreement is not that of answers that never change.
    answers.sort_by_key(|answer| format!("{answer:?}"));
    answers.dedup();
    assert!(answers.len() > contract_sets.len() * 2, "{answers:?}");
}

#[test]
fn a_second_read_at_one_position_does_not_judge_the_chains_again() {
    let steps = |sessions: usize| {
        let mut stored = WriterArchive::new();
        for n in 0..sessions {
            let session = format!("s{n}");
            stored.write(&session, prose(&session, 1, 1, "a prompt"));
        }
        let pool = ReadPool::open(&stored.archive).unwrap();
        let reader = pool.reader().unwrap();
        let contracts = [expected("claude-code", "synthetic=1")];
        assert!(reader.completeness(&contracts).unwrap().caught_up());
        let counted = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = std::sync::Arc::clone(&counted);
        let conn = reader.conn().unwrap();
        conn.progress_handler(
            1,
            Some(move || {
                counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                false
            }),
        )
        .unwrap();
        assert!(reader.completeness(&contracts).unwrap().caught_up());
        conn.progress_handler(0, None::<fn() -> bool>).unwrap();
        counted.load(std::sync::atomic::Ordering::Relaxed)
    };
    assert_eq!(steps(2), steps(40));
}

#[test]
fn an_empty_chain_is_done_without_a_parse_state() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    edit(
        &fixture.archive.archive_db(),
        "INSERT INTO input VALUES('empty','synthetic');
        INSERT INTO revision(input,number,produced,chain,length,commit_seq) VALUES('empty',1,'first',1,0,1);",
    );
    let contracts = [expected("synthetic", "v1")];
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    assert!(reader.completeness(&contracts).unwrap().caught_up());
    assert_eq!(
        reader.status(&contracts).unwrap().parse_states,
        BTreeMap::from([("done".to_owned(), 1)])
    );
}
