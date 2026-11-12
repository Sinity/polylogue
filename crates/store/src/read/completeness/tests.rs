use super::*;
use crate::fixture::{
    Fixture,
    writer::{WriterArchive, prose, step},
};

/// An index connection on which parse-state edits keep the chain counts,
/// as they are on the writer's connection.
fn index_editor(archive: &Archive) -> Connection {
    let index = Connection::open(archive.index_db()).unwrap();
    index
        .execute(
            "ATTACH DATABASE ?1 AS a",
            [archive.archive_db().to_string_lossy().as_ref()],
        )
        .unwrap();
    crate::source::views(&index).unwrap();
    crate::chain_progress::install_triggers(&index, false).unwrap();
    index
}

fn expected(origin: &str, version: &str) -> ReadingContract {
    ReadingContract::Expected {
        origin: origin.into(),
        version: version.into(),
    }
}

#[test]
fn real_derived_writer_uses_checkpoint_evidence_across_independent_counter_domains() {
    let mut stored = WriterArchive::new();
    stored.write("current", prose("current", 1, 1, "retained source text"));
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
    // proves its own snapshot; a fresh reader must observe the actual lag.
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
    let proof = fresh.completeness(&contracts).unwrap();
    assert!(!proof.caught_up());
    assert!(
        matches!(proof.parsing, CompletenessDimension::Incomplete(ref reasons)
        if reasons.contains(&CompletenessReason::ParserBehind))
    );
}

#[test]
fn exact_waiting_length_and_explicit_kept_contracts_have_distinct_meanings() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    let archive = Connection::open(fixture.archive.archive_db()).unwrap();
    archive
        .execute_batch(
            "INSERT INTO input VALUES('waiting','synthetic');
        INSERT INTO revision VALUES('waiting',1,'first',1,10,1);",
        )
        .unwrap();
    let index = index_editor(&fixture.archive);
    index
        .execute_batch(
            "UPDATE meta SET value='0' WHERE key='building';
        INSERT INTO input_ref(id,format) VALUES('waiting','synthetic');
        INSERT INTO parse_state(input_pk,chain,reading_version,status,parsed_upto,at_length)
        SELECT pk,1,'v1','waiting',4,10 FROM input_ref WHERE id='waiting';",
        )
        .unwrap();
    let pool = fixture.reads().unwrap();
    let exact = pool
        .reader()
        .unwrap()
        .completeness(&[expected("synthetic", "v1")])
        .unwrap();
    assert!(exact.caught_up());
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
    index
        .execute(
            "UPDATE parse_state SET at_length=9 WHERE reading_version='v1'",
            [],
        )
        .unwrap();
    assert!(
        !pool
            .reader()
            .unwrap()
            .completeness(&[expected("synthetic", "v1")])
            .unwrap()
            .caught_up()
    );
    index
        .execute(
            "UPDATE parse_state SET status='no_parser_yet',parsed_upto=10,at_length=NULL",
            [],
        )
        .unwrap();
    let raw = pool
        .reader()
        .unwrap()
        .completeness(&[expected("synthetic", "v1")])
        .unwrap();
    assert_eq!(raw.raw_only_chains, 1);
    assert!(!raw.caught_up());
}

#[test]
fn primary_projection_is_complete_only_when_caught_up_to_the_sinex_head() {
    let root = tempfile::tempdir().unwrap();
    let (archive, _lock) = Archive::open_primary_cache(root.path()).unwrap();
    let projection = |applied: Option<i64>, head: Option<i64>| {
        let conn = Connection::open(archive.index_db()).unwrap();
        crate::primary::projection::save_replay(
            &conn,
            &crate::primary::ReplayState {
                applied_position: applied,
                scope: Some("scope".into()),
                head,
                staged_delivery: None,
                basis: None,
                projection: None,
            },
        )
        .unwrap();
        let pool = ReadPool::open(&archive).unwrap();
        let completeness = pool.reader().unwrap().completeness(&[]).unwrap();
        (completeness.projection.clone(), completeness.degraded())
    };
    assert_eq!(
        projection(None, Some(5)),
        (
            CompletenessDimension::Incomplete(vec![CompletenessReason::StreamNotApplied]),
            true
        )
    );
    assert_eq!(
        projection(Some(3), Some(5)),
        (
            CompletenessDimension::Incomplete(vec![CompletenessReason::StreamBehind]),
            true
        )
    );
    assert_eq!(
        projection(Some(3), None),
        (
            CompletenessDimension::Unknown(vec![CompletenessReason::StreamBehind]),
            true
        )
    );
    assert_eq!(
        projection(Some(5), Some(5)),
        (CompletenessDimension::Complete, false)
    );
}

#[test]
fn explicit_raw_only_declaration_requires_no_parser_version_and_never_claims_parsed_coverage() {
    let root = tempfile::tempdir().unwrap();
    let fixture = Fixture::new(root.path()).unwrap();
    let contracts = [ReadingContract::RawOnly {
        origin: "unparsed-provider".into(),
    }];
    let pool = fixture.reads().unwrap();
    let empty = pool.reader().unwrap().completeness(&contracts).unwrap();
    assert_eq!(empty.parsing, CompletenessDimension::Complete);
    assert_eq!(empty.raw_only_chains, 0);
    let source = Connection::open(fixture.archive.archive_db()).unwrap();
    crate::chain_progress::record_source_changes(&source).unwrap();
    source.execute_batch("INSERT INTO input VALUES('raw-input','unparsed-provider'); INSERT INTO revision VALUES('raw-input',1,'first',1,12,1);").unwrap();
    let changes = crate::chain_progress::recorded(&source).unwrap();
    crate::chain_progress::refresh(&index_editor(&fixture.archive), &changes).unwrap();
    let present = pool.reader().unwrap().completeness(&contracts).unwrap();
    assert_eq!(
        present.parsing,
        CompletenessDimension::Incomplete(vec![CompletenessReason::KnownRawOnly])
    );
    assert_eq!(present.raw_only_chains, 1);
    assert_eq!(present.kept_chains, 0);
    assert!(!present.caught_up());
    assert!(
        pool.reader()
            .unwrap()
            .completeness(&[expected("unparsed-provider", "")])
            .is_err()
    );
}

/// The parsing answer of a walk over every chain of the source joined with
/// its parse state, as completeness was computed before the writer kept
/// chain counts: the independent oracle for those counts.
fn walked(reader: &Reader<'_>, contracts: &[ReadingContract]) -> (CompletenessDimension, u64, u64) {
    use polylogue_model::batch::ParseProgress;
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
                                 o.rowid DESC) AS choice
                FROM source_observation o WHERE o.class = 'imported' AND json_valid(o.details_json)
                    AND json_type(o.details_json, '$.import_format') = 'text'
            )
            SELECT COALESCE(f.format, i.format), c.length, COALESCE(p.parsed_upto, 0),
                p.input_pk IS NOT NULL, p.reading_version, p.status, p.at_length, p.reason
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
        let parsed_upto: i64 = row.get(2).unwrap();
        let parsed: bool = row.get(3).unwrap();
        let version: Option<String> = row.get(4).unwrap();
        let progress = row.get::<_, Option<String>>(5).unwrap().map(|status| {
            let at: Option<i64> = row.get(6).unwrap();
            ParseProgress::from_stored(&status, at.map(|at| at as u64), row.get(7).unwrap())
                .unwrap()
        });
        match declared.get(origin.as_str()) {
            None => {
                unknown = true;
                missing.insert(CompletenessReason::MissingDeclaration);
            }
            Some(ReadingContract::Kept { .. }) => kept += 1,
            Some(ReadingContract::RawOnly { .. }) => {
                raw_only += 1;
                missing.insert(CompletenessReason::KnownRawOnly);
            }
            Some(ReadingContract::Expected {
                version: wanted, ..
            }) => {
                if !parsed {
                    missing.insert(CompletenessReason::MissingCheckpoint);
                } else if version.as_ref() != Some(wanted) {
                    missing.insert(CompletenessReason::ReadingVersionChanged);
                } else {
                    match progress {
                        Some(ParseProgress::NoParserYet) => {
                            raw_only += 1;
                            missing.insert(CompletenessReason::KnownRawOnly);
                        }
                        Some(ParseProgress::Partial) => {
                            missing.insert(CompletenessReason::MissingSourcePrefix);
                        }
                        Some(ParseProgress::Waiting { length: at }) if at == length as u64 => {}
                        Some(ParseProgress::Done) if parsed_upto >= length => {}
                        _ => {
                            missing.insert(CompletenessReason::ParserBehind);
                        }
                    }
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
        required_disposition: None,
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
        acquisition: None,
        basis: None,
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

    // The appended chain re-parsed at a new reading version, then the import parsed.
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
    reparse.parse[0].reading_version = "synthetic=2".into();
    stored.writer.send(reparse).unwrap();
    assert!(stored.writer.flush().unwrap().errors.is_empty());
    agree("re-parse at a new reading version");
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

    // A new writer on the same archive starts from the counts the last one
    // kept; after one that stopped without closing, it counts every chain.
    stored.restart();
    agree("writer restarted");
    stored.writer.stop();
    Connection::open(stored.archive.index_db())
        .unwrap()
        .execute_batch(
            "UPDATE meta SET value = 'open' WHERE key = 'chain_progress';
             DELETE FROM chain_progress WHERE input = 'input-a';",
        )
        .unwrap();
    stored.restart();
    agree("writer restarted after an unclosed stop");
    // The stages differ, so agreement is not that of answers that never change.
    answers.sort_by_key(|answer| format!("{answer:?}"));
    answers.dedup();
    assert!(answers.len() > contract_sets.len() * 2, "{answers:?}");
}

#[test]
fn caught_up_read_steps_independent_of_chain_count() {
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
fn unreadable_chain_counts_are_stored_value_errors() {
    for (state, n) in [("lost", 1), ("current", -1)] {
        let mut stored = WriterArchive::new();
        stored.write("s", prose("s", 1, 1, "text"));
        let index = Connection::open(stored.archive.index_db()).unwrap();
        index
            .execute(
                "INSERT INTO chain_progress_count(origin, reading_version, state, n)
                 VALUES ('claude-code', 'other=1', ?1, ?2)",
                rusqlite::params![state, n],
            )
            .unwrap();
        let pool = ReadPool::open(&stored.archive).unwrap();
        let error = pool
            .reader()
            .unwrap()
            .completeness(&[expected("claude-code", "synthetic=1")])
            .unwrap_err();
        assert!(
            matches!(error, crate::StoreError::StoredValue { .. }),
            "{state} {n}: {error:?}"
        );
        assert_eq!(error.code(), "store", "{state} {n}");
    }
}
