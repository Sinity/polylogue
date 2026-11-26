use crate::{Archive, Change, ReadPool, Writer};
use polylogue_model::chain::{ChainState, Outcomes, ParserState};
use polylogue_model::{ChainFailure, ChainFailureKind};

/// What a test holds open: the archive's directory and lock, its writer,
/// the chain `bytes` were stored as, and a step over them not yet sent.
type Stored = (
    tempfile::TempDir,
    crate::ArchiveLock,
    Archive,
    Writer,
    i64,
    crate::Batch,
);

fn stored(bytes: &[u8]) -> Stored {
    let dir = tempfile::tempdir().unwrap();
    let (archive, lock) = Archive::open_or_create(&dir.path().join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let length = bytes.len() as u64;
    let mut batch = crate::fixture::writer::step(
        "state-input",
        "claude-code",
        "/synthetic/state.jsonl",
        Default::default(),
        0,
        length,
    );
    let mut raw = batch.clone();
    raw.parse.clear();
    let chain = writer
        .store_bytes(archive.prepare_bytes(bytes).unwrap(), raw)
        .unwrap()
        .chain;
    batch.change = Change::Reparse { chain };
    batch.observation = None;
    (dir, lock, archive, writer, chain, batch)
}

fn state(archive: &Archive, chain: i64) -> ChainState {
    ReadPool::open(archive)
        .unwrap()
        .reader()
        .unwrap()
        .chain_state("state-input", chain)
        .unwrap()
}

#[test]
fn each_state_the_writer_stores_reads_back_as_the_same_value() {
    let (_dir, _lock, archive, writer, chain, mut batch) = stored(b"one\ntwo\nthr");
    assert_eq!(state(&archive, chain), ChainState::Unparsed);
    // Two records parsed, and a third still being written past them.
    batch.parse[0].parsed_upto = 8;
    batch.parse[0].counts = polylogue_model::RecordCounts {
        records: 2,
        used: [("line".to_owned(), 2)].into(),
        ..Default::default()
    };
    batch.parse[0].resume = Some(ParserState(Some(serde_json::json!({"open": "response"}))));
    writer.send(batch.clone()).unwrap();
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    let stored = ReadPool::open(&archive)
        .unwrap()
        .reader()
        .unwrap()
        .latest_chain("state-input")
        .unwrap()
        .unwrap()
        .stored;
    assert_eq!(
        state(&archive, chain),
        ChainState::Parsed {
            version: "synthetic=1".into(),
            parsed_to: 8,
            seen_to: stored,
            resume: Some(ParserState(Some(serde_json::json!({"open": "response"})))),
            counts: Outcomes {
                used: 2,
                skipped: 0,
                unreadable: 0,
            },
        }
    );
    // The parser fails at the third record: the two before it stay.
    let mut failed = crate::ParsePart::failed(
        8..8,
        ChainFailure {
            kind: ChainFailureKind::Malformed,
            detail: "line 3 is not JSON".into(),
        },
    );
    failed.parse_version = "synthetic=1".into();
    batch.parse = vec![failed];
    writer.send(batch).unwrap();
    let report = writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(
        state(&archive, chain),
        ChainState::Failed {
            version: "synthetic=1".into(),
            parsed_to: 8,
            seen_to: stored,
            failure: ChainFailure {
                kind: ChainFailureKind::Malformed,
                detail: "line 3 is not JSON".into(),
            },
            counts: Outcomes {
                used: 2,
                skipped: 0,
                unreadable: 0,
            },
        }
    );
}

#[test]
fn a_part_that_does_not_start_where_the_parse_ends_is_refused_and_leaves_the_chain_due() {
    let (_dir, _lock, archive, writer, chain, mut batch) = stored(b"one\ntwo\n");
    batch.parse[0].parsed_upto = 4;
    batch.parse[0].at_end = false;
    batch.parse[0].counts = polylogue_model::RecordCounts {
        records: 1,
        used: [("line".to_owned(), 1)].into(),
        ..Default::default()
    };
    batch.parse[0].resume = Some(ParserState(None));
    writer.send(batch.clone()).unwrap();
    assert!(writer.flush().unwrap().errors.is_empty());
    let before = state(&archive, chain);
    assert_eq!(before.parsed_to(), 4);
    // Planned from an earlier state: it starts at byte 2, not 4.
    let mut stale = batch.clone();
    stale.parse[0].start = 2;
    stale.parse[0].reset = false;
    stale.parse[0].parsed_upto = 8;
    stale.parse[0].at_end = true;
    writer.send(stale).unwrap();
    let report = writer.flush().unwrap();
    let [failure] = &report.errors[..] else {
        panic!("one refusal: {report:?}");
    };
    assert_eq!(failure.cause, polylogue_model::Cause::Transient);
    assert!(!failure.cause.recurs());
    assert_eq!(state(&archive, chain), before);
    // The chain stays due from its stored state.
    let facts = polylogue_model::chain::ChainFacts {
        stored: 8,
        repeated_by: None,
        own_from: 0,
        parser: Some("synthetic=1".into()),
        finished: false,
    };
    assert_eq!(
        polylogue_model::chain::due(&before, &facts),
        polylogue_model::chain::Due::Extend { from: 4 }
    );
    assert_eq!(
        failure.message,
        format!(
            "the part of state-input chain {chain} starts at byte 2 under synthetic=1, where its stored parse ends at 4 under synthetic=1"
        )
    );
}
