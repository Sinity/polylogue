use super::*;
use crate::fixture::{
    Fixture,
    writer::{WriterArchive, prose, step},
};

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
    let index = Connection::open(fixture.archive.index_db()).unwrap();
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
    source.execute_batch("INSERT INTO input VALUES('raw-input','unparsed-provider'); INSERT INTO revision VALUES('raw-input',1,'first',1,12,1);").unwrap();
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
