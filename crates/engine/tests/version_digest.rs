//! Version digests over synthetic SQLite evidence.
use polylogue_engine::{
    state::codex as adapter,
    version_digest::{Advance, Driver, Error},
};
use polylogue_model::BlobHash;
use polylogue_sources::codex::{
    self,
    keyed::{Cell, Delta, EvidenceHead, EvidenceUnit, Key, Resume},
};
use polylogue_store::{
    Error as StoreError, ImportObservation, InputFormat, Writer,
    parser_state::{Accounting, Basis, Budget, Head, Phase, Progress},
};
use std::collections::BTreeMap;
fn budget() -> Budget {
    Budget {
        cells: 65536,
        encoded_bytes: 16 * 1024 * 1024,
    }
}
struct World {
    _dir: tempfile::TempDir,
    writer: Writer,
    head: Head,
}
impl World {
    fn new(head: EvidenceHead, units: Vec<(Key, EvidenceUnit)>) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(&dir.path().join("archive")).unwrap();
        let blob = writer.put_file_blob(b"").unwrap();
        let input = writer
            .store_immutable_input(
                blob,
                InputFormat::CodexRollout,
                &[],
                &ImportObservation {
                    id: "synthetic-version-proof",
                    machine: "synthetic-machine",
                    host: "synthetic-host",
                    path: "synthetic/input.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let predecessor = writer
            .begin_parser(&Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: codex::PARSER_VERSION,
                digest: [7; 32],
                acquired_upto: 0,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let mut cells = BTreeMap::new();
        cells.insert(Key::EvidenceHead(0), Some(Cell::EvidenceHead(head)));
        for (key, unit) in units {
            cells.insert(key, Some(Cell::Evidence(unit)));
        }
        let delta = Delta {
            resume: Resume::default(),
            cells,
            events: vec![],
            links: vec![],
            reached_back: false,
            pending_versions: vec![],
            conflicts: vec![],
            pending_results: vec![],
        };
        let encoded = adapter::encode_delta(&delta, 1, budget()).unwrap();
        let mut work = writer
            .prepare_parser(
                &predecessor,
                Progress {
                    inspected_upto: 0,
                    parsed_upto: 0,
                    phase: Phase::Finalizing,
                    finalization_pending: true,
                    counts: Accounting::default(),
                },
            )
            .unwrap();
        work = writer
            .stage_parser(&work.token, &encoded.changes, budget())
            .unwrap();
        work = writer.seal_parser(&work.token).unwrap();
        while work.token.applied < work.token.staged {
            work = writer.apply_parser(&work.token, budget()).unwrap();
        }
        let head = writer.publish_parser(&work.token).unwrap();
        Self {
            _dir: dir,
            writer,
            head,
        }
    }
}
fn head(units: u64) -> EvidenceHead {
    EvidenceHead {
        items: 1,
        units,
        revision: 7,
        incomplete: false,
    }
}
fn item(raw: &[u8], whole_record: bool) -> (Key, EvidenceUnit) {
    (
        Key::Evidence(0, 0, 0),
        EvidenceUnit {
            tag: "item".into(),
            canonical: raw.to_vec(),
            whole_record,
            origins: Default::default(),
        },
    )
}
fn digest(world: &World, bytes: u64) -> BlobHash {
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let mut driver = Driver::start(&snapshot, 0, 7, budget()).unwrap();
    for _ in 0..10000 {
        match driver.advance(budget(), bytes).unwrap() {
            Advance::Progress {
                read_cells,
                hashed_bytes,
                ..
            } => {
                assert!(read_cells <= 1);
                assert!(hashed_bytes <= bytes);
            }
            Advance::Complete(proof) => {
                assert_eq!(proof.token(), &world.head.token);
                assert_eq!(proof.message(), 0);
                assert_eq!(proof.head().revision, 7);
                return proof.digest();
            }
        }
    }
    panic!("proof did not finish")
}
#[test]
fn singleton_raw_exception_keeps_numeric_spelling_and_restarts_from_zero() {
    for raw in [br#"{"n":1E2}"#.as_slice(), br#"{"n":1e2}"#] {
        let world = World::new(head(1), vec![item(raw, true)]);
        assert_eq!(digest(&world, 64), BlobHash::of(raw));
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let mut first = Driver::start(&snapshot, 0, 7, budget()).unwrap();
        first.advance(budget(), 1).unwrap();
        assert_eq!(first.hashed_bytes(), 1);
        drop(first);
        let restarted = Driver::start(&snapshot, 0, 7, budget()).unwrap();
        assert_eq!(restarted.hashed_bytes(), 0);
    }
    assert_ne!(BlobHash::of(br#"{"n":1E2}"#), BlobHash::of(br#"{"n":1e2}"#));
}
#[test]
fn compound_digest_matches_unit_preimage_with_tiny_payload_steps() {
    let raw = format!("{{\"text\":\"{}\"}}", "x".repeat(5000));
    let source = (
        Key::Evidence(0, 1, 1),
        EvidenceUnit {
            tag: "context:source".into(),
            canonical: br#""cli""#.to_vec(),
            whole_record: false,
            origins: Default::default(),
        },
    );
    let world = World::new(head(2), vec![item(raw.as_bytes(), true), source]);
    let mut preimage = b"\0polylogue-message-version\0v1\0".to_vec();
    preimage.extend_from_slice(&2u64.to_be_bytes());
    for (tag, payload) in [
        ("item", raw.as_bytes()),
        ("context:source", br#""cli""#.as_slice()),
    ] {
        preimage.push(0);
        preimage.extend_from_slice(&(tag.len() as u64).to_be_bytes());
        preimage.extend_from_slice(tag.as_bytes());
        preimage.extend_from_slice(&(payload.len() as u64).to_be_bytes());
        preimage.extend_from_slice(payload);
    }
    assert_eq!(digest(&world, 64), BlobHash::of(&preimage));
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let mut driver = Driver::start(&snapshot, 0, 7, budget()).unwrap();
    // Load the large unit once. Later one-byte calls hash its buffer directly;
    // a zero read allowance shows nothing is read again.
    driver.advance(budget(), 64).unwrap();
    for _ in 0..100 {
        match driver
            .advance(
                Budget {
                    cells: 0,
                    encoded_bytes: 0,
                },
                1,
            )
            .unwrap()
        {
            Advance::Progress {
                read_cells,
                hashed_bytes,
                ..
            } => {
                assert_eq!(read_cells, 0);
                assert_eq!(hashed_bytes, 1);
            }
            _ => panic!("premature proof"),
        }
    }
}
#[test]
fn incomplete_missing_malformed_or_changed_evidence_yields_no_digest() {
    let mut incomplete = head(1);
    incomplete.incomplete = true;
    let world = World::new(incomplete, vec![item(b"{}", true)]);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    assert!(matches!(
        Driver::start(&snapshot, 0, 7, budget()),
        Err(Error::Evidence(_))
    ));
    let world = World::new(head(1), vec![item(b"{}", true)]);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    assert!(matches!(
        Driver::start(&snapshot, 0, 8, budget()),
        Err(Error::Revision {
            expected: 8,
            actual: 7
        })
    ));
    for (declared, units) in [
        (1, vec![]),
        (2, vec![item(b"{}", true)]),
        (1, vec![item(b"bad-json", true)]),
        (
            1,
            vec![(
                Key::Evidence(0, 0, 0),
                EvidenceUnit {
                    tag: "usage".into(),
                    canonical: b"{}".to_vec(),
                    whole_record: true,
                    origins: Default::default(),
                },
            )],
        ),
    ] {
        let world = World::new(head(declared), units);
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let mut driver = Driver::start(&snapshot, 0, 7, budget()).unwrap();
        let mut failed = false;
        for _ in 0..20 {
            match driver.advance(budget(), 1024) {
                Err(Error::Evidence(_)) => {
                    failed = true;
                    break;
                }
                Ok(Advance::Progress { .. }) => {}
                _ => panic!("invalid evidence yielded proof"),
            }
        }
        assert!(failed);
        assert!(matches!(
            driver.advance(budget(), 1024),
            Err(Error::Finished)
        ));
    }
}
#[test]
fn read_limit_errors_pass_through_and_no_partial_digest_escapes() {
    let world = World::new(head(1), vec![item(b"{}", true)]);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    assert!(matches!(
        Driver::start(
            &snapshot,
            0,
            7,
            Budget {
                cells: 0,
                ..budget()
            }
        ),
        Err(Error::Store(StoreError::ParserAdmission { .. }))
    ));
    let mut driver = Driver::start(&snapshot, 0, 7, budget()).unwrap();
    assert!(matches!(
        driver.advance(
            Budget {
                cells: 0,
                ..budget()
            },
            1024
        ),
        Err(Error::Store(StoreError::ParserAdmission { .. }))
    ));
    assert!(matches!(
        driver.advance(budget(), 1024),
        Err(Error::Finished)
    ));
}

#[test]
fn insufficient_framing_allowance_reports_retry_without_reloading_unit() {
    let world = World::new(head(1), vec![item(b"{}", false)]);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let mut driver = Driver::start(&snapshot, 0, 7, budget()).unwrap();
    let required = polylogue_model::version_evidence::Framer::compound_minimum_bytes();
    assert!(
        matches!(driver.advance(budget(),required-1).unwrap(),Advance::Progress {read_cells:1,hashed_bytes:0,minimum_bytes:Some(n)} if n==required)
    );
    assert!(
        matches!(driver.advance(Budget {cells:0,encoded_bytes:0},required).unwrap(),Advance::Progress {read_cells:0,hashed_bytes:n,minimum_bytes:Some(9)} if n==required)
    );
}

#[test]
fn last_payload_byte_yields_without_spending_next_slots_read_budget() {
    let raw = br#"{"x":1}"#;
    for compound in [false, true] {
        let mut units = vec![item(raw, true)];
        let opening = if compound {
            units.push((
                Key::Evidence(0, 1, 1),
                EvidenceUnit {
                    tag: "context:source".into(),
                    canonical: br#""cli""#.to_vec(),
                    whole_record: false,
                    origins: Default::default(),
                },
            ));
            polylogue_model::version_evidence::Framer::compound_minimum_bytes()
                + 9
                + "item".len() as u64
                + 8
        } else {
            0
        };
        let world = World::new(head(if compound { 2 } else { 1 }), units);
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let mut driver = Driver::start(&snapshot, 0, 7, budget()).unwrap();
        let first_bytes = opening + raw.len() as u64 - 1;
        assert!(
            matches!(driver.advance(budget(),first_bytes).unwrap(),Advance::Progress {read_cells:1,hashed_bytes:n,..} if n==first_bytes)
        );
        // Hashing the last byte must not also read the next optional slot.
        assert!(matches!(
            driver
                .advance(
                    Budget {
                        cells: 0,
                        encoded_bytes: 0
                    },
                    1
                )
                .unwrap(),
            Advance::Progress {
                read_cells: 0,
                hashed_bytes: 1,
                minimum_bytes: None
            }
        ));
        let expected = digest(&world, 64);
        let mut complete = false;
        for _ in 0..20 {
            match driver.advance(budget(), 64).unwrap() {
                Advance::Progress {
                    read_cells,
                    hashed_bytes,
                    ..
                } => {
                    assert!(read_cells <= 1);
                    assert!(hashed_bytes <= 64);
                }
                Advance::Complete(proof) => {
                    assert_eq!(proof.digest(), expected);
                    complete = true;
                    break;
                }
            }
        }
        assert!(complete);
    }
}
