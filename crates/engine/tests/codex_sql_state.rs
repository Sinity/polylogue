//! Real SQLite state tests. These do not certify canonical output publication.
#[path = "../src/state/codex.rs"]
mod adapter;
use polylogue_sources::codex::{
    self,
    keyed::{
        Cell, Delta, Key, KeyedRecords, MemoryReadView, NativeConflict, PendingVersion, ReadView,
        Resume,
    },
};
use polylogue_sources::frame::{Counts, drive_jsonl};
use polylogue_store::parser_state::{Accounting, Basis, Budget, CellChange, Head, Phase, Progress};
use polylogue_store::{Error, ImportObservation, InputFormat, Writer};
use serde_json::{Value, json};
use std::path::PathBuf;

fn budget() -> Budget {
    Budget {
        cells: 65536,
        encoded_bytes: 64 * 1024 * 1024,
    }
}
struct World {
    _dir: tempfile::TempDir,
    root: PathBuf,
    writer: Writer,
    head: Head,
}
impl World {
    fn new(bytes: &[u8]) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut writer = Writer::open(&root).unwrap();
        let blob = writer.put_file_blob(bytes).unwrap();
        let retained = writer
            .retain_immutable_input(
                blob,
                InputFormat::CodexRollout,
                &ImportObservation {
                    id: "synthetic-import",
                    machine: "synthetic-machine",
                    host: "synthetic-host",
                    path: "synthetic/codex.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = writer
            .begin_parser(&Basis {
                input_pk: retained.input_pk,
                chain: retained.chain,
                parser_version: codex::PARSER_VERSION,
                digest: [7; 32],
                acquired_upto: bytes.len() as u64,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        Self {
            _dir: dir,
            root,
            writer,
            head,
        }
    }
    fn reopen(self) -> Self {
        let Self {
            _dir,
            root,
            writer,
            head,
        } = self;
        drop(writer);
        let writer = Writer::open(&root).unwrap();
        let reopened = writer
            .parser_head(head.token.input_pk, head.token.chain)
            .unwrap()
            .unwrap();
        assert_eq!(reopened.token, head.token);
        Self {
            _dir,
            root,
            writer,
            head: reopened,
        }
    }
    fn install(&mut self, changes: &[CellChange], end: u64, counts: Accounting) {
        let mut work = self
            .writer
            .prepare_parser(
                &self.head,
                Progress {
                    inspected_upto: end,
                    parsed_upto: end,
                    phase: Phase::Parsing,
                    finalization_pending: true,
                    counts,
                },
            )
            .unwrap();
        work = self
            .writer
            .stage_parser(&work.token, changes, budget())
            .unwrap();
        work = self.writer.seal_parser(&work.token).unwrap();
        while work.token.applied < work.token.staged {
            work = self
                .writer
                .apply_parser(
                    &work.token,
                    Budget {
                        cells: 2,
                        ..budget()
                    },
                )
                .unwrap();
        }
        self.head = self.writer.publish_parser(&work.token).unwrap();
    }
    fn parse(&mut self, bytes: &[u8]) -> Delta {
        let snapshot = self.writer.parser_snapshot(&self.head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let framed = drive_jsonl(
            KeyedRecords::new(&attempt, attempt.resume().unwrap()),
            bytes,
            self.head.parsed_upto,
        );
        let mut delta = framed.output.unwrap();
        assert!(attempt.take_failure().is_none());
        delta.resume.parsed_upto = framed.parsed_upto;
        let encoded =
            adapter::encode_delta(&delta, (self.head.token.step + 1) as u64, budget()).unwrap();
        assert_eq!(
            encoded.encoded_bytes,
            encoded
                .changes
                .iter()
                .map(|c| c.namespace.len() + c.key.len() + c.value.as_ref().map_or(0, Vec::len))
                .sum::<usize>()
        );
        drop(attempt);
        drop(snapshot);
        self.install(
            &encoded.changes,
            framed.parsed_upto,
            accounting(&framed.counts),
        );
        delta
    }
}
fn accounting(c: &Counts) -> Accounting {
    Accounting {
        framed: c.framed,
        used: c.used,
        ignored: c.ignored.values().sum(),
        unrecognized: c.unrecognized.len() as u64,
        undecodable: c.undecodable.len() as u64,
    }
}
fn line(kind: &str, payload: Value) -> String {
    json!({"timestamp":"2026-10-09T14:30:00Z","type":kind,"payload":payload}).to_string() + "\n"
}
fn header() -> String {
    line(
        "session_meta",
        json!({"id":"synthetic-codex-state","source":"cli"}),
    )
}
fn raw(namespace: &str, key: Vec<u8>, value: Vec<u8>) -> CellChange {
    CellChange {
        namespace: namespace.into(),
        key,
        value: Some(value),
    }
}

#[test]
fn sqlite_reopen_each_record_matches_whole_shared_core_and_retains_effects() {
    let bytes = include_bytes!("../../sources/tests/data/codex-stateful.jsonl");
    let whole = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes,
        0,
    )
    .output
    .unwrap();
    let mut expected = MemoryReadView::default();
    expected.apply(&whole);
    let mut world = World::new(bytes);
    let mut versions = 0;
    for record in bytes.split_inclusive(|b| *b == b'\n') {
        let delta = world.parse(record);
        versions += delta.pending_versions.len();
        world = world.reopen();
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let encoded = snapshot
            .cell(
                adapter::EFFECTS_NAMESPACE,
                &(world.head.token.step as u64).to_be_bytes(),
                budget().encoded_bytes,
            )
            .unwrap()
            .unwrap();
        let effects = adapter::Effects::decode(&encoded).unwrap();
        assert_eq!(
            effects.pending_versions,
            delta
                .pending_versions
                .iter()
                .map(|v| (v.message, v.evidence_revision))
                .collect::<Vec<_>>()
        );
        assert_eq!(effects.conflicts, delta.conflicts);
        assert_eq!(effects.pending_results, delta.pending_results);
        assert_eq!(effects.events, delta.events);
        assert_eq!(effects.links, delta.links);
    }
    assert!(versions > 0);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let resume = attempt.resume().unwrap();
    assert_eq!(resume.message_count(), whole.resume.message_count());
    assert_eq!(resume.parsed_upto, bytes.len() as u64);
    for (key, cell) in &expected.cells {
        assert_eq!(
            json!(attempt.read(key).unwrap()),
            json!(Some(cell)),
            "{key:?}"
        );
    }
    assert_ne!(world.head.status, "done");
}

#[test]
fn absence_resume_cumulative_admission_and_failure_fuse() {
    let world = World::new(b"x");
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(
        &snapshot,
        Budget {
            cells: 2,
            encoded_bytes: 4096,
        },
    );
    attempt.resume().unwrap();
    assert!(attempt.read(&Key::Call("absent".into())).unwrap().is_none());
    assert_eq!(attempt.usage().cells, 2);
    assert!(attempt.usage().encoded_bytes > adapter::RESUME_NAMESPACE.len());
    assert!(attempt.read(&Key::Prompt).is_err());
    assert!(matches!(
        attempt.take_failure(),
        Some(Error::ParserAdmission {
            resource: "Codex read cells",
            required: 3
        })
    ));
    let used = attempt.usage();
    assert!(attempt.read(&Key::Prompt).is_err());
    assert_eq!(used, attempt.usage());
    let attempt = adapter::Attempt::new(
        &snapshot,
        Budget {
            cells: 9,
            encoded_bytes: 64,
        },
    );
    attempt.resume().unwrap();
    assert!(attempt.read(&Key::Call("\0".repeat(1_000_000))).is_err());
    assert!(matches!(
        attempt.take_failure(),
        Some(Error::ParserAdmission {
            resource: "Codex read encoded bytes",
            ..
        })
    ));
}

#[test]
fn committed_resume_and_cell_corruption_are_typed_failures() {
    for bad in [
        None,
        Some(b"not json".to_vec()),
        Some({
            let mut r = Resume::default();
            r.parsed_upto = 1;
            serde_json::to_vec(&r).unwrap()
        }),
    ] {
        let mut world = World::new(b"xx");
        let changes = bad
            .map(|v| raw(adapter::RESUME_NAMESPACE, vec![], v))
            .into_iter()
            .collect::<Vec<_>>();
        world.install(&changes, 0, Accounting::default());
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        assert!(attempt.resume().is_err());
        assert!(matches!(attempt.take_failure(), Some(Error::Invariant(_))));
    }
    let mut world = World::new(b"xx");
    world.install(
        &[raw(
            "codex.wrapper",
            Key::Wrapper(10).ordered_bytes(),
            serde_json::to_vec(&Cell::Present).unwrap(),
        )],
        0,
        Accounting::default(),
    );
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    assert!(attempt.next(&Key::Wrapper(9)).is_err());
    assert!(matches!(attempt.take_failure(), Some(Error::Invariant(_))));
}

#[test]
fn encoded_admission_and_all_obligations_survive_roundtrip() {
    let mut delta = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        header().as_bytes(),
        0,
    )
    .output
    .unwrap();
    delta.pending_versions.push(PendingVersion {
        message: 3,
        evidence_revision: 7,
    });
    delta.pending_results.push((3, 5));
    delta.conflicts.push(NativeConflict {
        message: 3,
        fields: vec!["type".into()],
    });
    let encoded = adapter::encode_delta(&delta, 1, budget()).unwrap();
    let effects = encoded
        .changes
        .iter()
        .find(|c| c.namespace == adapter::EFFECTS_NAMESPACE)
        .unwrap();
    let decoded = adapter::Effects::decode(effects.value.as_ref().unwrap()).unwrap();
    assert_eq!(decoded.pending_versions, vec![(3, 7)]);
    assert_eq!(decoded.pending_results, vec![(3, 5)]);
    assert_eq!(decoded.conflicts, delta.conflicts);
    assert!(
        adapter::encode_delta(
            &delta,
            1,
            Budget {
                cells: encoded.changes.len() - 1,
                ..budget()
            }
        )
        .is_err()
    );
    assert!(
        adapter::encode_delta(
            &delta,
            1,
            Budget {
                encoded_bytes: encoded.encoded_bytes - 1,
                ..budget()
            }
        )
        .is_err()
    );
    assert_eq!(
        adapter::encode_delta(
            &delta,
            1,
            Budget {
                encoded_bytes: encoded.encoded_bytes,
                ..budget()
            }
        )
        .unwrap()
        .encoded_bytes,
        encoded.encoded_bytes
    );
    assert!(adapter::encode_delta(&delta, 0, budget()).is_err());
}

#[test]
fn sealed_partial_application_restarts_without_reinterpreting_and_rejects_stale_work() {
    let bytes = header();
    let world = World::new(bytes.as_bytes());
    let framed = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes.as_bytes(),
        0,
    );
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    let encoded = adapter::encode_delta(&delta, 1, budget()).unwrap();
    let World {
        _dir,
        root,
        mut writer,
        head,
    } = world;
    let mut work = writer
        .prepare_parser(
            &head,
            Progress {
                inspected_upto: framed.parsed_upto,
                parsed_upto: framed.parsed_upto,
                phase: Phase::Parsing,
                finalization_pending: true,
                counts: accounting(&framed.counts),
            },
        )
        .unwrap();
    work = writer
        .stage_parser(&work.token, &encoded.changes, budget())
        .unwrap();
    work = writer.seal_parser(&work.token).unwrap();
    work = writer
        .apply_parser(
            &work.token,
            Budget {
                cells: 1,
                ..budget()
            },
        )
        .unwrap();
    assert_eq!(
        writer
            .parser_head(head.token.input_pk, head.token.chain)
            .unwrap()
            .unwrap()
            .parsed_upto,
        0
    );
    drop(writer);
    let mut writer = Writer::open(&root).unwrap();
    let resumed = writer.parser_work(&head.token).unwrap().unwrap();
    assert_eq!(work.token, resumed.token);
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(Error::ParserWorkPending)
    ));
    while work.token.applied < work.token.staged {
        work = writer
            .apply_parser(
                &work.token,
                Budget {
                    cells: 1,
                    ..budget()
                },
            )
            .unwrap();
    }
    let published = writer.publish_parser(&work.token).unwrap();
    assert!(matches!(
        writer.parser_snapshot(&head.token),
        Err(Error::ParserPredecessorChanged)
    ));
    assert!(matches!(
        writer.publish_parser(&work.token),
        Err(Error::ParserPredecessorChanged)
    ));
    let snapshot = writer.parser_snapshot(&published.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    assert_eq!(attempt.resume().unwrap().parsed_upto, bytes.len() as u64);
    assert!(
        snapshot
            .cell(
                adapter::EFFECTS_NAMESPACE,
                &1u64.to_be_bytes(),
                budget().encoded_bytes
            )
            .unwrap()
            .is_some()
    );
}

fn prefix(n: usize) -> String {
    let mut bytes = header();
    for i in 0..n {
        bytes += &line(
            "response_item",
            json!({"type":"message","id":format!("u{i}"),"role":"user","content":[{"text":format!("prompt {i}")}]}),
        );
        bytes += &line(
            "response_item",
            json!({"type":"function_call","id":format!("a{i}"),"name":"tool","call_id":format!("c{i}"),"arguments":"{}"}),
        );
        bytes += &line(
            "response_item",
            json!({"type":"function_call_output","call_id":format!("c{i}"),"output":"Exit code: 0"}),
        );
        bytes += &line("event_msg", json!({"type":"token_count","info":null}));
    }
    bytes
}
#[test]
fn fixed_suffix_sqlite_access_does_not_scale_with_unrelated_unique_prefix() {
    let mut usage = Vec::new();
    for n in [3, 1000] {
        let bytes = prefix(n);
        let suffix = line(
            "response_item",
            json!({"type":"function_call_output","call_id":"c0","output":"Exit code: 0"}),
        );
        let mut world = World::new((bytes.clone() + &suffix).as_bytes());
        world.parse(bytes.as_bytes());
        world = world.reopen();
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let attempt = adapter::Attempt::new(
            &snapshot,
            Budget {
                cells: 256,
                encoded_bytes: 2 * 1024 * 1024,
            },
        );
        let framed = drive_jsonl(
            KeyedRecords::new(&attempt, attempt.resume().unwrap()),
            suffix.as_bytes(),
            world.head.parsed_upto,
        );
        let delta = framed.output.unwrap();
        assert!(!delta.reached_back);
        usage.push(attempt.usage());
    }
    assert_eq!(usage[0].cells, usage[1].cells);
    assert!(usage[1].cells <= 16);
    assert!(usage[1].encoded_bytes < 16384);
    assert!(
        usage[0].encoded_bytes.abs_diff(usage[1].encoded_bytes) < 256,
        "{usage:?}"
    );
}

#[test]
fn native_value_admission_survives_provider_callback_and_retry_starts_fresh() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"message","id":"prompt","role":"user","content":[{"text":"synthetic ".repeat(2048)}]}),
        );
    let suffix = line(
        "event_msg",
        json!({"type":"user_message","message":"different echo"}),
    );
    let mut world = World::new((bytes.clone() + &suffix).as_bytes());
    world.parse(bytes.as_bytes());
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(
        &snapshot,
        Budget {
            cells: 99,
            encoded_bytes: 4096,
        },
    );
    let framed = drive_jsonl(
        KeyedRecords::new(&attempt, attempt.resume().unwrap()),
        suffix.as_bytes(),
        world.head.parsed_upto,
    );
    assert!(framed.output.is_err());
    assert_eq!(framed.parsed_upto, world.head.parsed_upto);
    assert!(matches!(
        attempt.take_failure(),
        Some(Error::ParserAdmission {
            resource: "parser cell bytes",
            ..
        })
    ));
    let usage = attempt.usage();
    assert!(attempt.read(&Key::Prompt).is_err());
    assert_eq!(usage, attempt.usage());
    let retry = adapter::Attempt::new(&snapshot, budget());
    let framed = drive_jsonl(
        KeyedRecords::new(&retry, retry.resume().unwrap()),
        suffix.as_bytes(),
        world.head.parsed_upto,
    );
    assert!(framed.output.is_ok());
    assert!(retry.take_failure().is_none());
}
