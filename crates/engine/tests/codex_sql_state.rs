//! Real SQLite state tests. These do not certify canonical output publication.
use polylogue_engine::state::codex as adapter;
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
            .store_immutable_input(
                blob,
                InputFormat::CodexRollout,
                &[],
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
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let effects = attempt
            .effects(world.head.token.step as u64)
            .unwrap()
            .collect()
            .unwrap();
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
    let mut world = World::new(b"");
    world.install(&encoded.changes, 0, Accounting::default());
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let decoded = attempt.effects(1).unwrap().collect().unwrap();
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
        Err(Error::StaleParserState)
    ));
    assert!(matches!(
        writer.publish_parser(&work.token),
        Err(Error::StaleParserState)
    ));
    let snapshot = writer.parser_snapshot(&published.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    assert_eq!(attempt.resume().unwrap().parsed_upto, bytes.len() as u64);
    assert!(attempt.effects(1).unwrap().collect().is_ok());
}

fn prefix(n: usize) -> String {
    let mut bytes = header();
    // Keep the early suffix unsettled while unrelated completed calls grow.
    bytes += &line(
        "response_item",
        json!({"type":"function_call","id":"early","name":"exec_command","call_id":"early","arguments":"{}"}),
    );
    bytes += &line(
        "response_item",
        json!({"type":"function_call_output","call_id":"early","output":"Process running with session ID 7"}),
    );
    bytes += &line("event_msg", json!({"type":"token_count","info":null}));
    for i in 0..n {
        bytes += &line(
            "response_item",
            json!({"type":"message","id":format!("u{i}"),"role":"user","content":[{"text":format!("prompt {i}")}]}),
        );
        bytes += &line(
            "response_item",
            json!({"type":"function_call","id":format!("a{i}"),"name":"exec_command","call_id":format!("c{i}"),"arguments":"{}"}),
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
    assert!(usage[1].cells <= 33);
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

#[test]
fn sortable_neighbors_and_distinct_effects_steps_survive_equal_raw_offsets() {
    let mut world = World::new(b"x");
    let mut resume = Resume::default();
    resume.parsed_upto = 0;
    let delta = Delta {
        resume,
        cells: std::collections::BTreeMap::from([
            (Key::Wrapper(9), Some(Cell::Text("nine".into()))),
            (Key::Wrapper(10), Some(Cell::Text("ten".into()))),
            (Key::Wrapper(256), Some(Cell::Text("later".into()))),
        ]),
        events: vec![],
        links: vec![],
        reached_back: false,
        pending_versions: vec![],
        conflicts: vec![],
        pending_results: vec![],
    };
    let first = adapter::encode_delta(&delta, 1, budget()).unwrap();
    world.install(&first.changes, 0, Accounting::default());
    let second = adapter::encode_delta(&delta, 2, budget()).unwrap();
    world.install(&second.changes, 0, Accounting::default());
    world = world.reopen();
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    attempt.resume().unwrap();
    assert!(matches!(
        attempt.next(&Key::Wrapper(9)).unwrap(),
        Some((Key::Wrapper(10), Cell::Text(_)))
    ));
    assert!(matches!(
        attempt.previous(&Key::Wrapper(256)).unwrap(),
        Some((Key::Wrapper(10), Cell::Text(_)))
    ));
    assert!(attempt.next(&Key::Model(0)).unwrap().is_none());
    for step in [1u64, 2] {
        assert!(attempt.effects(step).unwrap().collect().is_ok());
    }
}

fn obligations(n: usize) -> Delta {
    let mut delta = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        header().as_bytes(),
        0,
    )
    .output
    .unwrap();
    delta.pending_versions = (0..n)
        .map(|message| PendingVersion {
            message,
            evidence_revision: 7,
        })
        .collect();
    delta.pending_results = vec![(3, 5)];
    delta.conflicts = vec![NativeConflict {
        message: 3,
        fields: vec!["type".into()],
    }];
    delta.reached_back = true;
    delta
}
fn install_obligations(delta: &Delta) -> World {
    let mut world = World::new(b"");
    let encoded = adapter::encode_delta(delta, 1, budget()).unwrap();
    world.install(&encoded.changes, 0, Accounting::default());
    world.reopen()
}
#[test]
fn effect_cursor_reopens_from_scalar_and_collects_same_units_with_point_cost() {
    let delta = obligations(257);
    let world = install_obligations(&delta);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let reader = attempt.effects(1).unwrap();
    let expected = reader.collect().unwrap();
    let mut cursor = reader.start();
    loop {
        let before = attempt.usage();
        let step = reader.advance(&mut cursor).unwrap();
        assert!(attempt.usage().cells - before.cells <= 1);
        if matches!(
            step,
            adapter::EffectStep::Item(adapter::Effect::PendingVersion { message: 128, .. })
        ) {
            break;
        }
    }
    let serialized = serde_json::to_vec(&cursor).unwrap();
    assert!(serialized.len() < 80);
    let mut cursor: adapter::EffectsCursor = serde_json::from_slice(&serialized).unwrap();
    let resumed_attempt = adapter::Attempt::new(&snapshot, budget());
    let resumed = resumed_attempt.effects(1).unwrap();
    assert!(matches!(
        resumed.advance(&mut cursor).unwrap(),
        adapter::EffectStep::Item(adapter::Effect::PendingVersion {
            message: 129,
            evidence_revision: 7
        })
    ));
    assert_eq!(expected.pending_versions.len(), 257);
    assert_eq!(expected.pending_results, delta.pending_results);
    assert_eq!(expected.conflicts, delta.conflicts);
    assert!(expected.reached_back);
    assert!(attempt.take_failure().is_none());
}
#[test]
fn first_effect_point_cost_does_not_depend_on_unvisited_receipt_suffix() {
    fn first(n: usize) -> adapter::ReadUsage {
        let world = install_obligations(&obligations(n));
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let reader = attempt.effects(1).unwrap();
        let mut cursor = reader.start();
        // Empty event/link families each expose their own single absence probe.
        assert!(matches!(
            reader.advance(&mut cursor).unwrap(),
            adapter::EffectStep::Boundary
        ));
        assert!(matches!(
            reader.advance(&mut cursor).unwrap(),
            adapter::EffectStep::Boundary
        ));
        let before = attempt.usage();
        assert!(matches!(
            reader.advance(&mut cursor).unwrap(),
            adapter::EffectStep::Item(adapter::Effect::PendingVersion { message: 0, .. })
        ));
        let after = attempt.usage();
        adapter::ReadUsage {
            cells: after.cells - before.cells,
            encoded_bytes: after.encoded_bytes - before.encoded_bytes,
        }
    }
    assert_eq!(first(1), first(257));
}
#[test]
fn effect_header_payload_cursor_corruption_and_admission_are_fatal_and_fused() {
    for corruption in ["missing", "kind", "count", "layout", "step"] {
        let mut world = World::new(b"");
        let mut encoded = adapter::encode_delta(&obligations(2), 1, budget()).unwrap();
        let header = encoded
            .changes
            .iter_mut()
            .find(|c| c.namespace == adapter::EFFECTS_NAMESPACE && c.key[8] == 0)
            .unwrap();
        let mut value: Value = serde_json::from_slice(header.value.as_ref().unwrap()).unwrap();
        match corruption {
            "count" => value["counts"][2] = json!(0),
            "layout" => value["version"] = json!(adapter::STATE_LAYOUT_VERSION - 1),
            "step" => value["step"] = json!(2),
            _ => {}
        }
        header.value = Some(serde_json::to_vec(&value).unwrap());
        if corruption == "missing" {
            encoded.changes.retain(|c| {
                !(c.namespace == adapter::EFFECTS_NAMESPACE && c.key[8] == 3 && c.key[16] == 0)
            });
        }
        if corruption == "kind" {
            let unit = encoded
                .changes
                .iter_mut()
                .find(|c| c.namespace == adapter::EFFECTS_NAMESPACE && c.key[8] == 3)
                .unwrap();
            unit.value = Some(serde_json::to_vec(&adapter::Effect::ReachedBack).unwrap());
        }
        world.install(&encoded.changes, 0, Accounting::default());
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let result = attempt.effects(1).and_then(|r| r.collect());
        assert!(result.is_err(), "{corruption}");
        assert!(
            matches!(attempt.take_failure(), Some(Error::Invariant(_))),
            "{corruption}"
        );
        assert!(attempt.effects(1).is_err());
    }
    let world = install_obligations(&obligations(2));
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(
        &snapshot,
        Budget {
            cells: 1,
            ..budget()
        },
    );
    let reader = attempt.effects(1).unwrap();
    let mut cursor = reader.start();
    let before = cursor;
    assert!(reader.advance(&mut cursor).is_err());
    assert_eq!(cursor, before);
    assert!(matches!(
        attempt.take_failure(),
        Some(Error::ParserAdmission { .. })
    ));
    assert!(reader.advance(&mut cursor).is_err());
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let reader = attempt.effects(1).unwrap();
    let mut invalid: adapter::EffectsCursor =
        serde_json::from_value(json!({"step":1,"kind":3,"ordinal":999})).unwrap();
    assert!(reader.advance(&mut invalid).is_err());
    assert!(matches!(attempt.take_failure(), Some(Error::Invariant(_))));
}
#[test]
fn legacy_aggregate_receipt_requires_explicit_layout_replay() {
    let mut world = World::new(b"");
    world.install(
        &[raw(
            "codex.effects",
            1u64.to_be_bytes().to_vec(),
            br#"{"pending_versions":[[1,7]]}"#.to_vec(),
        )],
        0,
        Accounting::default(),
    );
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    assert!(attempt.effects(1).is_err());
    assert!(matches!(attempt.take_failure(), Some(Error::Invariant(_))));
}

#[test]
fn native_event_and_link_units_use_the_same_typed_wire_as_cursor_collection() {
    let bytes = line(
        "session_meta",
        json!({"id":"synthetic-child","source":"cli","forked_from_id":"synthetic-parent","base_instructions":{"text":"synthetic instructions"}}),
    );
    let delta = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes.as_bytes(),
        0,
    )
    .output
    .unwrap();
    assert!(!delta.events.is_empty());
    assert!(!delta.links.is_empty());
    let world = install_obligations(&delta);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let reader = attempt.effects(1).unwrap();
    let effects = reader.collect().unwrap();
    assert_eq!(effects.events, delta.events);
    assert_eq!(effects.links, delta.links);
    let mut cursor = reader.start();
    assert!(matches!(
        reader.advance(&mut cursor).unwrap(),
        adapter::EffectStep::Item(adapter::Effect::Event(_))
    ));
    assert!(matches!(
        reader.advance(&mut cursor).unwrap(),
        adapter::EffectStep::Boundary
    ));
    assert!(matches!(
        reader.advance(&mut cursor).unwrap(),
        adapter::EffectStep::Item(adapter::Effect::Link(_))
    ));
}

fn collect_impacts(reader: &adapter::ImpactsReader<'_, '_>) -> Vec<adapter::Impact> {
    let mut cursor = reader.start();
    let mut out = Vec::new();
    loop {
        match reader.advance(&mut cursor).unwrap() {
            adapter::ImpactStep::Item(v) => out.push(v),
            adapter::ImpactStep::Boundary => {}
            adapter::ImpactStep::Done => return out,
        }
    }
}
#[test]
fn message_impacts_union_version_body_presence_result_and_deleted_witness() {
    use polylogue_model::merge::{MessageField, Slot};
    let mut delta = obligations(0);
    delta.pending_versions.push(PendingVersion {
        message: 7,
        evidence_revision: 9,
    });
    delta.cells.insert(Key::Presence(2, "at".into()), None);
    delta.cells.insert(
        Key::PresenceBlock(2, 4),
        Some(Cell::Presence(Default::default())),
    );
    delta.cells.insert(
        Key::OutputChunk(3, 5, 9),
        Some(Cell::Text("changed".into())),
    );
    let world = install_obligations(&delta);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let effects = attempt.effects(1).unwrap().collect().unwrap();
    assert_eq!(effects.changed_messages, vec![2, 3, 7]);
    let reader = attempt.message_impacts(1, 2).unwrap();
    assert!(!reader.header_dirty());
    assert_eq!(
        collect_impacts(&reader),
        vec![
            adapter::Impact::Block { block: 4 },
            adapter::Impact::MessageWitness {
                key: "at".into(),
                slot: Slot::Message(MessageField::At)
            }
        ]
    );
    assert_eq!(
        collect_impacts(&attempt.message_impacts(1, 3).unwrap()),
        vec![adapter::Impact::Block { block: 5 }]
    );
    assert!(collect_impacts(&attempt.message_impacts(1, 7).unwrap()).is_empty());
    // A deleted unsupported witness must refuse; its semantic slot is never
    // guessed from a generated value or silently left in canonical rank state.
    delta
        .cells
        .insert(Key::Presence(9, "unknown-future-field".into()), None);
    assert!(adapter::encode_delta(&delta, 2, budget()).is_err());
}
#[test]
fn timestamp_only_delta_emits_work_without_manufacturing_a_version_change() {
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"message","id":"answer","role":"assistant","content":[{"text":"answer"}]}),
        );
    let mut world = World::new(bytes.as_bytes());
    let first = world.parse(bytes.as_bytes());
    // A synthetic mapped timestamp-only correction tests the adapter seam;
    // this does not assert that a repeated native item changes its first time.
    let mut cells = std::collections::BTreeMap::new();
    let mut witness = first
        .cells
        .get(&Key::Presence(0, "at".into()))
        .unwrap()
        .clone()
        .unwrap();
    if let Cell::Witness(w) = &mut witness {
        w.source_time.as_mut().unwrap().0 += 5000;
        w.canonical = br#""2026-10-09T14:30:05Z""#.to_vec();
    }
    cells.insert(Key::Presence(0, "at".into()), Some(witness));
    let delta = Delta {
        resume: first.resume,
        cells,
        events: vec![],
        links: vec![],
        reached_back: false,
        pending_versions: vec![],
        conflicts: vec![],
        pending_results: vec![],
    };
    let encoded = adapter::encode_delta(&delta, 2, budget()).unwrap();
    let end = world.head.parsed_upto;
    world.install(&encoded.changes, end, Accounting::default());
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let effects = attempt.effects(2).unwrap().collect().unwrap();
    assert!(effects.pending_versions.is_empty());
    assert_eq!(effects.changed_messages, vec![0]);
    assert!(
        collect_impacts(&attempt.message_impacts(2, 0).unwrap())
            .iter()
            .any(|v| matches!(v,adapter::Impact::MessageWitness {key,..} if key=="at"))
    );
}
#[test]
fn per_message_point_cursor_reopens_without_visiting_other_messages() {
    fn selected(n: usize) -> (adapter::ReadUsage, Vec<adapter::Impact>) {
        let mut delta = obligations(n);
        delta.cells.insert(
            Key::PresenceBlock(0, 9),
            Some(Cell::Presence(Default::default())),
        );
        let world = install_obligations(&delta);
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let reader = attempt.message_impacts(1, 0).unwrap();
        let mut cursor = reader.start();
        let first = reader.advance(&mut cursor).unwrap();
        assert!(matches!(
            first,
            adapter::ImpactStep::Item(adapter::Impact::Block { block: 9 })
        ));
        let serialized = serde_json::to_vec(&cursor).unwrap();
        assert!(serialized.len() < 100);
        let mut restored: adapter::ImpactCursor = serde_json::from_slice(&serialized).unwrap();
        let before = attempt.usage();
        assert!(matches!(
            reader.advance(&mut restored).unwrap(),
            adapter::ImpactStep::Boundary
        ));
        assert_eq!(attempt.usage().cells - before.cells, 1);
        let units = collect_impacts(&reader);
        (attempt.usage(), units)
    }
    let small = selected(1);
    let large = selected(257);
    assert_eq!(small, large);
}
#[test]
fn call_annotation_fanout_remains_one_typed_call_impact() {
    let mut world = World::new(b"");
    let mut prefix = obligations(0);
    prefix.pending_results.clear();
    prefix.conflicts.clear();
    prefix.reached_back = false;
    for message in 0..257 {
        prefix.cells.insert(
            Key::CallBlock("shared-call".into(), message, 0),
            Some(Cell::Present),
        );
    }
    let encoded = adapter::encode_delta(&prefix, 1, budget()).unwrap();
    world.install(&encoded.changes, 0, Accounting::default());
    let mut delta = obligations(0);
    delta.cells.clear();
    delta.pending_results.clear();
    delta.conflicts.clear();
    delta.reached_back = false;
    delta.cells.insert(
        Key::ChildCount("shared-call".into()),
        Some(Cell::Number(10000)),
    );
    let encoded = adapter::encode_delta(&delta, 2, budget()).unwrap();
    assert_eq!(encoded.changes.len(), 4);
    world.install(&encoded.changes, 0, Accounting::default());
    world = world.reopen();
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let effects = attempt.effects(2).unwrap().collect().unwrap();
    assert_eq!(effects.call_impacts, vec!["shared-call"]);
    assert!(effects.changed_messages.is_empty());
}
#[test]
fn impact_count_payload_and_cumulative_admission_fail_without_cursor_advance() {
    for corruption in ["missing", "kind", "count"] {
        let mut world = World::new(b"");
        let mut delta = obligations(0);
        delta.cells.insert(
            Key::PresenceBlock(0, 2),
            Some(Cell::Presence(Default::default())),
        );
        let mut encoded = adapter::encode_delta(&delta, 1, budget()).unwrap();
        if corruption == "missing" {
            encoded.changes.retain(|c| {
                !(c.namespace == adapter::EFFECTS_NAMESPACE
                    && c.key.len() == 26
                    && c.key[8] == 9
                    && c.key[17] == 1)
            });
        } else if corruption == "count" {
            let header = encoded
                .changes
                .iter_mut()
                .find(|c| {
                    c.namespace == adapter::EFFECTS_NAMESPACE
                        && c.key.len() == 26
                        && c.key[9..17] == 0u64.to_be_bytes()
                        && c.key[17] == 0
                })
                .unwrap();
            let mut value: Value = serde_json::from_slice(header.value.as_ref().unwrap()).unwrap();
            value["counts"][0] = json!(0);
            header.value = Some(serde_json::to_vec(&value).unwrap());
        } else {
            let unit = encoded
                .changes
                .iter_mut()
                .find(|c| {
                    c.namespace == adapter::EFFECTS_NAMESPACE && c.key.len() == 26 && c.key[17] == 1
                })
                .unwrap();
            unit.value = Some(
                serde_json::to_vec(&adapter::Impact::MessageWitness {
                    key: "model".into(),
                    slot: polylogue_model::merge::Slot::Message(
                        polylogue_model::merge::MessageField::Model,
                    ),
                })
                .unwrap(),
            );
        }
        world.install(&encoded.changes, 0, Accounting::default());
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let reader = attempt.message_impacts(1, 0).unwrap();
        let mut cursor = reader.start();
        let before = cursor;
        assert!(reader.advance(&mut cursor).is_err());
        assert_eq!(cursor, before);
        assert!(matches!(attempt.take_failure(), Some(Error::Invariant(_))));
    }
    let mut delta = obligations(0);
    delta.cells.insert(
        Key::PresenceBlock(0, 2),
        Some(Cell::Presence(Default::default())),
    );
    let world = install_obligations(&delta);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let attempt = adapter::Attempt::new(
        &snapshot,
        Budget {
            cells: 1,
            ..budget()
        },
    );
    let reader = attempt.message_impacts(1, 0).unwrap();
    let mut cursor = reader.start();
    assert!(reader.advance(&mut cursor).is_err());
    assert!(matches!(
        attempt.take_failure(),
        Some(Error::ParserAdmission { .. })
    ));
}
