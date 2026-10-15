//! Actual retained synthetic inputs, SQLite snapshots and the public prepared
//! barrier. No canonical Message writer or daemon completion is asserted.
use polylogue_engine::state::claude as adapter;
use polylogue_model::BlobHash;
use polylogue_sources::claude_code::keyed::{
    BodyPayload, Header, MemoryReadView, ReadView, ResponseKey,
};
use polylogue_sources::claude_code::{FileCtx, KeyedRecords, PARSER_VERSION};
use polylogue_sources::frame::{Counts, drive_jsonl};
use polylogue_store::parser_state::{Accounting, Basis, Budget, CellChange, Head, Phase, Progress};
use polylogue_store::{Error as StoreError, ImportObservation, InputFormat, Writer};
use serde_json::{Value, json};
use std::path::PathBuf;

fn broad() -> Budget {
    Budget {
        cells: 100_000,
        encoded_bytes: 64 * 1024 * 1024,
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
struct World {
    _dir: tempfile::TempDir,
    root: PathBuf,
    writer: Option<Writer>,
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
                InputFormat::ClaudeCodeJsonl,
                &ImportObservation {
                    id: "synthetic-claude",
                    machine: "synthetic-machine",
                    host: "synthetic-host",
                    path: "/synthetic/claude.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = writer
            .begin_parser(&Basis {
                input_pk: retained.input_pk,
                chain: retained.chain,
                parser_version: PARSER_VERSION,
                digest: *blob.bytes(),
                acquired_upto: retained.length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        Self {
            _dir: dir,
            root,
            writer: Some(writer),
            head,
        }
    }
    fn writer(&self) -> &Writer {
        self.writer.as_ref().unwrap()
    }
    fn restart(&mut self) {
        drop(self.writer.take());
        self.writer = Some(Writer::open(&self.root).unwrap());
        self.head = self
            .writer()
            .parser_head(self.head.token.input_pk, self.head.token.chain)
            .unwrap()
            .unwrap();
    }
    fn commit(&mut self, changes: &[CellChange], parsed: u64, counts: &Counts) {
        let end = self.head.acquired_upto;
        let progress = Progress {
            inspected_upto: parsed,
            parsed_upto: parsed,
            phase: if parsed == end {
                Phase::Finalizing
            } else {
                Phase::Parsing
            },
            finalization_pending: true,
            counts: accounting(counts),
        };
        let writer = self.writer.as_mut().unwrap();
        let mut work = writer.prepare_parser(&self.head, progress).unwrap();
        for chunk in changes.chunks(128) {
            work = writer.stage_parser(&work.token, chunk, broad()).unwrap();
        }
        work = writer.seal_parser(&work.token).unwrap();
        while work.token.applied < work.token.staged {
            work = writer
                .apply_parser(
                    &work.token,
                    Budget {
                        cells: 128,
                        encoded_bytes: broad().encoded_bytes,
                    },
                )
                .unwrap();
        }
        self.head = writer.publish_parser(&work.token).unwrap();
    }
    fn parse(&mut self, bytes: &[u8]) -> (adapter::Usage, usize) {
        let snapshot = self.writer().parser_snapshot(&self.head.token).unwrap();
        let view = adapter::View::new(&snapshot, broad()).unwrap();
        let framed = drive_jsonl(
            KeyedRecords::new(FileCtx::main("s"), &view).unwrap(),
            bytes,
            self.head.parsed_upto,
        );
        let mut output = framed.output.unwrap();
        assert!(view.take_error().is_none());
        let changes = adapter::encode_step(
            &mut output,
            &self.head,
            framed.parsed_upto,
            &framed.counts,
            broad(),
        )
        .unwrap();
        let size = changes
            .iter()
            .map(|c| c.namespace.len() + c.key.len() + c.value.as_ref().map_or(0, Vec::len))
            .sum();
        let usage = view.usage();
        drop(view);
        drop(snapshot);
        self.commit(&changes, framed.parsed_upto, &framed.counts);
        (usage, size)
    }
}
fn line(v: Value) -> String {
    v.to_string() + "\n"
}
fn user(uuid: &str) -> Value {
    json!({"type":"user","uuid":uuid,"sessionId":"s","promptSource":"typed","message":{"content":"synthetic"}})
}
fn assistant(uuid: &str, response: &str, request: &str) -> Value {
    json!({"type":"assistant","uuid":uuid,"sessionId":"s","requestId":request,"timestamp":"2026-10-01T00:00:00Z","message":{"id":response,"content":[{"type":"text","text":uuid}]}})
}
fn fixture() -> Vec<Value> {
    let mut side = user("side");
    side["isSidechain"] = json!(true);
    side["parentUuid"] = json!("u");
    let mut child = assistant("child", "m1", "r1");
    child["isSidechain"] = json!(true);
    child["parentUuid"] = json!("side");
    vec![
        user("u"),
        assistant("a1", "m1", "r1"),
        side,
        child,
        assistant("a2", "m2", "r2"),
        assistant("a3", "m1", "r1"),
        json!({"type":"assistant","uuid":"a1","message":{"content":[],"usage":{"output_tokens":0}}}),
        json!({"type":"custom-title","customTitle":"Synthetic title"}),
        json!({"type":"system","subtype":"compact_boundary"}),
        json!({"type":"continued-in","continuedInSessionId":"next"}),
        json!({"type":"new-synthetic-kind"}),
    ]
}

#[test]
fn reopen_each_line_preserves_effective_replacement_and_pending_evidence_cells() {
    let records = fixture();
    let text: String = records.iter().cloned().map(line).collect();
    let mut memory = MemoryReadView::default();
    let whole = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        text.as_bytes(),
        0,
    );
    let mut expected = whole.output.unwrap();
    expected.header.parsed_upto = whole.parsed_upto;
    memory.apply(expected.header.clone(), expected.delta.clone());
    let mut world = World::new(text.as_bytes());
    let mut events = Vec::new();
    let mut links = Vec::new();
    let mut unrecognized = Vec::new();
    for record in records {
        world.parse(line(record).as_bytes());
        world.restart();
        let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
        let view = adapter::View::new(&snapshot, broad()).unwrap();
        let effects = view.effects(world.head.token.step).unwrap().unwrap();
        events.extend(effects.events);
        links.extend(effects.links);
        unrecognized.extend(effects.unrecognized);
    }
    assert_eq!(events, expected.events);
    assert_eq!(links, expected.links);
    assert_eq!(
        unrecognized,
        whole
            .counts
            .unrecognized
            .iter()
            .map(|(s, n)| (s.to_string(), *n))
            .collect::<Vec<_>>()
    );
    assert_eq!(world.head.counts, accounting(&whole.counts));
    assert!(world.head.finalization_pending);
    assert_eq!(world.head.status, "finalizing");
    let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    assert_eq!(view.header().unwrap(), memory.header().unwrap());
    for native in ["s", "s~side-side"] {
        assert_eq!(
            view.session(native).unwrap(),
            memory.session(native).unwrap()
        );
    }
    for (native, key) in [
        ("s", ResponseKey::new("m1", "r1")),
        ("s", ResponseKey::new("m2", "r2")),
        ("s~side-side", ResponseKey::new("m1", "r1")),
    ] {
        let cell = view.response(native, &key).unwrap().unwrap();
        assert_eq!(Some(cell.clone()), memory.response(native, &key).unwrap());
        for item in 0..cell.items {
            assert_eq!(
                view.contribution(&cell.message.id, item).unwrap(),
                memory.contribution(&cell.message.id, item).unwrap()
            );
        }
        let pending = view.pending(&cell.message.id).unwrap().unwrap();
        assert_eq!(pending.revision, cell.revision);
        assert_eq!(pending.items, cell.items);
        assert!(cell.message.version_hash.is_none());
    }
    assert!(view.take_error().is_none());
}

fn prefix_cost(size: usize) -> (adapter::Usage, usize) {
    let mut prefix = line(user("u"));
    for n in 0..size {
        prefix += &line(assistant(&format!("a{n:05}"), &format!("m{n:05}"), "r"));
    }
    let append = line(assistant("continued", "m00000", "r"));
    let text = prefix.clone() + &append;
    let mut world = World::new(text.as_bytes());
    world.parse(prefix.as_bytes());
    world.restart();
    let cost = world.parse(append.as_bytes());
    assert!(cost.0.encoded_bytes < 8192 && cost.1 < 8192);
    cost
}
#[test]
fn append_only_loads_point_cells_after_large_unique_prefix() {
    let small = prefix_cost(2);
    let large = prefix_cost(1000);
    assert_eq!(small.0.cells, large.0.cells);
    assert!(small.0.encoded_bytes.abs_diff(large.0.encoded_bytes) < 64);
    assert!(small.1.abs_diff(large.1) < 64);
}

#[test]
fn header_default_is_genesis_only_and_missing_or_corrupt_committed_header_is_typed_failure() {
    let text = line(user("u"));
    let mut world = World::new(text.as_bytes());
    {
        let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
        let view = adapter::View::new(&snapshot, broad()).unwrap();
        assert_eq!(view.header().unwrap(), Header::default());
    }
    world.parse(text.as_bytes());
    let position = world.head.parsed_upto;
    for value in [Some(b"invalid-json".to_vec()), None] {
        world.commit(
            &[CellChange {
                namespace: "claude.header.v1".into(),
                key: b"null".to_vec(),
                value,
            }],
            position,
            &Counts::default(),
        );
        world.restart();
        let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
        let view = adapter::View::new(&snapshot, broad()).unwrap();
        assert!(view.header().is_err());
        assert!(matches!(view.take_error(), Some(StoreError::Invariant(_))));
    }
}

#[test]
fn cumulative_read_and_encode_admission_preserve_original_store_error() {
    let text = line(user("u"));
    let mut world = World::new(text.as_bytes());
    world.parse(text.as_bytes());
    let committed = world.head.token.clone();
    let snapshot = world.writer().parser_snapshot(&committed).unwrap();
    let full = adapter::View::new(&snapshot, broad()).unwrap();
    full.header().unwrap();
    full.session("s").unwrap();
    let total = full.usage().encoded_bytes;
    let limited = adapter::View::new(
        &snapshot,
        Budget {
            cells: 10,
            encoded_bytes: total - 1,
        },
    )
    .unwrap();
    limited.header().unwrap();
    assert!(limited.session("s").is_err());
    assert!(matches!(
        limited.take_error(),
        Some(StoreError::ParserAdmission { .. })
    ));
    let attempts = adapter::View::new(
        &snapshot,
        Budget {
            cells: 2,
            encoded_bytes: 4096,
        },
    )
    .unwrap();
    attempts.header().unwrap();
    assert!(attempts.session("absent").unwrap().is_none());
    assert!(attempts.side("absent-too").is_err());
    assert_eq!(attempts.usage().cells, 3);
    assert!(matches!(
        attempts.take_error(),
        Some(StoreError::ParserAdmission { required: 3, .. })
    ));
    let key_limited = adapter::View::new(
        &snapshot,
        Budget {
            cells: 10,
            encoded_bytes: 64,
        },
    )
    .unwrap();
    assert!(key_limited.session(&"x".repeat(10000)).is_err());
    assert!(key_limited.usage().encoded_bytes <= 64);
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        text.as_bytes(),
        0,
    );
    let mut output = framed.output.unwrap();
    assert!(matches!(
        adapter::encode_step(
            &mut output,
            &snapshot.head,
            position(&snapshot.head),
            &framed.counts,
            Budget {
                cells: 0,
                encoded_bytes: 4096
            }
        ),
        Err(StoreError::ParserAdmission { .. })
    ));
    let cells = adapter::encode_step(
        &mut output,
        &snapshot.head,
        position(&snapshot.head),
        &framed.counts,
        broad(),
    )
    .unwrap();
    let encoded: usize = cells
        .iter()
        .map(|c| c.namespace.len() + c.key.len() + c.value.as_ref().map_or(0, Vec::len))
        .sum();
    assert!(matches!(
        adapter::encode_step(
            &mut output,
            &snapshot.head,
            position(&snapshot.head),
            &framed.counts,
            Budget {
                cells: broad().cells,
                encoded_bytes: encoded - 1
            }
        ),
        Err(StoreError::ParserAdmission { .. })
    ));
    assert_eq!(
        world
            .writer()
            .parser_head(committed.input_pk, committed.chain)
            .unwrap()
            .unwrap()
            .token,
        committed
    );
}
fn position(head: &Head) -> u64 {
    head.parsed_upto
}

#[test]
fn invalid_sql_cell_fails_source_callback_and_cannot_publish_unknown_or_checkpoint() {
    let first = line(user("u"));
    let append = line(assistant("a", "m", "r"));
    let mut world = World::new((first.clone() + &append).as_bytes());
    world.parse(first.as_bytes());
    world.commit(
        &[CellChange {
            namespace: "claude.session.v1".into(),
            key: serde_json::to_vec("s").unwrap(),
            value: Some(b"invalid-json".to_vec()),
        }],
        world.head.parsed_upto,
        &Counts::default(),
    );
    let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &view).unwrap(),
        append.as_bytes(),
        world.head.parsed_upto,
    );
    assert!(framed.output.is_err());
    assert_eq!(framed.parsed_upto, world.head.parsed_upto);
    assert_eq!(framed.counts.used, 0);
    assert!(framed.counts.unrecognized.is_empty());
    assert!(matches!(view.take_error(), Some(StoreError::Invariant(_))));
}

#[test]
fn pinned_body_namespace_distinguishes_unavailable_from_invalid_cells() {
    let text = line(
        json!({"type":"user","uuid":"u","toolUseResult":{"persistedOutputPath":"c.txt"},"message":{"content":[{"type":"tool_result","tool_use_id":"c","content":"preview"}]}}),
    );
    let mut world = World::new(text.as_bytes());
    world.parse(text.as_bytes());
    {
        let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
        let view = adapter::View::new(&snapshot, broad()).unwrap();
        assert!(view.body("missing.txt").unwrap().is_none());
        assert!(view.take_error().is_none());
        let response = view
            .response("s", &ResponseKey::user("u"))
            .unwrap()
            .unwrap();
        assert!(
            view.pending(&response.message.id)
                .unwrap()
                .unwrap()
                .incomplete
        );
    }
    let bytes = b"synthetic retained body";
    let blob = world.writer.as_mut().unwrap().put_file_blob(bytes).unwrap();
    let payload = BodyPayload {
        blob,
        length: bytes.len() as u64,
    };
    let manifest = serde_json::to_string(&payload).unwrap();
    let mut basis_bytes = world.head.token.basis.to_vec();
    basis_bytes.extend_from_slice(manifest.as_bytes());
    let basis = Basis {
        input_pk: world.head.token.input_pk,
        chain: world.head.token.chain,
        parser_version: PARSER_VERSION,
        digest: *BlobHash::of(&basis_bytes).bytes(),
        acquired_upto: world.head.acquired_upto,
        companions_json: Some(&manifest),
        excisions_seen: 0,
    };
    world.head = world.writer.as_mut().unwrap().begin_parser(&basis).unwrap();
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        b"",
        0,
    );
    let mut output = framed.output.unwrap();
    let mut cells =
        adapter::encode_step(&mut output, &world.head, 0, &Counts::default(), broad()).unwrap();
    cells.push(adapter::encode_body("c.txt", Some(&payload), broad()).unwrap());
    world.commit(&cells, 0, &Counts::default());
    world.restart();
    world.parse(text.as_bytes());
    {
        let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
        let view = adapter::View::new(&snapshot, broad()).unwrap();
        assert_eq!(view.body("c.txt").unwrap(), Some(payload.clone()));
        let response = view
            .response("s", &ResponseKey::user("u"))
            .unwrap()
            .unwrap();
        assert!(
            !view
                .pending(&response.message.id)
                .unwrap()
                .unwrap()
                .incomplete
        );
        assert_eq!(
            view.contribution(&response.message.id, 0)
                .unwrap()
                .unwrap()
                .bodies[0]
                .payload,
            Some(payload)
        );
    }
    world.commit(
        &[CellChange {
            namespace: "claude.body.v1".into(),
            key: serde_json::to_vec("c.txt").unwrap(),
            value: Some(b"invalid-json".to_vec()),
        }],
        world.head.parsed_upto,
        &Counts::default(),
    );
    let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
    let view = adapter::View::new(&snapshot, broad()).unwrap();
    assert!(view.body("c.txt").is_err());
    assert!(matches!(view.take_error(), Some(StoreError::Invariant(_))));
}
