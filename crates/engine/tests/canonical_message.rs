//! Real retained Codex parse -> worker actions -> durable source certificate.
//! Canonical election/search/export and complete Effects consumption are open.
use polylogue_engine::{canonical_message as m, state::codex as adapter};
use polylogue_model::{
    BlockSlot,
    merge::{BlockField, MessageField, Slot},
};
use polylogue_sources::{
    codex::{self, keyed::KeyedRecords},
    frame::drive_jsonl,
};
use polylogue_store::{
    ImportObservation, InputFormat, Writer,
    contributions::{MessageLayout, ProjectionStep, SourceProjection},
    parser_state::{Accounting, Basis, Budget, Head, Phase, Progress},
};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::atomic::AtomicBool};
fn budget() -> Budget {
    Budget {
        cells: 65536,
        encoded_bytes: 32 * 1024 * 1024,
    }
}
fn work_budget() -> m::Budgets {
    m::Budgets {
        read: budget(),
        metadata: 65536,
        value_chunk: 65536,
        hash_bytes: 65536,
    }
}
fn line(kind: &str, payload: Value) -> String {
    json!({"type":kind,"timestamp":"2026-10-01T00:00:00Z","payload":payload}).to_string() + "\n"
}
fn text_input(text: &str) -> Vec<u8> {
    [
    line("session_meta",json!({"id":"synthetic-session","cwd":"/synthetic"})),
    line("response_item",json!({"type":"message","id":"response","role":"assistant","phase":null,"content":[{"type":"output_text","text":text}]})),
].concat().into_bytes()
}
struct World {
    dir: tempfile::TempDir,
    root: PathBuf,
    writer: Writer,
    head: Head,
    source_binding: polylogue_model::BlobHash,
}
impl World {
    fn new(bytes: &[u8]) -> Self {
        Self::with_parsed(bytes, bytes)
    }
    fn with_parsed(retained: &[u8], bytes: &[u8]) -> Self {
        Self::with_mutation(retained, bytes, |_| {})
    }
    fn with_mutation(
        retained: &[u8],
        bytes: &[u8],
        mutate: impl FnOnce(&mut codex::keyed::Delta),
    ) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut writer = Writer::open(&root).unwrap();
        let blob = writer.put_file_blob(retained).unwrap();
        let input = writer
            .retain_immutable_input(
                blob,
                InputFormat::CodexRollout,
                &[],
                &ImportObservation {
                    id: "synthetic-materializer",
                    machine: "synthetic",
                    host: "synthetic",
                    path: "synthetic/codex.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = writer
            .begin_parser(&Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: codex::PARSER_VERSION,
                digest: [7; 32],
                acquired_upto: bytes.len() as u64,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let snapshot = writer.parser_snapshot(&head.token).unwrap();
        let attempt = adapter::Attempt::new(&snapshot, budget());
        let framed = drive_jsonl(
            KeyedRecords::new(&attempt, attempt.resume().unwrap()),
            bytes,
            0,
        );
        let mut delta = framed.output.unwrap();
        mutate(&mut delta);
        assert!(attempt.take_failure().is_none());
        delta.resume.parsed_upto = framed.parsed_upto;
        let encoded = adapter::encode_delta(&delta, 1, budget()).unwrap();
        drop(attempt);
        drop(snapshot);
        let mut prepared = writer
            .prepare_parser(
                &head,
                Progress {
                    inspected_upto: framed.parsed_upto,
                    parsed_upto: framed.parsed_upto,
                    phase: Phase::Finalizing,
                    finalization_pending: true,
                    counts: Accounting {
                        framed: framed.counts.framed,
                        used: framed.counts.used,
                        ignored: framed.counts.ignored.values().sum(),
                        unrecognized: framed.counts.unrecognized.len() as u64,
                        undecodable: framed.counts.undecodable.len() as u64,
                    },
                },
            )
            .unwrap();
        prepared = writer
            .stage_parser(&prepared.token, &encoded.changes, budget())
            .unwrap();
        prepared = writer.seal_parser(&prepared.token).unwrap();
        while prepared.token.applied < prepared.token.staged {
            prepared = writer.apply_parser(&prepared.token, budget()).unwrap();
        }
        let head = writer.publish_parser(&prepared.token).unwrap();
        Self {
            dir,
            root,
            writer,
            head,
            source_binding: blob,
        }
    }
    fn request(&self) -> m::Request {
        m::Request {
            message: 0,
            revision: 1,
            logical_key: [3; 32],
            source_binding: self.source_binding,
        }
    }
    fn begin(&mut self) -> m::Handle {
        let request = self.request();
        let current = self
            .writer
            .current_source_projection(&self.head.token, &request.logical_key, 65536)
            .unwrap();
        let snapshot = self.writer.parser_snapshot(&self.head.token).unwrap();
        let action = m::start(&snapshot, &request, current, budget(), 65536).unwrap();
        drop(snapshot);
        match m::apply(&mut self.writer, action).unwrap() {
            m::Applied::Prepared(h) => h,
            _ => panic!("expected resumable build"),
        }
    }
    fn compute(&mut self, h: &m::Handle) -> m::Action {
        self.compute_with(h, work_budget()).unwrap()
    }
    fn compute_with(&mut self, h: &m::Handle, b: m::Budgets) -> Result<m::Action, m::Error> {
        let state = m::inspect(&mut self.writer, h, 65536)?;
        let snapshot = self.writer.parser_snapshot(&self.head.token)?;
        let reader = self.writer.read_handle()?;
        m::compute(&snapshot, &reader, &state, b, &AtomicBool::new(false))
    }
    fn complete(&mut self, h: &m::Handle) -> SourceProjection {
        for _ in 0..4000 {
            let action = self.compute(h);
            if let m::Applied::Source(source) = m::apply(&mut self.writer, action).unwrap() {
                return source;
            }
        }
        panic!("source projection did not complete");
    }
    fn reopen(self) -> Self {
        let Self {
            dir,
            root,
            writer,
            head,
            source_binding,
        } = self;
        drop(writer);
        // Concurrent subprocess tests can briefly inherit CLOEXEC lock file
        // descriptions between fork and exec. Retry only our parent's PID;
        // an owned authority leak still fails at this short deadline.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
        let writer = loop {
            match Writer::open(&root) {
                Ok(writer) => break writer,
                Err(polylogue_store::Error::WriterHeld { ref holder, .. })
                    if holder == &format!("pid={}", std::process::id())
                        && std::time::Instant::now() < deadline =>
                {
                    std::thread::sleep(std::time::Duration::from_millis(1))
                }
                Err(error) => panic!("reopen after owned writer drop: {error}"),
            }
        };
        Self {
            dir,
            root,
            writer,
            head,
            source_binding,
        }
    }
}
#[test]
fn real_text_actions_restart_values_and_source_publication_keep_eof_pending() {
    let text = "large synthetic text \"\\\n".repeat(12000);
    let mut world = World::new(&text_input(&text));
    let mut handle = world.begin();
    let mut peer_committed = false;
    let mut restarts = 0;
    let mut appends = 0;
    let source: SourceProjection = loop {
        let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
        if restarts == 0
            && state
                .active_value()
                .is_some_and(|v| v.appended > 0 && v.appended < v.length)
        {
            drop(state);
            world = world.reopen();
            handle = world.begin();
            restarts += 1;
            continue;
        }
        let action = world.compute(&handle);
        let kind = action.operation();
        if kind == "value_append" {
            appends += 1;
        }
        if kind == "block_install" && !peer_committed {
            assert_eq!(restarts, 1);
            // Proofs retain archive authority, so create and consume this one
            // within this turn; no proof may span a deliberate writer restart.
            let proof = world
                .writer
                .blob_capture()
                .unwrap()
                .capture(&b"peer durable metadata"[..], &AtomicBool::new(false))
                .unwrap();
            let before = world.writer.status().unwrap().write_counter;
            world.writer.retain_captured_blob(proof).unwrap();
            assert!(world.writer.status().unwrap().write_counter > before);
            peer_committed = true;
        }
        if kind == "source_publish" && restarts == 1 {
            world = world.reopen();
            handle = world.begin();
            restarts += 1;
            continue;
        }
        match m::apply(&mut world.writer, action).unwrap() {
            m::Applied::Source(s) => break s,
            m::Applied::Progress => {}
            _ => panic!("unexpected apply"),
        }
    };
    assert_eq!(restarts, 2);
    assert!(appends > 2);
    assert!(peer_committed);
    assert!(
        world
            .writer
            .parser_head(world.head.token.input_pk, world.head.token.chain)
            .unwrap()
            .unwrap()
            .finalization_pending
    );
    assert!(
        world
            .writer
            .read_handle()
            .unwrap()
            .selected_message(source.message())
            .unwrap()
            .is_none()
    );
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let action = m::start(
        &snapshot,
        &world.request(),
        Some(source.clone()),
        budget(),
        65536,
    )
    .unwrap();
    drop(snapshot);
    assert_eq!(action.operation(), "existing");
    assert!(matches!(
        m::apply(&mut world.writer, action).unwrap(),
        m::Applied::Existing(_)
    ));
    assert_eq!(source.predecessor(), &world.head.token);
    world
        .writer
        .select_compatible_source_message_for_test(&source, 65536)
        .unwrap();
    let reader = world.writer.read_handle().unwrap();
    let view = reader.selected_message(source.message()).unwrap().unwrap();
    let text_block = view.block(BlockSlot::Item(0)).unwrap().unwrap();
    let chunk = text_block.text_range(0, 65536).unwrap().unwrap();
    assert_eq!(
        chunk.bytes.as_slice(),
        &text.as_bytes()[..chunk.bytes.len()]
    );
    match view
        .field(
            &Slot::Extra {
                block: None,
                key: "phase".into(),
            },
            65536,
        )
        .unwrap()
        .unwrap()
    {
        polylogue_store::contributions::FieldRead::Value(v) => {
            assert!(v.supplied);
            assert_eq!(v.range(0, 10).unwrap().bytes, b"null");
            let origin: Value =
                serde_json::from_slice(&v.origin_range(0, 65536).unwrap().unwrap().bytes).unwrap();
            assert_eq!(origin["format"], "codex-field-origin");
            assert_eq!(
                origin["binding"]["input"],
                polylogue_model::InputId::imported(&world.source_binding, "codex-rollout", &[])
                    .as_str()
            );
            assert_eq!(origin["selection"]["transform"], "JsonValueV1");
        }
        _ => panic!("known null became a conflict"),
    }
    match view
        .field(&Slot::Message(MessageField::At), 65536)
        .unwrap()
        .unwrap()
    {
        polylogue_store::contributions::FieldRead::Value(v) => {
            let typed: Value = serde_json::from_slice(&v.range(0, 1024).unwrap().bytes).unwrap();
            assert_eq!(typed["time_quality"], "intrinsic");
            assert!(typed["at"].is_i64());
        }
        _ => panic!("normalized source time became a conflict"),
    }
    assert!(
        matches!(text_block.field(&Slot::Block{block:BlockSlot::Item(0),field:BlockField::Text},65536).unwrap(),Some(polylogue_store::contributions::FieldRead::Value(v)) if v.supplied)
    );
}
#[test]
fn stale_turn_rolls_back_and_layout_cannot_change_after_install() {
    let mut world = World::new(&text_input("small"));
    let handle = world.begin();
    let first = world.compute(&handle);
    let stale = world.compute(&handle);
    assert_eq!(first.operation(), "source_range");
    m::apply(&mut world.writer, first).unwrap();
    assert!(matches!(
        m::apply(&mut world.writer, stale),
        Err(m::Error::Store(polylogue_store::Error::StaleMessage))
    ));
    for _ in 0..200 {
        let action = world.compute(&handle);
        let installed = action.operation() == "block_install";
        let duplicate = if action.operation() == "value_append" {
            Some(world.compute(&handle))
        } else {
            None
        };
        m::apply(&mut world.writer, action).unwrap();
        if let Some(stale) = duplicate {
            assert!(matches!(
                m::apply(&mut world.writer, stale),
                Err(m::Error::Store(polylogue_store::Error::StaleMessage))
            ));
        }
        if installed {
            let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
            assert!(
                world
                    .writer
                    .set_message_layout(
                        handle.build(),
                        MessageLayout::Flat,
                        &ProjectionStep {
                            expected_sequence: state.sequence(),
                            next_cursor: b"illegal layout"
                        },
                        65536
                    )
                    .is_err()
            );
            return;
        }
    }
    panic!("block never installed");
}
#[test]
fn tool_without_invocation_id_leaves_typed_pending_without_source_certificate() {
    let bytes = [
        line(
            "session_meta",
            json!({"id":"synthetic-session","cwd":"/synthetic"}),
        ),
        line(
            "response_item",
            json!({"type":"function_call","id":"call-record","name":"synthetic","arguments":"{}"}),
        ),
    ]
    .concat()
    .into_bytes();
    let mut world = World::new(&bytes);
    let handle = world.begin();
    for _ in 0..200 {
        let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
        let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
        let reader = world.writer.read_handle().unwrap();
        let result = m::compute(
            &snapshot,
            &reader,
            &state,
            work_budget(),
            &AtomicBool::new(false),
        );
        drop(reader);
        drop(snapshot);
        match result {
            Err(m::Error::Unsupported(_)) => {
                assert!(
                    world
                        .writer
                        .current_source_projection(&world.head.token, &[3; 32], 65536)
                        .unwrap()
                        .is_none()
                );
                assert!(!state.projection_ready());
                return;
            }
            Ok(a) => {
                m::apply(&mut world.writer, a).unwrap();
            }
            Err(e) => panic!("unexpected failure: {e}"),
        }
    }
    panic!("unsupported tool was incorrectly completed");
}

#[test]
fn textless_reasoning_preserves_unknown_encryption_without_empty_text() {
    let bytes = [
        line(
            "session_meta",
            json!({"id":"synthetic-session","cwd":"/synthetic"}),
        ),
        line(
            "response_item",
            json!({"type":"reasoning","id":"reasoning","summary":[]}),
        ),
    ]
    .concat()
    .into_bytes();
    let mut world = World::new(&bytes);
    let handle = world.begin();
    let mut source = None;
    for _ in 0..200 {
        let action = world.compute(&handle);
        match m::apply(&mut world.writer, action).unwrap() {
            m::Applied::Source(s) => {
                source = Some(s);
                break;
            }
            m::Applied::Progress => {}
            _ => panic!("unexpected apply"),
        }
    }
    let source = source.unwrap_or_else(|| panic!("reasoning source certificate missing"));
    world
        .writer
        .select_compatible_source_message_for_test(&source, 65536)
        .unwrap();
    let reader = world.writer.read_handle().unwrap();
    let view = reader.selected_message(source.message()).unwrap().unwrap();
    assert!(view.block(BlockSlot::Item(0)).unwrap().is_none());
    let thinking = view.block(BlockSlot::Reasoning(0)).unwrap().unwrap();
    assert_eq!(thinking.encrypted, None);
    assert!(thinking.text_range(0, 100).unwrap().is_none());
    assert!(
        thinking
            .field(
                &Slot::Block {
                    block: BlockSlot::Reasoning(0),
                    field: BlockField::Encrypted
                },
                65536
            )
            .unwrap()
            .is_none()
    );
}

#[test]
fn cancelled_and_unadmitted_compute_do_not_advance_the_durable_cursor() {
    let mut world = World::new(&text_input("small"));
    let handle = world.begin();
    let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let reader = world.writer.read_handle().unwrap();
    assert!(matches!(
        m::compute(
            &snapshot,
            &reader,
            &state,
            work_budget(),
            &AtomicBool::new(true)
        ),
        Err(m::Error::Cancelled)
    ));
    let mut b = work_budget();
    b.read.cells = 0;
    assert!(matches!(
        m::compute(&snapshot, &reader, &state, b, &AtomicBool::new(false)),
        Err(m::Error::Store(
            polylogue_store::Error::ParserAdmission { .. }
        ))
    ));
    drop(reader);
    drop(snapshot);
    assert_eq!(
        m::inspect(&mut world.writer, &handle, 65536)
            .unwrap()
            .sequence(),
        0
    );
}

#[test]
fn worker_actions_and_errors_are_transferable_without_sqlite_handles() {
    fn send<T: Send>() {}
    send::<m::Action>();
    send::<m::Handle>();
    send::<m::State>();
    send::<m::Error>();
}

#[test]
fn an_unhandled_origin_producer_cannot_ride_a_computed_field_or_restart() {
    let mut world = World::new(&text_input("synthetic source"));
    let handle = world.begin();
    let action = world.compute(&handle);
    let before = m::inspect(&mut world.writer, &handle, 65536)
        .unwrap()
        .sequence();
    let origin = world
        .writer
        .begin_message_origin_value(handle.build(), 1)
        .unwrap();
    assert!(matches!(
        m::apply(&mut world.writer, action),
        Err(m::Error::Store(polylogue_store::Error::StaleMessage))
    ));
    assert_eq!(
        m::inspect(&mut world.writer, &handle, 65536)
            .unwrap()
            .sequence(),
        before
    );
    let mut world = world.reopen();
    let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let reader = world.writer.read_handle().unwrap();
    assert!(matches!(
        m::compute(
            &snapshot,
            &reader,
            &state,
            work_budget(),
            &AtomicBool::new(false)
        ),
        Err(m::Error::Unsupported(_))
    ));
    drop(reader);
    drop(snapshot);
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &[3; 32], 65536)
            .unwrap()
            .is_none()
    );
    assert_eq!(world.writer.value_head(&origin.id).unwrap().appended, 0);
    world
        .writer
        .retire_message_preparation(handle.build())
        .unwrap();
}

#[test]
fn verified_origin_producer_restarts_and_selected_descriptor_outlives_parser_cells() {
    let bytes = text_input("synthetic original text");
    let mut world = World::new(&bytes);
    let mut handle = world.begin();
    let mut restarted_value = false;
    let mut restarted_origin = false;
    let mut b = work_budget();
    b.value_chunk = 7;
    let mut attempts = 0;
    let source = loop {
        attempts += 1;
        assert!(attempts < 4000, "verified source did not converge");
        let state = m::inspect(&mut world.writer, &handle, 65536).unwrap();
        let value_partial = state
            .active_value()
            .is_some_and(|v| v.appended > 0 && v.appended < v.length);
        let origin_partial = state
            .active_origin_value()
            .is_some_and(|v| v.appended > 0 && v.appended < v.length);
        if (!restarted_value && value_partial) || (!restarted_origin && origin_partial) {
            drop(state);
            world = world.reopen();
            handle = world.begin();
            if origin_partial {
                restarted_origin = true;
            } else {
                restarted_value = true;
            }
            continue;
        }
        let action = world.compute_with(&handle, b).unwrap();
        if let m::Applied::Source(source) = m::apply(&mut world.writer, action).unwrap() {
            break source;
        }
    };
    assert!(restarted_value && restarted_origin);
    world
        .writer
        .select_compatible_source_message_for_test(&source, 65536)
        .unwrap();
    let expected_input =
        polylogue_model::InputId::imported(&world.source_binding, "codex-rollout", &[]);
    let read_descriptor = |world: &World| {
        let reader = world.writer.read_handle().unwrap();
        let view = reader.selected_message(source.message()).unwrap().unwrap();
        let slot = Slot::Block {
            block: BlockSlot::Item(0),
            field: BlockField::Text,
        };
        let polylogue_store::contributions::FieldRead::Value(field) =
            view.field(&slot, 65536).unwrap().unwrap()
        else {
            panic!("field conflict")
        };
        // Range API reads one admitted chunk; assemble only this tiny synthetic
        // fixture descriptor, not a product whole-field path.
        let mut descriptor = Vec::new();
        let mut offset = 0;
        loop {
            let range = field.origin_range(offset, 65536).unwrap().unwrap();
            descriptor.extend_from_slice(&range.bytes);
            offset = range.next_offset;
            if range.eof {
                break;
            }
        }
        descriptor
    };
    let before = read_descriptor(&world);
    let bound: Value = serde_json::from_slice(&before).unwrap();
    assert_eq!(bound["binding"]["input"], expected_input.as_str());
    assert_eq!(bound["binding"]["chain"], world.head.token.chain);
    assert_eq!(bound["selection"]["transform"], "TextJoinLfV1");
    let selection: codex::selection::FactSelection =
        serde_json::from_value(bound["selection"].clone()).unwrap();
    assert_eq!(
        selection.inputs[0].path.last(),
        Some(&codex::selection::PathPart::Key("text".into()))
    );
    let mut recovered = Vec::new();
    selection
        .write_value(
            |range| -> Result<Vec<u8>, std::io::Error> {
                Ok(bytes[range.start as usize..range.end as usize].to_vec())
            },
            65536,
            &mut recovered,
        )
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<String>(&recovered).unwrap(),
        "synthetic original text"
    );
    // New parser interpretation generation must not rewrite a selected immutable
    // field's retained origin descriptor or require the original provider path.
    world
        .writer
        .begin_parser(&Basis {
            input_pk: world.head.token.input_pk,
            chain: world.head.token.chain,
            parser_version: codex::PARSER_VERSION,
            digest: [8; 32],
            acquired_upto: bytes.len() as u64,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert_eq!(read_descriptor(&world), before);
}
#[test]
fn wrong_source_binding_cannot_publish_a_finished_build_after_restart() {
    let mut world = World::new(&text_input("source binding"));
    let handle = world.begin();
    loop {
        let action = world.compute(&handle);
        if action.operation() == "source_publish" {
            break;
        }
        m::apply(&mut world.writer, action).unwrap();
    }
    world = world.reopen();
    let mut wrong = world.request();
    wrong.source_binding = polylogue_model::BlobHash::of(b"different retained input");
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let action = m::start(&snapshot, &wrong, None, budget(), 65536).unwrap();
    drop(snapshot);
    let m::Applied::Prepared(wrong_handle) = m::apply(&mut world.writer, action).unwrap() else {
        panic!("build")
    };
    assert!(matches!(
        world.compute_with(&wrong_handle, work_budget()),
        Err(m::Error::Invariant(_))
    ));
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &[3; 32], 65536)
            .unwrap()
            .is_none()
    );
    let handle = world.begin();
    let source = world.complete(&handle);
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    assert!(matches!(
        m::start(&snapshot, &wrong, Some(source), budget(), 65536),
        Err(m::Error::Invariant(_))
    ));
}
#[test]
fn source_selector_value_and_type_mismatches_never_publish_a_certificate() {
    let parsed = text_input("A");
    let changed = text_input("B");
    let numeric = String::from_utf8(parsed.clone())
        .unwrap()
        .replace("\"text\":\"A\"", "\"text\":123")
        .into_bytes();
    assert_eq!(numeric.len(), parsed.len());
    for retained in [&changed, &numeric] {
        let mut world = World::with_parsed(retained, &parsed);
        let handle = world.begin();
        let mut refused = false;
        for _ in 0..1000 {
            match world.compute_with(&handle, work_budget()) {
                Ok(action) => {
                    assert!(!matches!(
                        m::apply(&mut world.writer, action).unwrap(),
                        m::Applied::Source(_)
                    ));
                }
                Err(m::Error::Invariant(_) | m::Error::Selection(_)) => {
                    refused = true;
                    break;
                }
                Err(error) => panic!("unexpected error: {error}"),
            }
        }
        assert!(refused);
        assert!(
            world
                .writer
                .current_source_projection(&world.head.token, &[3; 32], 65536)
                .unwrap()
                .is_none()
        );
    }
    // Deliberately corrupt one typed source-owned selector in the disposable
    // fixture, keeping its origin/path contract internally coherent. The real
    // evaluator must reject its missing path, never attach unchecked JSON.
    let mut world = World::with_mutation(&parsed, &parsed, |delta| {
        let Some(Some(codex::keyed::Cell::Witness(w))) = delta
            .cells
            .get_mut(&codex::keyed::Key::Presence(0, "at".into()))
        else {
            panic!("timestamp witness")
        };
        let selected = &mut w.selection.as_mut().unwrap().inputs[0];
        selected.path = vec![codex::selection::PathPart::Key("missing".into())];
        w.origins = codex::origins::Origins::Located(selected.clone());
    });
    let handle = world.begin();
    let mut refused = false;
    for _ in 0..1000 {
        match world.compute_with(&handle, work_budget()) {
            Ok(action) => {
                m::apply(&mut world.writer, action).unwrap();
            }
            Err(m::Error::Selection(_)) => {
                refused = true;
                break;
            }
            Err(e) => panic!("{e}"),
        }
    }
    assert!(refused);
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &[3; 32], 65536)
            .unwrap()
            .is_none()
    );
}
#[test]
fn source_set_is_exact_noncontiguous_and_includes_timestamp_only_witness() {
    use codex::{
        keyed::{Cell, Key},
        origins::{Origins, RecordRange},
        selection::SelectedRaw,
    };
    let header = line("session_meta", json!({"id":"synthetic-session"}));
    let context = line("turn_context", json!({"model":"synthetic-model"}));
    let ignored = line("event_msg", json!({"type":"task_complete"}));
    let response = line(
        "response_item",
        json!({"type":"message","id":"response","role":"assistant","content":[{"text":"source"}]}),
    );
    let stamp = line("event_msg", json!({"type":"task_complete"}));
    let bytes = format!("{header}{context}{ignored}{response}{stamp}").into_bytes();
    let response_start = header.len() + context.len() + ignored.len();
    let stamp_start = response_start + response.len();
    let mut world = World::with_mutation(&bytes, &bytes, |delta| {
        // Synthetic valid witness placement exercises the owner seam: timestamps
        // are not digest members, and must not be dropped by an evidence-only scan.
        let Some(Some(Cell::Witness(w))) = delta.cells.get_mut(&Key::Presence(0, "at".into()))
        else {
            panic!("timestamp witness")
        };
        let selected = SelectedRaw {
            record: RecordRange {
                start: stamp_start as u64,
                end: (bytes.len() - 1) as u64,
            },
            path: vec![codex::selection::PathPart::Key("timestamp".into())],
        };
        w.selection.as_mut().unwrap().inputs[0] = selected.clone();
        w.origins = Origins::Located(selected);
    });
    let handle = world.begin();
    loop {
        let action = world.compute(&handle);
        let stop = action.operation() == "layout";
        m::apply(&mut world.writer, action).unwrap();
        if stop {
            break;
        }
    }
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let has = |start, end| {
        snapshot
            .has_message_source(
                handle.build(),
                polylogue_store::contributions::SourceRange { start, end },
            )
            .unwrap()
    };
    assert!(has(
        header.len() as u64,
        (header.len() + context.len() - 1) as u64
    ));
    assert!(has(response_start as u64, (stamp_start - 1) as u64));
    assert!(has(stamp_start as u64, (bytes.len() - 1) as u64));
    assert!(!has(
        (header.len() + context.len()) as u64,
        response_start as u64
    ));
    assert!(!has(header.len() as u64, bytes.len() as u64));
    drop(snapshot);
    world.complete(&handle);
}

#[path = "canonical_message/claude_tests.rs"]
mod claude_tests;
#[path = "canonical_message/codex_extras.rs"]
mod codex_extras;

// Append inside engine/tests/canonical_message.rs after the publication freeze.
// Diagnostic: demonstrates the currently explicit refusal, not desired support.
#[test]
fn parser_generation_replay_replaces_only_verified_prior_candidate_and_preserves_pinned_reads() {
    let bytes = text_input("retained source generation discriminator");
    let mut world = World::new(&bytes);
    let handle = world.begin();
    let previous = world.complete(&handle);
    world
        .writer
        .select_compatible_source_message_for_test(&previous, 65536)
        .unwrap();
    let old_generation = world.head.token.generation;
    let new = world
        .writer
        .begin_parser(&Basis {
            input_pk: world.head.token.input_pk,
            chain: world.head.token.chain,
            parser_version: codex::PARSER_VERSION,
            digest: [8; 32],
            acquired_upto: bytes.len() as u64,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    assert_ne!(new.token.generation, old_generation);
    let snapshot = world.writer.parser_snapshot(&new.token).unwrap();
    let attempt = adapter::Attempt::new(&snapshot, budget());
    let resume = attempt.resume().unwrap();
    let framed = drive_jsonl(KeyedRecords::new(&attempt, resume), &bytes, 0);
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    let encoded = adapter::encode_delta(&delta, 1, budget()).unwrap();
    assert!(attempt.take_failure().is_none());
    drop(attempt);
    drop(snapshot);
    let mut prepared = world
        .writer
        .prepare_parser(
            &new,
            Progress {
                inspected_upto: framed.parsed_upto,
                parsed_upto: framed.parsed_upto,
                phase: Phase::Finalizing,
                finalization_pending: true,
                counts: Accounting {
                    framed: framed.counts.framed,
                    used: framed.counts.used,
                    ignored: framed.counts.ignored.values().sum(),
                    unrecognized: framed.counts.unrecognized.len() as u64,
                    undecodable: framed.counts.undecodable.len() as u64,
                },
            },
        )
        .unwrap();
    prepared = world
        .writer
        .stage_parser(&prepared.token, &encoded.changes, budget())
        .unwrap();
    prepared = world.writer.seal_parser(&prepared.token).unwrap();
    while prepared.token.applied < prepared.token.staged {
        prepared = world
            .writer
            .apply_parser(&prepared.token, budget())
            .unwrap();
    }
    world.head = world.writer.publish_parser(&prepared.token).unwrap();
    assert!(
        world
            .writer
            .current_source_projection(&world.head.token, &[3; 32], 65536)
            .unwrap()
            .is_none()
    );
    world = world.reopen();
    let reader = world.writer.read_handle().unwrap();
    let pinned = reader
        .selected_message(previous.message())
        .unwrap()
        .unwrap();
    let handle = world.begin();
    let replacement = world.complete(&handle);
    assert_eq!(replacement.version_hash(), previous.version_hash());
    assert_ne!(replacement.body(), previous.body());
    assert_eq!(
        &reader
            .selected_message(previous.message())
            .unwrap()
            .unwrap()
            .body_id(),
        previous.body()
    );
    use polylogue_store::contributions::BodyRetirement;
    assert_eq!(
        world.writer.retire_message_body(previous.body()).unwrap(),
        BodyRetirement::Selected
    );
    world
        .writer
        .select_compatible_source_message_for_test(&replacement, 65536)
        .unwrap();
    assert_eq!(
        &reader
            .selected_message(previous.message())
            .unwrap()
            .unwrap()
            .body_id(),
        replacement.body()
    );
    assert_eq!(&pinned.body_id(), previous.body());
    assert_eq!(
        world.writer.retire_message_body(previous.body()).unwrap(),
        BodyRetirement::DetachedCandidate
    );
    assert_eq!(
        world.writer.retire_message_body(previous.body()).unwrap(),
        BodyRetirement::Body
    );
    assert_eq!(
        pinned
            .block(BlockSlot::Item(0))
            .unwrap()
            .unwrap()
            .text_range(0, 256)
            .unwrap()
            .unwrap()
            .bytes,
        b"retained source generation discriminator"
    );
    drop(pinned);
    drop(reader);
    world = world.reopen();
    let current = world
        .writer
        .current_source_projection(&world.head.token, &[3; 32], 65536)
        .unwrap()
        .unwrap();
    assert_eq!(current.body(), replacement.body());
    let snapshot = world.writer.parser_snapshot(&world.head.token).unwrap();
    let action = m::start(&snapshot, &world.request(), Some(current), budget(), 65536).unwrap();
    assert!(matches!(
        m::apply(&mut world.writer, action).unwrap(),
        m::Applied::Existing(_)
    ));
}

trait SelectCompatibleSourceForTest {
    fn select_compatible_source_message_for_test(
        &mut self,
        source: &polylogue_store::contributions::SourceProjection,
        allowance: usize,
    ) -> std::result::Result<polylogue_store::contributions::SourceSelection, polylogue_store::Error>;
}
impl SelectCompatibleSourceForTest for Writer {
    fn select_compatible_source_message_for_test(
        &mut self,
        source: &polylogue_store::contributions::SourceProjection,
        allowance: usize,
    ) -> std::result::Result<polylogue_store::contributions::SourceSelection, polylogue_store::Error>
    {
        use polylogue_store::contributions::{CompatibilityTurn, SourceSelection};
        let reader = self.read_handle()?;
        let mut work = self.begin_compatible_source_selection(source, allowance)?;
        for _ in 0..100000 {
            work = match work.advance(
                &reader,
                allowance,
                &std::sync::atomic::AtomicBool::new(false),
            )? {
                CompatibilityTurn::Progress(w) => w,
                CompatibilityTurn::Certified(w, c) => {
                    self.certify_body_compatibility(c)?;
                    w
                }
                CompatibilityTurn::Ready(w) => {
                    return self.install_compatible_source_selection(w, allowance);
                }
                CompatibilityTurn::Contested => return Ok(SourceSelection::Contested),
            };
        }
        Err(polylogue_store::Error::Invariant(
            "test selection failed to make bounded progress".into(),
        ))
    }
}
