mod ordered_extras;
mod visibility;
use super::*;
use crate::contributions::SelectCompatibleSourceForTest as _;
use crate::contributions::directory as d;
use crate::values;
use crate::{
    ImportObservation, InputFormat, Writer,
    contributions::{
        BlockKind, BlockPresence, Identity, ProjectionStep, Source, SourceProjection, SourceRange,
    },
    parser_state::{self, Accounting, Basis, Budget, Head, Phase, Progress},
};
use polylogue_model::{
    BlobHash,
    merge::{BlockField, FinishState, MessageField, RouteKind, Slot},
};
use polylogue_model::{BlockSlot, MessageId, Origin, Role, SessionId, SessionKind, WrittenBy};
use rusqlite::params;
fn budget() -> Budget {
    Budget {
        cells: 1000,
        encoded_bytes: 1 << 20,
    }
}
// Builds messages through the writer API from a synthetic input.
struct Fixture {
    _dir: tempfile::TempDir,
    writer: Writer,
    head: Head,
    session: SessionId,
    origin: Origin,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut writer = Writer::open(dir.path()).unwrap();
        let hash = writer
            .put_file_blob(&b"synthetic source bytes\n"[..])
            .unwrap();
        let input = writer
            .store_immutable_input(
                hash,
                InputFormat::CodexRollout,
                &[],
                &ImportObservation {
                    id: "synthetic-reader",
                    machine: "synthetic",
                    host: "synthetic",
                    path: "synthetic/source.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = writer
            .begin_parser(&Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: 1,
                digest: [1; 32],
                acquired_upto: input.length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let origin = Origin::new("synthetic").unwrap();
        let session = SessionId::new(&origin, "session:opaque").unwrap();
        let mut f = Self {
            _dir: dir,
            writer,
            head,
            session,
            origin,
        };
        f.advance();
        f
    }
    fn advance(&mut self) {
        let path = self.writer.root.join("index.db");
        self.head = parser_state::write_step(
            &mut self.writer._index,
            &path,
            &self.head,
            Progress {
                inspected_upto: self.head.acquired_upto,
                parsed_upto: self.head.acquired_upto,
                phase: Phase::Finalizing,
                finalization_pending: true,
                counts: Accounting::default(),
            },
            &[],
            budget(),
            |_| Ok(()),
        )
        .unwrap();
    }
    fn prepare(&mut self, name: &str, seq: u64, key: u8) -> crate::contributions::Preparation {
        let id = MessageId::new(self.session.clone(), name).unwrap();
        self.writer
            .prepare_message(
                Identity {
                    session: &self.session,
                    origin: &self.origin,
                    native_session: "session:opaque",
                    native_session_encoding: None,
                    session_kind: SessionKind::Main,
                    message: &id,
                    sequence: seq,
                    role: Role::Assistant,
                    written_by: WrittenBy::Model,
                },
                Source {
                    token: &self.head.token,
                    revision: 1,
                    evidence_revision: self.head.token.step as u64,
                    logical_key: [key; 32],
                    route: RouteKind::Native,
                },
                65536,
            )
            .unwrap()
    }
    fn ready(
        &mut self,
        name: &str,
        seq: u64,
        key: u8,
        model: &str,
        old_extra: bool,
        spans: &[(u64, u64)],
    ) -> SourceProjection {
        self.ready_with_block(
            name,
            seq,
            key,
            model,
            old_extra,
            spans,
            BlockKind::Thinking { encrypted: None },
        )
    }
    fn ready_with_block(
        &mut self,
        name: &str,
        seq: u64,
        key: u8,
        model: &str,
        old_extra: bool,
        spans: &[(u64, u64)],
        kind: BlockKind<'_>,
    ) -> SourceProjection {
        let thinking = matches!(kind, BlockKind::Thinking { .. });
        let previous = self
            .writer
            .current_source_projection(&self.head.token, &[key; 32], 65536)
            .unwrap();
        let build = self.prepare(name, seq, key).build;
        let mut sequence = 0;
        for &(start, end) in spans {
            self.writer
                .add_message_source(
                    &build,
                    SourceRange { start, end },
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"source",
                    },
                    65536,
                )
                .unwrap();
            sequence += 1;
        }
        self.json(&build, &serde_json::to_vec(model).unwrap());
        self.writer
            .write_message_field(
                &build,
                &Slot::Message(MessageField::Model),
                None,
                true,
                &ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"model",
                },
                65536,
            )
            .unwrap();
        sequence += 1;
        if old_extra {
            self.json(&build, b"false");
            self.writer
                .write_message_field(
                    &build,
                    &Slot::Extra {
                        block: None,
                        key: "old:extra".into(),
                    },
                    None,
                    true,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"extra",
                    },
                    65536,
                )
                .unwrap();
            sequence += 1;
        }
        self.writer
            .begin_block(
                &build,
                0,
                0,
                kind,
                BlockPresence {
                    source_block: true,
                    source_text: false,
                },
                &ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"block",
                },
                65536,
            )
            .unwrap();
        sequence += 1;
        if thinking {
            self.json(&build, b"\"stored-signature\"");
            self.writer
                .write_block_field(
                    &build,
                    &Slot::Block {
                        block: BlockSlot::Item(0),
                        field: BlockField::ThinkingSignature,
                    },
                    None,
                    false,
                    &ProjectionStep {
                        expected_sequence: sequence,
                        next_cursor: b"signature",
                    },
                    65536,
                )
                .unwrap();
            sequence += 1;
        }
        self.writer
            .install_block(
                &build,
                &ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"installed",
                },
                65536,
            )
            .unwrap();
        sequence += 1;
        let prepared = self
            .writer
            .finish_message_projection(
                &build,
                &ProjectionStep {
                    expected_sequence: sequence,
                    next_cursor: b"finished",
                },
                BlobHash::from_bytes([key.wrapping_add(self.head.token.step as u8); 32]),
                FinishState::Finished,
                65536,
            )
            .unwrap();
        let source = self
            .writer
            .publish_source_projection(
                &prepared,
                previous.as_ref(),
                b"synthetic-reader-proof",
                65536,
            )
            .unwrap();
        self.writer
            .select_compatible_source_message_for_test(&source, 65536)
            .unwrap();
        source
    }
    fn json(&mut self, build: &crate::contributions::Build, bytes: &[u8]) {
        let value = self
            .writer
            .begin_message_value(build, values::Codec::Json, bytes.len() as u64)
            .unwrap();
        self.writer
            .append_value_chunk(&value.id, 0, bytes, bytes.len())
            .unwrap();
        let reader = self.writer.read_handle().unwrap();
        let mut proof = reader.validate_value(&value.id).unwrap();
        while !proof.progress().complete {
            proof.advance(&reader, 65536).unwrap();
        }
        drop(reader);
        self.writer.seal_message_value(build, &proof).unwrap();
    }
    fn session(&self) -> SessionView {
        self.writer
            .read_handle()
            .unwrap()
            .selected_session(&self.session, 65536)
            .unwrap()
            .unwrap()
    }
}
fn model(view: &MessageView) -> String {
    let FieldRead::Value(field) = view
        .field(&Slot::Message(MessageField::Model), 65536)
        .unwrap()
        .unwrap()
    else {
        panic!("conflict")
    };
    serde_json::from_slice(&field.range(0, 65536).unwrap().bytes).unwrap()
}
#[test]
fn dependencies_include_unregistered_inputs_and_stay_pinned() {
    let mut f = Fixture::new();
    f.ready("catalog", 0, 1, "model", false, &[(0, 1)]);
    let old = f.session();
    let old_archive_counter = old.source_archive_counter();
    let hash = f.writer.put_file_blob(b"{malformed}\n").unwrap();
    let input = f
        .writer
        .store_immutable_input(
            hash,
            InputFormat::ClaudeCodeJsonl,
            &["claude-main", "context-only"],
            &ImportObservation {
                id: "catalog-gap",
                machine: "synthetic",
                host: "synthetic",
                path: "synthetic/context.jsonl",
                seen_at: 0,
            },
            65536,
        )
        .unwrap();
    assert!(old.next_dependency(None, 1).is_err());
    let (old_cursor, old_input) = old.next_dependency(None, 65536).unwrap().unwrap();
    assert_ne!(old_input.id(), input.id);
    assert!(
        old.next_dependency(Some(&old_cursor), 65536)
            .unwrap()
            .is_none()
    );
    // An input stored in the archive is listed even before the index registers
    // it or any message comes from it.
    f.writer
        ._index
        .execute("DELETE FROM input_ref WHERE pk=?1", [input.input_pk])
        .unwrap();
    let fresh = f.session();
    assert!(fresh.source_archive_counter() > old_archive_counter);
    assert!(matches!(
        fresh.next_dependency(Some(&old_cursor), 65536),
        Err(Error::StaleMessage)
    ));
    let mut after = None;
    let mut found = false;
    while let Some((cursor, dependency)) = fresh.next_dependency(after.as_ref(), 65536).unwrap() {
        if dependency.id() == input.id {
            assert_eq!(dependency.context(), ["claude-main", "context-only"]);
            assert!(dependency.registered().is_none());
            assert!(dependency.head().is_none());
            found = true;
        }
        after = Some(cursor);
    }
    assert!(found);
    // Change the stored context: a new catalog refuses the mismatch, while the
    // already open one keeps the original.
    f.writer
        .archive
        .execute(
            "UPDATE input SET context_json='[\"claude-main\",\"changed\"]' WHERE id=?1",
            [&input.id],
        )
        .unwrap();
    let newer = f.session();
    let mut after = None;
    let mut refused = false;
    loop {
        match newer.next_dependency(after.as_ref(), 65536) {
            Ok(Some((cursor, _))) => after = Some(cursor),
            Ok(None) => break,
            Err(Error::Invariant(_)) => {
                refused = true;
                break;
            }
            Err(e) => panic!("unexpected: {e}"),
        }
    }
    assert!(refused);
    let mut after = None;
    let mut original = false;
    while let Some((cursor, dependency)) = fresh.next_dependency(after.as_ref(), 65536).unwrap() {
        if dependency.id() == input.id {
            assert_eq!(dependency.context(), ["claude-main", "context-only"]);
            original = true;
        }
        after = Some(cursor);
    }
    assert!(original);
}
#[test]
fn session_view_is_pinned_and_reports_pending_messages() {
    fn assert_send<T: Send>() {}
    assert_send::<MessageView>();
    assert_send::<SessionView>();
    assert_send::<SessionCursor>();
    assert_send::<FieldCursor>();
    assert_send::<SourceCursor>();
    let mut f = Fixture::new();
    let a = f.ready("a:opaque", 0, 1, "old-a", false, &[(0, 1)]);
    let b = f.ready("b:opaque", 1, 2, "old-b", false, &[(2, 3)]);
    let _pending = f.prepare("pending:opaque", 2, 3);
    let view = f.session();
    let counter = view.position().counter;
    assert_eq!(view.header().unwrap().id, f.session);
    assert_eq!(view.header().unwrap().native_id, "session:opaque");
    let SessionStep::Ready {
        cursor,
        sequence,
        message: first,
    } = view.next(None, 65536).unwrap()
    else {
        panic!("first")
    };
    assert_eq!(sequence, 0);
    assert_eq!(first.id(), a.message());
    assert_eq!(first.header().unwrap().in_progress, Some(false));
    assert_eq!(model(&first), "old-a");
    f.advance();
    f.ready("b:opaque", 1, 2, "new-b", false, &[(2, 3)]);
    let SessionStep::Ready {
        cursor: second_cursor,
        message: second,
        ..
    } = view.next(Some(&cursor), 65536).unwrap()
    else {
        panic!("second")
    };
    assert_eq!(second.id(), b.message());
    assert_eq!(model(&second), "old-b");
    assert_eq!(
        model(
            &f.writer
                .read_handle()
                .unwrap()
                .selected_message(b.message())
                .unwrap()
                .unwrap()
        ),
        "new-b"
    );
    assert_eq!(view.position().counter, counter);
    assert!(f.session().position().counter > counter);
    let SessionStep::Pending {
        cursor,
        id,
        sequence,
    } = view.next(Some(&second_cursor), 65536).unwrap()
    else {
        panic!("pending member")
    };
    assert_eq!(sequence, 2);
    assert_eq!(
        id,
        MessageId::new(f.session.clone(), "pending:opaque").unwrap()
    );
    assert!(matches!(
        view.next(Some(&cursor), 65536).unwrap(),
        SessionStep::End {
            coverage: SessionCoverage::KnownCatalogOnly
        }
    ));
    let other = f.session();
    assert!(matches!(
        other.next(Some(&cursor), 65536),
        Err(Error::StaleMessage)
    ));
}
#[test]
fn duplicate_sequence_reports_unresolved_order() {
    let mut f = Fixture::new();
    f.ready("a", 4, 1, "a", false, &[]);
    f.prepare("b", 4, 2);
    assert!(matches!(
        f.session().next(None, 65536).unwrap(),
        SessionStep::UnresolvedOrder { sequence: 4 }
    ));
}
#[test]
fn oversized_session_id_is_refused_before_lookup() {
    let f = Fixture::new();
    let reader = f.writer.read_handle().unwrap();
    let catalog = reader.session_catalog(65536).unwrap();
    let boundary = SessionId::from_stored("x".repeat(128)).unwrap();
    let oversized = SessionId::from_stored("x".repeat(129)).unwrap();
    assert!(catalog.session(&boundary, 128).unwrap().is_none());
    for result in [
        catalog.session(&oversized, 128),
        reader.selected_session(&oversized, 128),
    ] {
        assert!(matches!(
            result,
            Err(Error::ParserAdmission {
                resource: "session read identity",
                required: 129
            })
        ));
    }
}
#[test]
fn session_reads_report_absence_size_limits_and_hiding() {
    let mut f = Fixture::new();
    let source = f.ready("a", 0, 1, "old", false, &[]);
    let reader = f.writer.read_handle().unwrap();
    let missing = SessionId::from_stored("opaque:absent:session").unwrap();
    assert!(reader.selected_session(&missing, 65536).unwrap().is_none());
    assert!(matches!(
        reader.selected_session(&f.session, 1),
        Err(Error::ParserAdmission { .. })
    ));
    drop(reader);
    let view = f.session();
    let SessionStep::Ready { message, .. } = view.next(None, 65536).unwrap() else {
        panic!("ready")
    };
    f.writer
        ._index
        .execute(
            "UPDATE message SET hidden=1,removal_epoch=removal_epoch+1 WHERE id=?1",
            [source.message().as_str()],
        )
        .unwrap();
    assert!(matches!(message.header(), Err(Error::StaleMessage)));
    assert!(matches!(
        message.read_certificate(),
        Err(Error::StaleMessage)
    ));
    assert!(matches!(view.next(None, 65536), Err(Error::StaleMessage)));
    f.writer
        ._index
        .execute("UPDATE session SET title=?1", ["large".repeat(1000)])
        .unwrap();
    assert!(matches!(
        f.writer
            .read_handle()
            .unwrap()
            .selected_session(&f.session, 256),
        Err(Error::ParserAdmission { .. })
    ));
}
#[test]
fn next_field_follows_selected_root_and_includes_unranked_fields() {
    let mut f = Fixture::new();
    let first = f.ready("a", 0, 1, "old", true, &[]);
    let old = f
        .writer
        .read_handle()
        .unwrap()
        .selected_message(first.message())
        .unwrap()
        .unwrap();
    let old_entry = old.next_field(None, 65536).unwrap().unwrap();
    let mut count = 1;
    let mut cursor = old_entry.cursor;
    while let Some(entry) = old.next_field(Some(&cursor), 65536).unwrap() {
        count += 1;
        cursor = entry.cursor
    }
    assert_eq!(count, 2);
    let block = old.block_at(0).unwrap().unwrap();
    let auxiliary = block.next_field(None, 65536).unwrap().unwrap();
    assert!(matches!(
        auxiliary.address,
        BlockFieldAddress::Field(BlockField::ThinkingSignature)
    ));
    let FieldRead::Value(value) = auxiliary.value else {
        panic!("auxiliary")
    };
    assert!(!value.supplied);
    assert_eq!(
        value.range(0, 65536).unwrap().bytes,
        b"\"stored-signature\""
    );
    assert!(
        block
            .next_field(Some(&auxiliary.cursor), 65536)
            .unwrap()
            .is_none()
    );
    f.advance();
    let second = f.ready("a", 0, 1, "new", false, &[]);
    let new = f
        .writer
        .read_handle()
        .unwrap()
        .selected_message(second.message())
        .unwrap()
        .unwrap();
    let entry = new.next_field(None, 65536).unwrap().unwrap();
    assert!(matches!(entry.slot, Slot::Message(MessageField::Model)));
    assert!(
        new.next_field(Some(&entry.cursor), 65536)
            .unwrap()
            .is_none()
    );
    assert!(matches!(
        new.next_field(Some(&cursor), 65536),
        Err(Error::StaleMessage)
    ));
    assert_eq!(model(&old), "old");
    assert_eq!(model(&new), "new");
    assert_ne!(
        old.read_certificate().unwrap(),
        new.read_certificate().unwrap()
    );
    assert!(matches!(
        new.next_field(None, 1),
        Err(Error::ParserAdmission { .. })
    ));
}
#[test]
fn source_spans_are_deduplicated_and_preferred_first() {
    let mut f = Fixture::new();
    let first = f.ready("a", 0, 1, "old", false, &[(0, 1), (4, 5)]);
    let old = f
        .writer
        .read_handle()
        .unwrap()
        .selected_message(first.message())
        .unwrap()
        .unwrap();
    f.advance();
    let second = f.ready("a", 0, 1, "new", false, &[(0, 1), (2, 3)]);
    let new = f
        .writer
        .read_handle()
        .unwrap()
        .selected_message(second.message())
        .unwrap()
        .unwrap();
    assert!(matches!(
        new.next_source(None, 1),
        Err(Error::ParserAdmission { .. })
    ));
    fn collect(view: &MessageView) -> Vec<(u64, u64, bool, String)> {
        let mut cursor = None;
        let mut spans = Vec::new();
        for _ in 0..100 {
            match view.next_source(cursor.as_ref(), 65536).unwrap() {
                SourceStep::Continue { cursor: next } => cursor = Some(next),
                SourceStep::Span {
                    cursor: next,
                    source,
                } => {
                    spans.push((source.start, source.end, source.preferred, source.input));
                    cursor = Some(next)
                }
                SourceStep::End => return spans,
            }
        }
        panic!("source traversal did not end")
    }
    let spans = collect(&new);
    assert_eq!(
        spans.iter().map(|s| (s.0, s.1, s.2)).collect::<Vec<_>>(),
        vec![(0, 1, true), (2, 3, true), (4, 5, false)]
    );
    assert!(spans.iter().all(|s| s.3 == spans[0].3));
    let old_spans = collect(&old);
    assert_eq!(
        old_spans
            .iter()
            .map(|s| (s.0, s.1, s.2))
            .collect::<Vec<_>>(),
        vec![(0, 1, true), (4, 5, true)]
    );
    let SourceStep::Span { cursor, .. } = old.next_source(None, 65536).unwrap() else {
        panic!("preferred")
    };
    assert!(matches!(
        new.next_source(Some(&cursor), 65536),
        Err(Error::StaleMessage)
    ));
}
#[test]
fn tool_result_keeps_unknown_error_and_zero_values() {
    use polylogue_model::{ErrorReason, ToolOutcome, UnknownReason};
    let mut f = Fixture::new();
    for (sequence, outcome) in [
        ToolOutcome::Unknown {
            reason: UnknownReason::OutputNotRecorded,
        },
        ToolOutcome::Error {
            reason: Some(ErrorReason::Denied("permission".into())),
        },
    ]
    .iter()
    .enumerate()
    {
        let source = f.ready_with_block(
            &format!("result-{sequence}"),
            sequence as u64,
            sequence as u8 + 1,
            "model",
            false,
            &[],
            BlockKind::ToolResult {
                outcome,
                exit_code: Some(0),
                duration_ms: Some(0),
            },
        );
        let view = f
            .writer
            .read_handle()
            .unwrap()
            .selected_message(source.message())
            .unwrap()
            .unwrap();
        let block = view.block_at(0).unwrap().unwrap();
        assert!(block.next_field(None, 65536).unwrap().is_none());
        let header = block.tool_result(65536).unwrap().unwrap();
        assert_eq!(&header.outcome, outcome);
        assert_eq!(header.exit_code, Some(0));
        assert_eq!(header.duration_ms, Some(0));
        assert!(matches!(
            block.tool_result(1),
            Err(Error::ParserAdmission { .. })
        ));
    }
    let source = f.ready("thinking", 2, 3, "model", false, &[]);
    let view = f
        .writer
        .read_handle()
        .unwrap()
        .selected_message(source.message())
        .unwrap()
        .unwrap();
    assert!(
        view.block_at(0)
            .unwrap()
            .unwrap()
            .tool_result(0)
            .unwrap()
            .is_none()
    );
}
