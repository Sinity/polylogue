use super::super::writer::{Identity, Source};
use super::*;
use crate::{ImportObservation, InputFormat};
use polylogue_model::{
    MessageId, Origin, Role, SessionId, SessionKind, WrittenBy, merge::RouteKind,
};
use std::path::PathBuf;
struct Fixture {
    dir: tempfile::TempDir,
    root: PathBuf,
    writer: Writer,
    head: parser_state::Head,
    build: Build,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut writer = Writer::open(&root).unwrap();
        let blob = writer
            .put_file_blob(b"{\"synthetic\":true}\n".repeat(8).as_slice())
            .unwrap();
        let input = writer
            .store_immutable_input(
                blob,
                InputFormat::CodexRollout,
                &[],
                &ImportObservation {
                    id: "synthetic-range",
                    machine: "synthetic",
                    host: "synthetic",
                    path: "synthetic.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = writer
            .begin_parser(&parser_state::Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: 1,
                digest: [7; 32],
                acquired_upto: input.length,
                companions_json: None,
                excisions_seen: 0,
            })
            .unwrap();
        let path = root.join("index.db");
        // The tests use arbitrary byte ranges, not record boundaries.
        let head = parser_state::write_step(
            &mut writer._index,
            &path,
            &head,
            parser_state::Progress {
                inspected_upto: 80,
                parsed_upto: 80,
                phase: parser_state::Phase::Finalizing,
                finalization_pending: true,
                counts: Default::default(),
            },
            &[],
            parser_state::Budget {
                cells: 1,
                encoded_bytes: 4096,
            },
            |_| Ok(()),
        )
        .unwrap();
        let origin = Origin::new("synthetic").unwrap();
        let session = SessionId::new(&origin, "session").unwrap();
        let message = MessageId::new(session.clone(), "message").unwrap();
        let build = writer
            .prepare_message(
                Identity {
                    session: &session,
                    origin: &origin,
                    native_session: "session",
                    native_session_encoding: None,
                    session_kind: SessionKind::Main,
                    message: &message,
                    sequence: 0,
                    role: Role::Assistant,
                    written_by: WrittenBy::Model,
                },
                Source {
                    predecessor: &head.token,
                    revision: 1,
                    evidence_revision: 1,
                    logical_key: [3; 32],
                    route: RouteKind::Native,
                },
                4096,
            )
            .unwrap()
            .build;
        Self {
            dir,
            root,
            writer,
            head,
            build,
        }
    }
    fn add(&mut self, range: SourceRange, sequence: u64) -> Result<()> {
        self.writer.add_message_source(
            &self.build,
            range,
            &ProjectionStep {
                expected_sequence: sequence,
                next_cursor: b"source",
            },
            4096,
        )
    }
    fn count(&self) -> i64 {
        self.writer
            ._index
            .query_row("SELECT count(*) FROM message_source", [], |r| r.get(0))
            .unwrap()
    }
    fn reopen(self) -> Self {
        let Self {
            dir,
            root,
            writer,
            head,
            build,
        } = self;
        drop(writer);
        let writer = crate::tests::reopen_after_drop(&root);
        Self {
            dir,
            root,
            writer,
            head,
            build,
        }
    }
}
#[test]
fn ranges_are_deduplicated_and_survive_reopen() {
    let mut f = Fixture::new();
    let a = SourceRange { start: 0, end: 10 };
    let b = SourceRange { start: 20, end: 30 };
    let old = f.writer.parser_snapshot(&f.head.token).unwrap();
    assert!(!old.has_message_source(&f.build, a).unwrap());
    f.add(a, 0).unwrap();
    f.add(b, 1).unwrap();
    f.add(a, 2).unwrap();
    assert_eq!(f.count(), 2);
    assert!(!old.has_message_source(&f.build, a).unwrap());
    let now = f.writer.parser_snapshot(&f.head.token).unwrap();
    assert!(now.has_message_source(&f.build, a).unwrap());
    assert!(now.has_message_source(&f.build, b).unwrap());
    assert!(
        !now.has_message_source(&f.build, SourceRange { start: 0, end: 30 })
            .unwrap()
    );
    drop(now);
    drop(old);
    let f = f.reopen();
    let view = f.writer.parser_snapshot(&f.head.token).unwrap();
    assert!(view.has_message_source(&f.build, a).unwrap());
    assert_eq!(
        f.writer
            .message_projection(&f.build, 4096)
            .unwrap()
            .sequence,
        3
    );
}
#[test]
fn failed_add_rolls_back_range_and_cursor() {
    let mut f = Fixture::new();
    let a = SourceRange { start: 1, end: 9 };
    assert!(matches!(f.add(a, 1), Err(Error::StaleMessage)));
    assert_eq!(f.count(), 0);
    f.writer._index.execute_batch("CREATE TRIGGER refuse_source_cursor BEFORE INSERT ON parser_cell WHEN NEW.namespace='message.projection.v1' BEGIN SELECT RAISE(ABORT,'synthetic late cursor failure'); END;").unwrap();
    assert!(matches!(f.add(a, 0), Err(Error::Sqlite { .. })));
    assert_eq!(f.count(), 0);
    assert_eq!(
        f.writer
            .message_projection(&f.build, 4096)
            .unwrap()
            .sequence,
        0
    );
    f.writer
        ._index
        .execute_batch("DROP TRIGGER refuse_source_cursor")
        .unwrap();
    f.add(a, 0).unwrap();
    assert!(matches!(f.add(a, 0), Err(Error::StaleMessage)));
    assert_eq!(f.count(), 1);
    assert!(matches!(
        f.writer.add_message_source(
            &f.build,
            a,
            &ProjectionStep {
                expected_sequence: 1,
                next_cursor: b"source"
            },
            1
        ),
        Err(Error::ParserAdmission { .. })
    ));
    assert_eq!(
        f.writer
            .message_projection(&f.build, 4096)
            .unwrap()
            .sequence,
        1
    );
}
#[test]
fn add_rejects_bad_range_stale_build_and_hidden_or_retired_message() {
    let mut f = Fixture::new();
    let valid = SourceRange { start: 0, end: 1 };
    for range in [
        SourceRange { start: 0, end: 0 },
        SourceRange { start: 2, end: 1 },
        SourceRange { start: 0, end: 81 },
        SourceRange {
            start: u64::MAX - 1,
            end: u64::MAX,
        },
    ] {
        assert!(f.add(range, 0).is_err());
    }
    let mut wrong = f.build.clone();
    wrong.source.basis = [8; 32];
    assert!(
        f.writer
            .add_message_source(
                &wrong,
                valid,
                &ProjectionStep {
                    expected_sequence: 0,
                    next_cursor: b"x"
                },
                4096
            )
            .is_err()
    );
    let view = f.writer.parser_snapshot(&f.head.token).unwrap();
    assert!(view.has_message_source(&wrong, valid).is_err());
    drop(view);
    f.writer
        ._index
        .execute("UPDATE message SET hidden=1", [])
        .unwrap();
    assert!(f.add(valid, 0).is_err());
    f.writer
        ._index
        .execute("UPDATE message SET hidden=0", [])
        .unwrap();
    f.writer.retire_message_preparation(&f.build).unwrap();
    assert!(f.add(valid, 0).is_err());
    assert_eq!(f.count(), 0);
}
#[test]
fn add_waits_for_open_value_origin_or_block() {
    use super::super::blocks::{BlockKind, BlockPresence};
    for choice in 0..3 {
        let mut f = Fixture::new();
        match choice {
            0 => {
                f.writer
                    .begin_message_value(&f.build, crate::values::Codec::Utf8, 0)
                    .unwrap();
            }
            1 => {
                f.writer.begin_message_origin_value(&f.build, 0).unwrap();
            }
            _ => {
                f.writer
                    .begin_block(
                        &f.build,
                        0,
                        0,
                        BlockKind::Thinking { encrypted: None },
                        BlockPresence::default(),
                        &ProjectionStep {
                            expected_sequence: 0,
                            next_cursor: b"block",
                        },
                        4096,
                    )
                    .unwrap();
            }
        }
        assert!(matches!(
            f.add(
                SourceRange { start: 0, end: 1 },
                if choice == 2 { 1 } else { 0 }
            ),
            Err(Error::MessagePreparationPending)
        ));
        assert_eq!(f.count(), 0);
    }
}
#[test]
fn new_parser_step_keeps_old_snapshot_and_rejects_old_build() {
    let mut f = Fixture::new();
    let range = SourceRange { start: 0, end: 10 };
    f.add(range, 0).unwrap();
    let old = f.writer.parser_snapshot(&f.head.token).unwrap();
    let path = f.root.join("index.db");
    let next = parser_state::write_step(
        &mut f.writer._index,
        &path,
        &f.head,
        parser_state::Progress {
            inspected_upto: 90,
            parsed_upto: 90,
            phase: parser_state::Phase::Finalizing,
            finalization_pending: true,
            counts: Default::default(),
        },
        &[],
        parser_state::Budget {
            cells: 1,
            encoded_bytes: 4096,
        },
        |_| Ok(()),
    )
    .unwrap();
    assert!(old.has_message_source(&f.build, range).unwrap());
    assert!(matches!(
        f.add(SourceRange { start: 30, end: 40 }, 1),
        Err(Error::StaleMessage)
    ));
    let new = f.writer.parser_snapshot(&next.token).unwrap();
    assert!(matches!(
        new.has_message_source(&f.build, range),
        Err(Error::StaleMessage)
    ));
    assert_eq!(f.count(), 1);
}
#[test]
fn input_id_is_size_limited_checked_and_snapshot_pinned() {
    let f = Fixture::new();
    let hash = polylogue_model::BlobHash::of(b"{\"synthetic\":true}\n".repeat(8).as_slice());
    let expected = polylogue_model::InputId::imported(&hash, "codex-rollout", &[]);
    let view = f.writer.parser_snapshot(&f.head.token).unwrap();
    assert_eq!(
        view.message_source_input_id(&f.build, expected.as_str().len())
            .unwrap(),
        expected.as_str()
    );
    assert!(
        matches!(view.message_source_input_id(&f.build,expected.as_str().len()-1),Err(Error::ParserAdmission {required,..}) if required == expected.as_str().len() as u64)
    );
    let mut wrong = f.build.clone();
    wrong.source.basis = [9; 32];
    assert!(matches!(
        view.message_source_input_id(&wrong, 4096),
        Err(Error::StaleMessage)
    ));
    // Change the index row: the old snapshot keeps the old id, and the lookup
    // does not validate ids.
    f.writer
        ._index
        .execute(
            "UPDATE input_ref SET id=?1 WHERE pk=?2",
            params!["synthetic-index-replacement", f.head.token.input_pk],
        )
        .unwrap();
    assert_eq!(
        view.message_source_input_id(&f.build, 4096).unwrap(),
        expected.as_str()
    );
    let fresh = f.writer.parser_snapshot(&f.head.token).unwrap();
    assert_eq!(
        fresh.message_source_input_id(&f.build, 4096).unwrap(),
        "synthetic-index-replacement"
    );
}
