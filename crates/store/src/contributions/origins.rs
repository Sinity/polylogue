//! Exact source-range SET membership, separate from origin descriptor semantics.
//! Numeric containment does not establish record boundaries, selected evidence,
//! or the meaning of a selector. Those remain source-mapper obligations.
use super::{
    blocks::{self, ProjectionStep},
    writer::{self, Build},
};
use crate::{Error, Result, Writer, db, parser_state};
use rusqlite::params;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceRange {
    pub start: u64,
    pub end: u64,
}
fn checked(range: SourceRange, parsed_upto: u64) -> Result<(i64, i64)> {
    if range.start >= range.end || range.end > parsed_upto {
        return Err(Error::Invariant(
            "message source range is outside the committed parsed prefix".into(),
        ));
    }
    Ok((writer::integer(range.start)?, writer::integer(range.end)?))
}
impl Writer {
    /// Add one exact locator to this candidate's source SET and advance its
    /// scalar cursor atomically. Input/chain come only from the pinned build.
    pub fn add_message_source(
        &mut self,
        build: &Build,
        range: SourceRange,
        step: &ProjectionStep<'_>,
        allowance: usize,
    ) -> Result<()> {
        let remaining = allowance.checked_sub(16).ok_or(Error::ParserAdmission {
            resource: "message source metadata bytes",
            required: 16,
        })?;
        if let Err(error) = blocks::admit(step, remaining) {
            return Err(match error {
                Error::ParserAdmission { required, .. } => Error::ParserAdmission {
                    resource: "message source metadata bytes",
                    required: required.checked_add(16).ok_or(Error::Range)?,
                },
                other => other,
            });
        }
        let path = self.root.join("index.db");
        let tx = db(&path, || self._index.transaction())?;
        writer::check(&tx, &path, build)?;
        let head = writer::source_head(&tx, &path, build.predecessor())?;
        let (start, end) = checked(range, head.parsed_upto)?;
        let pending: bool = db(&path, || {
            tx.query_row("SELECT active_value_pk IS NOT NULL OR active_origin_value_pk IS NOT NULL OR active_block_pk IS NOT NULL FROM message_build WHERE pk=?1",[build.pk],|r|r.get(0))
        })?;
        if pending {
            return Err(Error::MessagePreparationPending);
        }
        db(&path, || {
            tx.execute("INSERT INTO message_source(candidate_pk,message_pk,session_pk,input_pk,chain,source_start,source_end) VALUES(?1,?2,?3,?4,?5,?6,?7) ON CONFLICT(candidate_pk,input_pk,chain,source_start,source_end) DO NOTHING",params![build.candidate,build.message,build.session,build.source.input_pk,build.source.chain,start,end])
        })?;
        blocks::commit_cursor(&tx, &path, build, step)?;
        db(&path, || tx.commit())?;
        Ok(())
    }
}
impl parser_state::Snapshot {
    /// Bound the returned identity before allocating it, from this same pinned
    /// view. The caller compares it with its immutable raw-input recipe; this
    /// lookup alone does not certify a caller-supplied hash or descriptor.
    pub fn message_source_input_id(&self, build: &Build, allowance: usize) -> Result<String> {
        if self.head.token != *build.predecessor() {
            return Err(Error::MessagePredecessorChanged);
        }
        writer::check_completed(&self.db, &self.path, build)?;
        let allowance = writer::integer(u64::try_from(allowance).map_err(|_| Error::Range)?)?;
        let (required, value): (i64, Option<String>) = db(&self.path, || {
            self.db.query_row("SELECT length(CAST(id AS BLOB)),CASE WHEN length(CAST(id AS BLOB))<=?2 THEN id ELSE NULL END FROM input_ref WHERE pk=?1",params![build.source.input_pk,allowance],|r|Ok((r.get(0)?,r.get(1)?)))
        })?;
        let required = u64::try_from(required).map_err(|_| Error::Range)?;
        value.ok_or(Error::ParserAdmission {
            resource: "message source input identity bytes",
            required,
        })
    }
    /// Membership in THIS pinned provider snapshot, never a second connection.
    /// A current writer must still validate its own basis before applying work
    /// derived from this historical view.
    pub fn has_message_source(&self, build: &Build, range: SourceRange) -> Result<bool> {
        if self.head.token != *build.predecessor() {
            return Err(Error::MessagePredecessorChanged);
        }
        writer::check(&self.db, &self.path, build)?;
        let (start, end) = checked(range, self.head.parsed_upto)?;
        db(&self.path, || {
            self.db.query_row("SELECT EXISTS(SELECT 1 FROM message_source WHERE candidate_pk=?1 AND input_pk=?2 AND chain=?3 AND source_start=?4 AND source_end=?5)",params![build.candidate,build.source.input_pk,build.source.chain,start,end],|r|r.get(0))
        })
    }
}

#[cfg(test)]
mod tests {
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
                .retain_immutable_input(
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
            // Synthetic committed byte prefix: this fixture does not assert that
            // arbitrary subranges below coincide with JSON record boundaries.
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
    fn source_set_duplicates_disjoint_ranges_and_pinned_membership_survive_reopen() {
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
    fn stale_cursor_and_late_native_failure_roll_back_locator_and_cursor() {
        let mut f = Fixture::new();
        let a = SourceRange { start: 1, end: 9 };
        assert!(matches!(f.add(a, 1), Err(Error::MessagePredecessorChanged)));
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
        assert!(matches!(f.add(a, 0), Err(Error::MessagePredecessorChanged)));
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
    fn range_containment_and_live_basis_privacy_and_retirement_are_required() {
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
    fn active_normalized_origin_or_block_lease_cannot_be_skipped_by_source_range() {
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
    fn advancing_source_head_preserves_old_snapshot_but_rejects_old_build_on_new_head() {
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
            Err(Error::MessagePredecessorChanged)
        ));
        let new = f.writer.parser_snapshot(&next.token).unwrap();
        assert!(matches!(
            new.has_message_source(&f.build, range),
            Err(Error::MessagePredecessorChanged)
        ));
        assert_eq!(f.count(), 1);
    }
    #[test]
    fn source_input_identity_is_bounded_basis_checked_and_snapshot_pinned() {
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
            Err(Error::MessagePredecessorChanged)
        ));
        // Deliberately corrupt only this disposable fixture's index identity:
        // the old view stays coherent; lookup does not claim raw-ID validation.
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
}
