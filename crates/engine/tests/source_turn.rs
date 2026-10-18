//! Actual retained bytes, keyed sources and prepared provider installation.
//! These tests do not certify canonical Message publication or public clients.
use polylogue_engine::{
    converge::{PendingInstall, Turn},
    source_turn::{
        self, Budgets, CommittedContinuation, Computation, Error, IdleReason, PendingTurn,
        SourceContext, Window,
    },
};
use polylogue_model::BlobHash;
use polylogue_sources::frame::StepBudget;
use polylogue_sources::{
    claude_code::{self, FileCtx},
    codex,
};
use polylogue_store::{
    Error as StoreError, ImportObservation, InputFormat, Writer,
    parser_state::{Basis, Budget, Head},
};
use serde_json::json;
use std::path::PathBuf;
fn budgets() -> Budgets {
    Budgets {
        framing: StepBudget {
            bytes: 1024 * 1024,
            records: 100,
            record_bytes: 1024 * 1024,
        },
        reads: Budget {
            cells: 65536,
            encoded_bytes: 16 * 1024 * 1024,
        },
        encoding: Budget {
            cells: 65536,
            encoded_bytes: 16 * 1024 * 1024,
        },
    }
}
fn context(claude: bool) -> SourceContext {
    if claude {
        SourceContext::Claude(FileCtx::main("synthetic-session"))
    } else {
        SourceContext::Codex
    }
}
fn fixture(claude: bool) -> Vec<u8> {
    let lines = if claude {
        vec![
            json!({"type":"user","uuid":"u","sessionId":"synthetic-session","promptSource":"typed","message":{"content":"hello"}}),
            json!({"type":"assistant","uuid":"a","sessionId":"synthetic-session","requestId":"r","message":{"id":"answer","content":[{"type":"text","text":"hello back"}]}}),
        ]
    } else {
        vec![
            json!({"type":"session_meta","payload":{"id":"synthetic-session","source":"cli"}}),
            json!({"type":"response_item","payload":{"type":"message","id":"u","role":"user","content":[{"text":"hello"}]}}),
        ]
    };
    lines
        .iter()
        .map(|v| v.to_string() + "\n")
        .collect::<String>()
        .into_bytes()
}
struct World {
    _dir: tempfile::TempDir,
    root: PathBuf,
    writer: Option<Writer>,
    head: Head,
    hash: BlobHash,
    claude: bool,
}
impl World {
    fn new(bytes: &[u8], claude: bool, acquired: u64) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("archive");
        let mut writer = Writer::open(&root).unwrap();
        let hash = writer.put_file_blob(bytes).unwrap();
        let input = writer
            .store_immutable_input(
                hash,
                if claude {
                    InputFormat::ClaudeCodeJsonl
                } else {
                    InputFormat::CodexRollout
                },
                if claude {
                    &["claude-main", "synthetic-session"]
                } else {
                    &[]
                },
                &ImportObservation {
                    id: "synthetic-import",
                    machine: "synthetic-machine",
                    host: "synthetic-host",
                    path: "/synthetic/explicit-session.jsonl",
                    seen_at: 0,
                },
                4096,
            )
            .unwrap();
        let head = writer
            .begin_parser(&Basis {
                input_pk: input.input_pk,
                chain: input.chain,
                parser_version: if claude {
                    claude_code::PARSER_VERSION
                } else {
                    codex::PARSER_VERSION
                },
                digest: *hash.bytes(),
                acquired_upto: acquired,
                companions_json: Some(r#"{"synthetic_context":"synthetic-session"}"#),
                excisions_seen: 0,
            })
            .unwrap();
        Self {
            _dir: dir,
            root,
            writer: Some(writer),
            head,
            hash,
            claude,
        }
    }
    fn writer(&self) -> &Writer {
        self.writer.as_ref().unwrap()
    }
    fn writer_mut(&mut self) -> &mut Writer {
        self.writer.as_mut().unwrap()
    }
    fn grow(&mut self, acquired: u64) {
        let old = &self.head;
        let basis = Basis {
            input_pk: old.token.input_pk,
            chain: old.token.chain,
            parser_version: old.parser_version,
            digest: old.token.basis,
            acquired_upto: acquired,
            companions_json: Some(r#"{"synthetic_context":"synthetic-session"}"#),
            excisions_seen: 0,
        };
        self.head = self.writer.as_mut().unwrap().begin_parser(&basis).unwrap();
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
    fn compute(
        &self,
        continuation: Option<CommittedContinuation>,
        length: usize,
        budget: Budgets,
    ) -> Result<Computation, Error> {
        let offset = source_turn::read_offset(&self.head, continuation.as_ref())?;
        let reader = self.writer().read_handle().unwrap();
        let bytes = reader
            .read_blob_range(self.hash, offset, length as u64)
            .unwrap();
        let snapshot = self.writer().parser_snapshot(&self.head.token).unwrap();
        source_turn::compute(
            &snapshot,
            continuation,
            Window {
                offset,
                bytes: &bytes,
            },
            context(self.claude),
            budget,
        )
    }
    fn publish(&mut self, pending: PendingTurn) -> CommittedContinuation {
        let (install, guard) = pending.into_install();
        let mut install = PendingInstall::begin(
            self.writer_mut(),
            &install.predecessor,
            install.progress,
            install.changes,
        )
        .unwrap();
        let head = loop {
            match install
                .turn(
                    self.writer_mut(),
                    Budget {
                        cells: 2,
                        encoded_bytes: 2 * 1024 * 1024,
                    },
                )
                .unwrap()
            {
                Turn::Yielded => {}
                Turn::Published(h) => break h,
            }
        };
        let continuation = guard.published(&head).unwrap();
        self.head = head;
        continuation
    }
}
fn pending(result: Computation) -> PendingTurn {
    match result {
        Computation::Pending(p) => p,
        Computation::Idle { .. } | Computation::History { .. } => {
            panic!("expected JSONL computation")
        }
    }
}
#[test]
fn both_sources_publish_only_after_real_prepared_install_and_reopen_resume() {
    for claude in [false, true] {
        let bytes = fixture(claude);
        let mut world = World::new(&bytes, claude, bytes.len() as u64);
        let mut budget = budgets();
        budget.framing.records = 1;
        let first = pending(world.compute(None, bytes.len(), budget).unwrap());
        assert_eq!(first.counts().framed, 1);
        assert!(first.parsed_upto() < (bytes.len() as u64));
        let continuation = world.publish(first);
        assert_eq!(continuation.predecessor(), &world.head.token);
        assert_eq!(world.head.status, "parsing");
        let offset = continuation.scanned_upto();
        world.restart();
        let second = pending(
            world
                .compute(Some(continuation), bytes.len() - offset as usize, budgets())
                .unwrap(),
        );
        assert_eq!(second.counts().framed, 1);
        let continuation = world.publish(second);
        assert_eq!(world.head.counts.framed, 2);
        assert_eq!(world.head.parsed_upto, bytes.len() as u64);
        assert_eq!(world.head.status, "finalizing");
        assert!(world.head.finalization_pending);
        assert!(matches!(
            world.compute(Some(continuation), 0, budgets()).unwrap(),
            Computation::Idle {
                reason: IdleReason::FinalizationPending,
                ..
            }
        ));
        let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
        if claude {
            use claude_code::keyed::ReadView;
            let view =
                polylogue_engine::state::claude::View::new(&snapshot, budgets().reads).unwrap();
            assert_eq!(view.header().unwrap().parsed_upto, bytes.len() as u64);
            assert!(view.effects(world.head.token.step).unwrap().is_some());
        } else {
            let view = polylogue_engine::state::codex::Attempt::new(&snapshot, budgets().reads);
            assert_eq!(view.resume().unwrap().parsed_upto, bytes.len() as u64);
            assert!(
                view.effects(world.head.token.step as u64)
                    .unwrap()
                    .collect()
                    .is_ok()
            );
        }
    }
}
#[test]
fn partial_tail_waits_without_spin_and_resumes_or_replays_from_complete_record_head() {
    for claude in [false, true] {
        for restart in [false, true] {
            let bytes = fixture(claude);
            let first_end = bytes.iter().position(|b| *b == b'\n').unwrap() + 1;
            let cut = first_end + 9;
            let mut world = World::new(&bytes, claude, cut as u64);
            let first = pending(world.compute(None, cut, budgets()).unwrap());
            assert_eq!(first.parsed_upto(), first_end as u64);
            assert_eq!(first.scanned_upto(), cut as u64);
            let continuation = world.publish(first);
            assert_eq!(world.head.status, "waiting_tail");
            let token = world.head.token.clone();
            let idle = world.compute(Some(continuation), 0, budgets()).unwrap();
            let continuation = match idle {
                Computation::Idle {
                    reason: IdleReason::WaitingForAppend,
                    continuation,
                } => continuation,
                _ => panic!("static tail was not waiting"),
            };
            assert_eq!(world.head.token, token);
            let continuation = if restart {
                drop(continuation);
                world.restart();
                None
            } else {
                continuation
            };
            if restart {
                assert_eq!(
                    source_turn::read_offset(&world.head, None).unwrap(),
                    first_end as u64
                );
                assert!(matches!(
                    world.compute(None, cut - first_end, budgets()).unwrap(),
                    Computation::Idle {
                        reason: IdleReason::WaitingForAppend,
                        ..
                    }
                ));
            }
            world.grow(bytes.len() as u64);
            let offset = source_turn::read_offset(&world.head, continuation.as_ref()).unwrap();
            assert_eq!(offset, if restart { first_end } else { cut } as u64);
            let next = pending(
                world
                    .compute(continuation, bytes.len() - offset as usize, budgets())
                    .unwrap(),
            );
            assert_eq!(next.counts().framed, 1);
            world.publish(next);
            assert_eq!(world.head.counts.framed, 2);
            assert_eq!(world.head.parsed_upto, bytes.len() as u64);
            assert_eq!(world.head.status, "finalizing");
        }
    }
}
#[test]
fn stale_install_drops_checkpoint_and_replays_from_committed_predecessor() {
    for claude in [false, true] {
        let bytes = fixture(claude);
        let mut world = World::new(&bytes, claude, bytes.len() as u64);
        let mut budget = budgets();
        budget.framing.records = 1;
        let first = pending(world.compute(None, bytes.len(), budget).unwrap());
        let stale = pending(world.compute(None, bytes.len(), budget).unwrap());
        drop(world.publish(first));
        let (stale, guard) = stale.into_install();
        assert!(matches!(
            PendingInstall::begin(
                world.writer_mut(),
                &stale.predecessor,
                stale.progress,
                stale.changes
            ),
            Err(StoreError::StaleParserState)
        ));
        drop(guard);
        world.restart();
        let offset = source_turn::read_offset(&world.head, None).unwrap();
        assert_eq!(offset, world.head.parsed_upto);
        let replay = pending(
            world
                .compute(None, bytes.len() - offset as usize, budgets())
                .unwrap(),
        );
        world.publish(replay);
        assert_eq!(world.head.counts.framed, 2);
    }
}
#[test]
fn wrong_publication_receipt_or_window_never_releases_uncommitted_framing() {
    let bytes = fixture(false);
    let world = World::new(&bytes, false, bytes.len() as u64);
    let pending = pending(world.compute(None, bytes.len(), budgets()).unwrap());
    let (_, guard) = pending.into_install();
    assert!(matches!(
        guard.published(&world.head),
        Err(Error::Store(StoreError::StaleParserState))
    ));
    let snapshot = world.writer().parser_snapshot(&world.head.token).unwrap();
    assert!(matches!(
        source_turn::compute(
            &snapshot,
            None,
            Window {
                offset: 1,
                bytes: &bytes[1..]
            },
            context(false),
            budgets()
        ),
        Err(Error::InvalidWindow {
            expected: 0,
            actual: 1
        })
    ));
    assert!(matches!(
        source_turn::compute(
            &snapshot,
            None,
            Window {
                offset: 0,
                bytes: &[]
            },
            context(false),
            budgets()
        ),
        Err(Error::EmptyWindowBeforeFrontier)
    ));
    let mut budget = budgets();
    budget.framing.record_bytes = 1;
    assert!(matches!(
        world.compute(None, bytes.len(), budget),
        Err(Error::FramingCapacity { .. })
    ));
    assert_eq!(
        world
            .writer()
            .parser_head(world.head.token.input_pk, world.head.token.chain)
            .unwrap()
            .unwrap()
            .token,
        world.head.token
    );
}
#[test]
fn native_state_and_encode_admission_are_not_provider_refusals() {
    for claude in [false, true] {
        let bytes = fixture(claude);
        let world = World::new(&bytes, claude, bytes.len() as u64);
        let mut budget = budgets();
        budget.reads.cells = 0;
        assert!(matches!(
            world.compute(None, bytes.len(), budget),
            Err(Error::Store(StoreError::ParserAdmission { .. }))
        ));
        let mut budget = budgets();
        budget.encoding.cells = 0;
        assert!(matches!(
            world.compute(None, bytes.len(), budget),
            Err(Error::Store(StoreError::ParserAdmission { .. }))
        ));
        assert_eq!(
            world
                .writer()
                .parser_head(world.head.token.input_pk, world.head.token.chain)
                .unwrap()
                .unwrap()
                .token,
            world.head.token
        );
    }
}

#[test]
fn accessible_empty_registered_inputs_still_owe_canonical_finalization() {
    for claude in [false, true] {
        let mut world = World::new(&[], claude, 0);
        let empty = pending(world.compute(None, 0, budgets()).unwrap());
        assert_eq!(empty.counts().framed, 0);
        let continuation = world.publish(empty);
        assert_eq!(world.head.status, "finalizing");
        assert!(world.head.finalization_pending);
        assert_eq!(world.head.parsed_upto, 0);
        assert!(matches!(
            world.compute(Some(continuation), 0, budgets()).unwrap(),
            Computation::Idle {
                reason: IdleReason::FinalizationPending,
                ..
            }
        ));
    }
}
