//! Parses one window of raw bytes against a pinned parser-state snapshot. No
//! file IO and no SQL writes. The framing continuation is released for reuse
//! only after the matching install has been published.
use crate::state::{claude, codex};
use polylogue_sources::frame::{
    Counts, JsonlContinuation, JsonlDriver, ParseFailure, PendingJsonlStep, RecordParser, Step,
    StepBudget, StepState,
};
use polylogue_sources::{claude_code, codex as codex_source};
use polylogue_store::{
    Error as StoreError,
    parser_state::{Accounting, Budget, CellChange, Head, Phase, Progress, Snapshot, Token},
};

#[derive(Clone, Debug)]
pub enum SourceContext {
    Codex,
    Claude(claude_code::FileCtx),
}
#[derive(Clone, Copy)]
pub struct Budgets {
    pub framing: StepBudget,
    pub reads: Budget,
    pub encoding: Budget,
}
pub struct Window<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}
#[derive(Debug)]
pub enum Error {
    Store(StoreError),
    Provider(ParseFailure<String>),
    ProviderInitialization {
        offset: u64,
        error: String,
    },
    InvalidWindow {
        expected: u64,
        actual: u64,
    },
    WindowBeyondAcquisition,
    EmptyWindowBeforeFrontier,
    FramingCapacity {
        required: usize,
        allocation_failed: bool,
    },
    EmptyFramingBudget,
    /// The append reaches output that is already finished. Parse the input
    /// again instead of applying a partial delta.
    ReinterpretationRequired,
}
impl From<StoreError> for Error {
    fn from(e: StoreError) -> Self {
        Self::Store(e)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => std::fmt::Display::fmt(e, f),
            Self::Provider(e) => std::fmt::Display::fmt(e, f),
            Self::ProviderInitialization { offset, error } => {
                write!(f, "source initialization failed at {offset}: {error}")
            }
            Self::InvalidWindow { expected, actual } => {
                write!(f, "source window begins at {actual}, expected {expected}")
            }
            Self::EmptyWindowBeforeFrontier => {
                f.write_str("source window is empty before its acquisition frontier")
            }
            Self::WindowBeyondAcquisition => {
                f.write_str("source window exceeds its pinned acquisition")
            }
            Self::FramingCapacity {
                required,
                allocation_failed,
            } => write!(
                f,
                "source framing requires {required} bytes (allocation failed: {allocation_failed})"
            ),
            Self::EmptyFramingBudget => {
                f.write_str("source framing needs positive byte and record allowances")
            }
            Self::ReinterpretationRequired => {
                f.write_str("source append reaches finished output; the input must be parsed again")
            }
        }
    }
}
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Store(e) => Some(e),
            _ => None,
        }
    }
}
type Result<T> = std::result::Result<T, Error>;

/// Not cloneable and not constructible from inspected offsets: its token is
/// the publication that released the framing state.
pub struct CommittedContinuation {
    predecessor: Token,
    framing: JsonlContinuation,
}
impl CommittedContinuation {
    pub fn predecessor(&self) -> &Token {
        &self.predecessor
    }
    pub fn parsed_upto(&self) -> u64 {
        self.framing.parsed_upto()
    }
    pub fn scanned_upto(&self) -> u64 {
        self.framing.scanned_upto()
    }
    pub fn buffer_capacity(&self) -> usize {
        self.framing.buffer_capacity()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdleReason {
    WaitingForAppend,
    FinalizationPending,
}
pub enum Computation {
    Idle {
        reason: IdleReason,
        continuation: Option<CommittedContinuation>,
    },
    Pending(PendingTurn),
    History {
        progress: Progress,
        changes: Vec<CellChange>,
    },
}
pub struct PendingTurn {
    head: Head,
    pending: PendingJsonlStep<()>,
    changes: Vec<CellChange>,
    progress: Progress,
    expected_counts: Accounting,
    step: Step,
}
/// Submit these fields to `PendingInstall` and keep the `CommitGuard` until
/// `Turn::Published`. A failed or abandoned install drops the guard and parsing
/// restarts from the last committed complete record.
pub struct Install {
    pub predecessor: Head,
    pub progress: Progress,
    pub changes: Vec<CellChange>,
}
pub struct CommitGuard {
    predecessor: Head,
    pending: PendingJsonlStep<()>,
    expected_counts: Accounting,
    phase: Phase,
}
impl PendingTurn {
    pub fn parsed_upto(&self) -> u64 {
        self.pending.parsed_upto()
    }
    pub fn scanned_upto(&self) -> u64 {
        self.pending.scanned_upto()
    }
    pub fn counts(&self) -> &Counts {
        self.pending.counts()
    }
    pub fn buffer_capacity(&self) -> usize {
        self.pending.buffer_capacity()
    }
    pub fn step(&self) -> Step {
        self.step
    }
    pub fn into_install(self) -> (Install, CommitGuard) {
        let phase = self.progress.phase;
        let guard = CommitGuard {
            predecessor: self.head.clone(),
            pending: self.pending,
            expected_counts: self.expected_counts,
            phase,
        };
        (
            Install {
                predecessor: self.head,
                progress: self.progress,
                changes: self.changes,
            },
            guard,
        )
    }
}
impl CommitGuard {
    /// Call only with the receipt of a published `PendingInstall`. A stale or
    /// mismatched receipt discards the continuation.
    pub fn published(self, receipt: &Head) -> Result<CommittedContinuation> {
        let Self {
            predecessor,
            pending,
            expected_counts,
            phase,
        } = self;
        let (predecessor, framing) = pending.commit_with(|(), _, parsed, scanned| {
            let mut next = predecessor.token.clone();
            next.step = next.step.checked_add(1).ok_or(StoreError::Range)?;
            let expected_status = if receipt.acquired_upto > predecessor.acquired_upto
                && matches!(phase, Phase::WaitingTail)
            {
                Phase::Parsing
            } else {
                phase
            };
            if receipt.token != next
                || receipt.parser_version != predecessor.parser_version
                || receipt.parsed_upto != parsed
                || receipt.inspected_upto != scanned
                || receipt.counts != expected_counts
                || !receipt.finalization_pending
                || receipt.acquired_upto < predecessor.acquired_upto
                || receipt.status != expected_status
            {
                return Err(Error::Store(StoreError::StaleParserState));
            }
            Ok(receipt.token.clone())
        })?;
        Ok(CommittedContinuation {
            predecessor,
            framing,
        })
    }
}
/// After a restart there is no continuation: reread from `parsed_upto`,
/// including any inspected but unparsed tail. Never start at the inspected offset.
pub fn read_offset(head: &Head, continuation: Option<&CommittedContinuation>) -> Result<u64> {
    match continuation {
        None => Ok(head.parsed_upto),
        Some(c)
            if c.predecessor == head.token
                && c.parsed_upto() == head.parsed_upto
                && c.scanned_upto() == head.inspected_upto =>
        {
            Ok(c.scanned_upto())
        }
        Some(_) => Err(StoreError::StaleParserState.into()),
    }
}
fn accounting(c: &Counts) -> Result<Accounting> {
    let ignored = c
        .ignored
        .values()
        .try_fold(0u64, |n, v| n.checked_add(*v).ok_or(StoreError::Range))?;
    let result = Accounting {
        framed: c.framed,
        used: c.used,
        ignored,
        unrecognized: u64::try_from(c.unrecognized.len()).map_err(|_| StoreError::Range)?,
        undecodable: u64::try_from(c.undecodable.len()).map_err(|_| StoreError::Range)?,
    };
    if !c.balanced() {
        return Err(StoreError::Invariant("unbalanced source framing counts".into()).into());
    }
    Ok(result)
}
fn sum(a: Accounting, b: Accounting) -> Result<Accounting> {
    let add = |a: u64, b: u64| a.checked_add(b).ok_or(Error::Store(StoreError::Range));
    Ok(Accounting {
        framed: add(a.framed, b.framed)?,
        used: add(a.used, b.used)?,
        ignored: add(a.ignored, b.ignored)?,
        unrecognized: add(a.unrecognized, b.unrecognized)?,
        undecodable: add(a.undecodable, b.undecodable)?,
    })
}
fn run<P>(
    parser: P,
    continuation: Option<CommittedContinuation>,
    window: &[u8],
    base: u64,
    budget: StepBudget,
) -> Result<(PendingJsonlStep<P::Output>, Step)>
where
    P: RecordParser<Error = String>,
{
    let mut driver = match continuation {
        Some(c) => JsonlDriver::resume(parser, c.framing),
        None => JsonlDriver::new(parser, base),
    };
    let step = driver.advance(window, budget);
    match step {
        Ok(Step {
            state: StepState::NeedsCapacity { required },
            ..
        }) => {
            return Err(Error::FramingCapacity {
                required,
                allocation_failed: false,
            });
        }
        Ok(Step {
            state: StepState::AllocationFailed { required },
            ..
        }) => {
            return Err(Error::FramingCapacity {
                required,
                allocation_failed: true,
            });
        }
        Err(_) => {
            let framed = driver.finish();
            return Err(match framed.output {
                Err(e) => Error::Provider(e),
                Ok(_) => Error::Store(StoreError::Invariant(
                    "failed source callback returned successful output".into(),
                )),
            });
        }
        _ => {}
    }
    let step = step.map_err(|_| StoreError::Invariant("source framing lost its failure".into()))?;
    let pending = driver.checkpoint().map_err(|f| match f.output {
        Err(e) => Error::Provider(e),
        Ok(_) => Error::Store(StoreError::Invariant(
            "failed source checkpoint returned successful output".into(),
        )),
    })?;
    Ok((pending, step))
}
/// `SourceContext` and the companion manifest must already be part of the
/// actor's parser identity. Never picks a Claude path from competing observations.
pub fn compute(
    snapshot: &Snapshot,
    continuation: Option<CommittedContinuation>,
    window: Window<'_>,
    context: SourceContext,
    budgets: Budgets,
) -> Result<Computation> {
    compute_with_bodies(snapshot, continuation, window, context, budgets, &[])
}
pub fn compute_with_bodies(
    snapshot: &Snapshot,
    continuation: Option<CommittedContinuation>,
    window: Window<'_>,
    context: SourceContext,
    budgets: Budgets,
    bodies: &[polylogue_store::claude_bodies::Companion],
) -> Result<Computation> {
    let head = &snapshot.head;
    let version = match &context {
        SourceContext::Codex => codex_source::PARSER_VERSION,
        SourceContext::Claude(_) => claude_code::PARSER_VERSION,
    };
    if head.parser_version != version {
        return Err(StoreError::Invariant(
            "source context parser differs from the pinned parser identity".into(),
        )
        .into());
    }
    let offset = read_offset(head, continuation.as_ref())?;
    if window.offset != offset {
        return Err(Error::InvalidWindow {
            expected: offset,
            actual: window.offset,
        });
    }
    let end = window
        .offset
        .checked_add(u64::try_from(window.bytes.len()).map_err(|_| StoreError::Range)?)
        .ok_or(StoreError::Range)?;
    if end > head.acquired_upto {
        return Err(Error::WindowBeyondAcquisition);
    }
    // A stored stable tail also prevents rereading in a loop after restart.
    if head.status == Phase::WaitingTail && head.inspected_upto == head.acquired_upto {
        return Ok(Computation::Idle {
            reason: IdleReason::WaitingForAppend,
            continuation,
        });
    }
    if head.token.step > 0
        && head.parsed_upto == head.acquired_upto
        && head.inspected_upto == head.acquired_upto
    {
        return Ok(Computation::Idle {
            reason: IdleReason::FinalizationPending,
            continuation,
        });
    }
    if window.bytes.is_empty() && offset < head.acquired_upto {
        return Err(Error::EmptyWindowBeforeFrontier);
    }
    if budgets.framing.bytes == 0 || budgets.framing.records == 0 {
        return Err(Error::EmptyFramingBudget);
    }
    let mut changes = None;
    let (pending, step) = match context {
        SourceContext::Codex => {
            let attempt = codex::Attempt::new(snapshot, budgets.reads);
            let resume = match attempt.resume() {
                Ok(v) => v,
                Err(e) => {
                    return Err(attempt.take_failure().map(Error::Store).unwrap_or_else(|| {
                        Error::ProviderInitialization {
                            offset: head.parsed_upto,
                            error: e,
                        }
                    }));
                }
            };
            let outcome = run(
                codex_source::keyed::KeyedRecords::new(&attempt, resume),
                continuation,
                window.bytes,
                head.parsed_upto,
                budgets.framing,
            );
            if let Some(e) = attempt.take_failure() {
                return Err(Error::Store(e));
            }
            let (pending, step) = outcome?;
            let pending = pending.try_map_output(|mut delta, _, parsed, _| -> Result<()> {
                if delta.reached_back {
                    return Err(Error::ReinterpretationRequired);
                }
                delta.resume.parsed_upto = parsed;
                let next = head.token.step.checked_add(1).ok_or(StoreError::Range)?;
                let next = u64::try_from(next).map_err(|_| StoreError::Range)?;
                changes = Some(codex::encode_delta(&delta, next, budgets.encoding)?.changes);
                Ok(())
            })?;
            (pending, step)
        }
        SourceContext::Claude(ctx) => {
            let view = claude::View::new(snapshot, budgets.reads)?.with_bodies(bodies);
            let parser = match claude_code::KeyedRecords::new(ctx, &view) {
                Ok(p) => p,
                Err(e) => {
                    return Err(view.take_error().map(Error::Store).unwrap_or_else(|| {
                        Error::ProviderInitialization {
                            offset: head.parsed_upto,
                            error: e,
                        }
                    }));
                }
            };
            let outcome = run(
                parser,
                continuation,
                window.bytes,
                head.parsed_upto,
                budgets.framing,
            );
            if let Some(e) = view.take_error() {
                return Err(Error::Store(e));
            }
            let (pending, step) = outcome?;
            let pending =
                pending.try_map_output(|mut output, counts, parsed, _| -> Result<()> {
                    let mut encoded =
                        claude::encode_step(&mut output, head, parsed, counts, budgets.encoding)?;
                    for body in bodies {
                        encoded.push(claude::encode_body(
                            &body.path,
                            Some(&claude_code::keyed::BodyPayload {
                                blob: body.blob,
                                length: body.length,
                            }),
                            budgets.encoding,
                        )?);
                    }
                    changes = Some(encoded);
                    Ok(())
                })?;
            (pending, step)
        }
    };
    let phase = if pending.scanned_upto() == head.acquired_upto {
        if pending.parsed_upto() < pending.scanned_upto() {
            Phase::WaitingTail
        } else {
            Phase::Finalizing
        }
    } else {
        Phase::Parsing
    };
    let counts = accounting(pending.counts())?;
    let expected_counts = sum(head.counts, counts)?;
    let progress = Progress {
        inspected_upto: pending.scanned_upto(),
        parsed_upto: pending.parsed_upto(),
        phase,
        finalization_pending: true,
        counts,
    };
    Ok(Computation::Pending(PendingTurn {
        head: head.clone(),
        pending,
        changes: changes.ok_or_else(|| {
            StoreError::Invariant("source encoder produced no install batch".into())
        })?,
        progress,
        expected_counts,
        step,
    }))
}
