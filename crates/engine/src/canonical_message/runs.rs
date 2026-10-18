//! Homogeneous runs of already verified fields or citations. Structural and
//! streaming actions remain one ordinary turn. No mutations occur on this worker.
use super::*;
enum Run {
    Fields(Vec<SmallFieldEntry>),
    Ranges(Vec<SourceRangeEntry>),
}
fn cost(change: &Change, _sequence: u64, allowance: usize) -> Result<Option<(usize, u64)>> {
    Ok(match change {
        Change::SmallField {
            field, slot, next, ..
        } => {
            let key = bounded_json(slot, allowance)?;
            let bytes = field
                .encoded_bytes()
                .checked_add(key.len())
                .and_then(|n| n.checked_add(next.len()))
                .and_then(|n| n.checked_add(std::mem::size_of::<SmallFieldEntry>()))
                .ok_or(StoreError::Range)?;
            Some((bytes, field.hash_work()))
        }
        Change::SourceRange { next, .. } => Some((
            next.len()
                .checked_add(std::mem::size_of::<SourceRangeEntry>())
                .ok_or(StoreError::Range)?,
            0,
        )),
        _ => None,
    })
}
impl Run {
    fn from(change: Change, sequence: u64) -> Self {
        match change {
            Change::SmallField {
                field,
                slot,
                time,
                block,
                supplied,
                next,
            } => Self::Fields(vec![SmallFieldEntry {
                field,
                slot,
                source_time: time,
                block,
                supplied,
                expected_sequence: sequence,
                next_cursor: next,
            }]),
            Change::SourceRange { range, next } => Self::Ranges(vec![SourceRangeEntry {
                range,
                expected_sequence: sequence,
                next_cursor: next,
            }]),
            _ => unreachable!("admitted homogeneous run"),
        }
    }
    fn compatible(&self, change: &Change) -> bool {
        matches!(
            (self, change),
            (Self::Fields(_), Change::SmallField { .. })
                | (Self::Ranges(_), Change::SourceRange { .. })
        )
    }
    fn push(&mut self, change: Change, sequence: u64) {
        match (self, Self::from(change, sequence)) {
            (Self::Fields(a), Self::Fields(mut b)) => a.append(&mut b),
            (Self::Ranges(a), Self::Ranges(mut b)) => a.append(&mut b),
            _ => unreachable!("checked homogeneous run"),
        }
    }
    fn last(&self) -> (u64, &[u8]) {
        match self {
            Self::Fields(a) => {
                let e = a.last().unwrap();
                (e.expected_sequence, &e.next_cursor)
            }
            Self::Ranges(a) => {
                let e = a.last().unwrap();
                (e.expected_sequence, &e.next_cursor)
            }
        }
    }
    fn change(self) -> Change {
        match self {
            Self::Fields(mut a) if a.len() == 1 => {
                let e = a.pop().unwrap();
                Change::SmallField {
                    field: e.field,
                    slot: e.slot,
                    time: e.source_time,
                    block: e.block,
                    supplied: e.supplied,
                    next: e.next_cursor,
                }
            }
            Self::Ranges(mut a) if a.len() == 1 => {
                let e = a.pop().unwrap();
                Change::SourceRange {
                    range: e.range,
                    next: e.next_cursor,
                }
            }
            Self::Fields(a) => Change::SmallFieldRun(a),
            Self::Ranges(a) => Change::SourceRangeRun(a),
        }
    }
}
fn admission(error: &Error) -> bool {
    use polylogue_sources::selection::ValueError;
    matches!(
        error,
        Error::Store(StoreError::ParserAdmission { .. })
            | Error::Selection(
                ValueError::Admission { .. }
                    | ValueError::Resolve(ResolveError::Store(StoreError::ParserAdmission { .. }))
            )
            | Error::ClaudeSelection(
                ValueError::Admission { .. }
                    | ValueError::Resolve(ResolveError::Store(StoreError::ParserAdmission { .. }))
            )
    )
}
fn can_follow(next: &[u8], fields: bool) -> Result<bool> {
    let c: Cursor = serde_json::from_slice(next).map_err(|e| Error::Source(e.to_string()))?;
    Ok(match (&c.stage, fields) {
        (Stage::Citations(_), false) => true,
        (Stage::Fields { .. } | Stage::MessageField { .. }, true) => true,
        (Stage::Claude(stage), _) => match (stage, fields) {
            (
                claude::Cursor::Citations { .. }
                | claude::Cursor::CompletionCitation { .. }
                | claude::Cursor::SelectedCitations { .. }
                | claude::Cursor::ParentCitations { .. }
                | claude::Cursor::RecordExtraCitations { .. }
                | claude::Cursor::AuthorCitation,
                false,
            ) => true,
            (
                claude::Cursor::Author { .. }
                | claude::Cursor::MessageFields { .. }
                | claude::Cursor::Selected { .. }
                | claude::Cursor::RecordExtras { .. },
                true,
            ) => true,
            (
                claude::Cursor::Block {
                    phase:
                        claude::BlockPhase::Fields(_)
                        | claude::BlockPhase::Retained(_)
                        | claude::BlockPhase::Signature,
                    ..
                },
                true,
            ) => true,
            _ => false,
        },
        _ => false,
    })
}
fn seekable(next: &[u8]) -> Result<bool> {
    let cursor: Cursor = serde_json::from_slice(next).map_err(|e| Error::Source(e.to_string()))?;
    if cursor.verified.is_some() {
        return Ok(false);
    }
    Ok(!matches!(
        cursor.stage,
        Stage::Layout
            | Stage::Install { .. }
            | Stage::Proof
            | Stage::AwaitPresence { .. }
            | Stage::Claude(
                claude::Cursor::Layout
                    | claude::Cursor::Finish
                    | claude::Cursor::Block {
                        phase: claude::BlockPhase::Begin
                            | claude::BlockPhase::Command
                            | claude::BlockPhase::Install,
                        ..
                    }
            )
    ))
}
fn first_cost(state: &State) -> usize {
    state
        .projection
        .cursor
        .as_ref()
        .map_or(0, Vec::len)
        .saturating_add(state.handle.guard.len())
        .saturating_mul(3)
        .saturating_add(std::mem::size_of::<Guard>() + std::mem::size_of::<State>())
}
// Only cursor replacement is virtual. There is no virtual field/value/block or
// source membership overlay. Producers and verified proof checkpoints never enter.
fn seek(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    b: Budgets,
    flag: &AtomicBool,
    first: Action,
) -> Result<Action> {
    let ActionKind::Turn {
        guard,
        change: Change::Advance(mut next),
    } = first.inner
    else {
        return Ok(first);
    };
    let fixed = first_cost(state);
    let mut virtual_state = state.clone();
    loop {
        cancelled(flag)?;
        if !seekable(&next)? {
            break;
        }
        let mut left = b;
        left.metadata = b
            .metadata
            .saturating_sub(fixed)
            .saturating_sub(next.len().saturating_mul(3));
        left.value_chunk = left.value_chunk.min(left.metadata);
        if left.metadata == 0 || left.value_chunk == 0 {
            break;
        }
        virtual_state.projection.cursor = Some(next.clone());
        // No durable cursor was committed. Its sequence and active producer
        // heads remain EXACTLY those inspected at entry throughout this seek.
        let candidate = match compute_one(snapshot, reader, &virtual_state, left, flag) {
            Ok(a) => a,
            Err(e) if admission(&e) => break,
            Err(e) => return Err(e),
        };
        match candidate.inner {
            ActionKind::Turn {
                change: Change::Advance(cursor),
                ..
            } => next = cursor,
            ActionKind::Turn {
                change: change @ (Change::SmallField { .. } | Change::SourceRange { .. }),
                ..
            } => {
                return Ok(Action {
                    inner: ActionKind::Turn { guard, change },
                });
            }
            // Persist the chosen cursor BEFORE an action which does not carry
            // it (BeginValue/BeginOrigin), or before structural/live effects.
            _ => break,
        }
    }
    cancelled(flag)?;
    Ok(Action {
        inner: ActionKind::Turn {
            guard,
            change: Change::Advance(next),
        },
    })
}
pub(super) fn compute(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<Action> {
    // One scope for every adapter and raw-selector record in this worker. A
    // later admission failure yields the already verified prefix, never resets it.
    // Streaming leases and their Verified cursor retain every existing
    // checkpoint, including the verification before an origin producer starts.
    if state.preparation.active_value.is_some() || state.preparation.active_origin_value.is_some() {
        return compute_one(snapshot, reader, state, b, flag);
    }
    let scope = snapshot.bounded_reads(b.read)?;
    let first = compute_one(snapshot, reader, state, b, flag)?;
    if !scope.enforce() {
        return Ok(first);
    }
    let first = seek(snapshot, reader, state, b, flag, first)?;
    let ActionKind::Turn { guard, change } = first.inner else {
        return Ok(first);
    };
    let (bytes, hash) = match cost(&change, guard.sequence, b.metadata) {
        Ok(Some(cost)) => cost,
        Ok(None) => {
            return Ok(Action {
                inner: ActionKind::Turn { guard, change },
            });
        }
        Err(e) if admission(&e) => {
            return Ok(Action {
                inner: ActionKind::Turn { guard, change },
            });
        }
        Err(e) => return Err(e),
    };
    let head_bytes = match bounded_json(&state.handle.head, b.metadata) {
        Ok(bytes) => bytes.len(),
        Err(e) if admission(&e) => {
            return Ok(Action {
                inner: ActionKind::Turn { guard, change },
            });
        }
        Err(e) => return Err(e),
    };
    let fixed = guard
        .cursor
        .as_ref()
        .map_or(0, Vec::len)
        .saturating_add(head_bytes)
        .saturating_add(state.handle.guard.len())
        .saturating_mul(3)
        .saturating_add(std::mem::size_of::<Guard>() + std::mem::size_of::<State>());
    if bytes > b.metadata.saturating_sub(fixed) || hash > b.hash_bytes {
        return Ok(Action {
            inner: ActionKind::Turn { guard, change },
        });
    }
    let mut retained = fixed + bytes;
    let mut hash_left = b.hash_bytes - hash;
    let mut run = Run::from(change, guard.sequence);
    let mut virtual_state = state.clone();
    loop {
        cancelled(flag)?;
        let (sequence, next) = run.last();
        if next.len().saturating_mul(2) > b.metadata.saturating_sub(retained)
            || !can_follow(next, matches!(&run, Run::Fields(_)))?
        {
            break;
        }
        virtual_state.projection = ProjectionState {
            sequence: sequence.checked_add(1).ok_or(StoreError::Range)?,
            cursor: Some(next.to_vec()),
        };
        // Lookahead has no writes. Both its reads and temporary serialization
        // use the remaining original allowance, even when it ends this run.
        let mut remaining = b;
        remaining.metadata = b
            .metadata
            .saturating_sub(retained)
            .saturating_sub(next.len().saturating_mul(2));
        remaining.value_chunk = remaining.value_chunk.min(remaining.metadata);
        remaining.hash_bytes = hash_left;
        if remaining.metadata == 0 || remaining.value_chunk == 0 || hash_left == 0 {
            break;
        }
        let next = match compute_one(snapshot, reader, &virtual_state, remaining, flag) {
            Ok(a) => a,
            Err(e) if admission(&e) => break,
            Err(e) => return Err(e),
        };
        let ActionKind::Turn {
            guard: next_guard,
            change,
        } = next.inner
        else {
            break;
        };
        if !run.compatible(&change) {
            break;
        }
        let (bytes, hash) = match cost(&change, next_guard.sequence, remaining.metadata) {
            Ok(Some(cost)) => cost,
            Ok(None) => break,
            Err(e) if admission(&e) => break,
            Err(e) => return Err(e),
        };
        if bytes > remaining.metadata || hash > hash_left {
            break;
        }
        retained += bytes;
        hash_left -= hash;
        run.push(change, next_guard.sequence);
    }
    cancelled(flag)?;
    Ok(Action {
        inner: ActionKind::Turn {
            guard,
            change: run.change(),
        },
    })
}
