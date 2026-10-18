//! The framework that owns iteration (`spec/sources/README.md`, "The framework
//! owns iteration; a provider accounts for every record").
//!
//! The framework frames the input, decodes each record, keeps its byte range
//! and counts every outcome. A provider sees one record at a time and must
//! answer with an [`RecordOutcome`]; ignoring a record requires a kind from the
//! provider's own closed `Ignored` type, so no record can disappear unseen.

use std::collections::BTreeMap;
use std::fmt::Debug;

use polylogue_model::{RawJson as Value, RawJsonError, TextRepairs};

/// One framed, decoded record.
#[derive(Debug)]
pub struct Record<'a> {
    pub bytes: &'a [u8],
    pub value: Value,
    /// Byte range of the record in the input (end exclusive, newline excluded).
    /// For a SQLite row, `start` and `end` are both the row's rowid.
    pub start: u64,
    pub end: u64,
    /// For a SQLite row, its table; empty for records framed from bytes.
    pub table: &'a str,
}

pub use polylogue_model::types::RecordOutcome;

/// A provider's closed list of record kinds it ignores.
pub trait IgnoredKind: Copy + Debug {
    fn name(self) -> &'static str;
    fn reason(self) -> &'static str;
}

/// The per-record interface every provider implements (rule R88).
pub trait RecordParser {
    type Ignored: IgnoredKind;
    type Output;
    /// Pure adapters use Infallible. A state lookup/write failure is an error,
    /// never an unknown record or successful end of input.
    type Error;
    fn record(&mut self, rec: &Record<'_>) -> Result<RecordOutcome<Self::Ignored>, Self::Error>;
    fn finish(self) -> Result<Self::Output, Self::Error>;
}

/// Every record framed, and what became of it.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Additional interpreted-text diagnostics, separate from exclusive outcomes.
    pub text_repairs: TextRepairs,
    pub framed: u64,
    pub used: u64,
    pub ignored: BTreeMap<&'static str, u64>,
    /// (what, byte offset) of each unrecognized record.
    pub unrecognized: Vec<(&'static str, u64)>,
    /// Byte offsets of records that did not decode as JSON.
    pub undecodable: Vec<u64>,
}

impl Counts {
    /// The accounting identity law 120 checks.
    pub fn balanced(&self) -> bool {
        let ignored: u64 = self.ignored.values().sum();
        self.used + ignored + self.unrecognized.len() as u64 + self.undecodable.len() as u64
            == self.framed
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailurePhase {
    Decoding,
    Record,
    Finalization,
}

#[derive(Debug)]
pub enum ParseError<E> {
    Provider(E),
    Decoding(RawJsonError),
}
impl<E: std::fmt::Display> std::fmt::Display for ParseError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Provider(e) => e.fmt(f),
            Self::Decoding(e) => std::fmt::Display::fmt(e, f),
        }
    }
}
#[derive(Debug)]
pub struct ParseFailure<E> {
    pub offset: u64,
    pub phase: FailurePhase,
    pub error: ParseError<E>,
}

impl<E: std::fmt::Display> std::fmt::Display for ParseFailure<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} failed at {}: {}",
            self.phase, self.offset, self.error
        )
    }
}

/// The attempt has failed. Its detailed error stays owned by the driver until
/// finish returns it. Further advance calls cannot invoke the provider again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailedAttempt;

pub struct Framed<O, E> {
    pub output: Result<O, ParseFailure<E>>,
    /// Completed prefix accounting. On failure, the offending record remains
    /// owed and is identified by output's error. These diagnostic counts and
    /// position do not authorize committing a failed attempt's ephemeral state.
    pub counts: Counts,
    /// Offset just past the last complete line; a partial last line waits.
    pub parsed_upto: u64,
}

/// A scheduling allowance, not a provider format limit. `record_bytes` is the
/// caller's currently admitted assembled-record bytes and retained byte capacity. A record that
/// needs more capacity remains owed; the caller must acquire a larger allowance
/// or report an explicit admission failure, never skip or truncate that record.
#[derive(Clone, Copy, Debug)]
pub struct StepBudget {
    pub bytes: usize,
    pub records: u64,
    pub record_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    /// Every supplied byte was scanned. This does not certify input EOF.
    Exhausted,
    /// The byte or record scheduling allowance was used.
    Yielded,
    /// The next record exceeds its current admission. No part of that record
    /// was interpreted; already retained bytes and accounting remain intact.
    NeedsCapacity { required: usize },
    /// Allocation failed before interpreting the next record. The caller must
    /// report or retry admission explicitly; progress never skips the record.
    AllocationFailed { required: usize },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// How much of the supplied slice was consumed, including a partial record.
    pub consumed: usize,
    /// Nonblank records interpreted or accounted as undecodable this step.
    pub records: u64,
    pub state: StepState,
}

/// Framing state moved between committed provider steps. This value is linear:
/// it cannot be cloned, and the previous provider no longer owns its partial
/// record. The caller must bind it to the writer's committed interpretation
/// predecessor and charge its retained capacity for its entire lifetime.
pub struct JsonlContinuation {
    parsed_upto: u64,
    scanned_upto: u64,
    partial: Vec<u8>,
}
impl JsonlContinuation {
    pub fn parsed_upto(&self) -> u64 {
        self.parsed_upto
    }
    pub fn scanned_upto(&self) -> u64 {
        self.scanned_upto
    }
    pub fn buffer_capacity(&self) -> usize {
        self.partial.capacity()
    }
}

/// A provider's completed step, before its state/output/checkpoint transaction.
/// Finishing provider interpretation alone does not release resumable framing
/// state. A failed writer consumes and discards that state; retry reconstructs
/// from the last committed predecessor instead of retaining uncommitted effects.
/// This protocol does not establish that the writer's row finalization is done.
pub struct PendingJsonlStep<O> {
    output: O,
    counts: Counts,
    continuation: JsonlContinuation,
}
impl<O> PendingJsonlStep<O> {
    pub fn parsed_upto(&self) -> u64 {
        self.continuation.parsed_upto
    }
    pub fn scanned_upto(&self) -> u64 {
        self.continuation.scanned_upto
    }
    pub fn counts(&self) -> &Counts {
        &self.counts
    }
    pub fn buffer_capacity(&self) -> usize {
        self.continuation.partial.capacity()
    }

    /// Encode/transform the provider output without releasing its linear
    /// framing continuation. A failed transform consumes the attempt; retry
    /// reconstructs from the committed predecessor. Publication still belongs
    /// to commit_with after every staged member has actually committed.
    pub fn try_map_output<U, E>(
        self,
        transform: impl FnOnce(O, &Counts, u64, u64) -> Result<U, E>,
    ) -> Result<PendingJsonlStep<U>, E> {
        let Self {
            output,
            counts,
            continuation,
        } = self;
        let output = transform(
            output,
            &counts,
            continuation.parsed_upto,
            continuation.scanned_upto,
        )?;
        Ok(PendingJsonlStep {
            output,
            counts,
            continuation,
        })
    }

    /// The owning writer validates the pinned predecessor and atomically commits
    /// state, output and progress inside this callback. A continuation is returned
    /// only after success. The callback must not return success before committing.
    pub fn commit_with<T, E>(
        self,
        writer: impl FnOnce(O, &Counts, u64, u64) -> Result<T, E>,
    ) -> Result<(T, JsonlContinuation), E> {
        let Self {
            output,
            counts,
            continuation,
        } = self;
        let receipt = writer(
            output,
            &counts,
            continuation.parsed_upto,
            continuation.scanned_upto,
        )?;
        Ok((receipt, continuation))
    }
}

/// Retains the same provider parser across framing turns. Scheduling a turn does
/// not call `RecordParser::finish`, clone its state, or encode a carry. Provider
/// interpretation is unchanged: one decoded record enters `record` exactly once.
///
/// `advance` bounds bytes scanned and record callbacks. Buffer relocation,
/// decoding one admitted record, provider state/output and finalization have their own costs;
/// this driver alone does not establish a bounded owner turn. Its retained byte
/// capacity is observable so the caller can keep charging it across yields.
pub struct JsonlDriver<P: RecordParser> {
    parser: P,
    counts: Counts,
    parsed_upto: u64,
    scanned_upto: u64,
    partial: Vec<u8>,
    failure: Option<ParseFailure<P::Error>>,
}

impl<P: RecordParser> JsonlDriver<P> {
    pub fn new(parser: P, base: u64) -> Self {
        Self {
            parser,
            counts: Counts::default(),
            parsed_upto: base,
            scanned_upto: base,
            partial: Vec::new(),
            failure: None,
        }
    }

    /// Resume only after the previous provider step committed successfully.
    /// The fresh provider reads the matching pinned committed state. Accounting
    /// starts empty for this step; retained partial bytes are moved, not copied.
    pub fn resume(parser: P, continuation: JsonlContinuation) -> Self {
        let JsonlContinuation {
            parsed_upto,
            scanned_upto,
            partial,
        } = continuation;
        Self {
            parser,
            counts: Counts::default(),
            parsed_upto,
            scanned_upto,
            partial,
            failure: None,
        }
    }

    /// The next supplied byte must begin at this absolute position.
    pub fn scanned_upto(&self) -> u64 {
        self.scanned_upto
    }
    /// Only complete lines advance this position; a scanned partial record does not.
    pub fn parsed_upto(&self) -> u64 {
        self.parsed_upto
    }
    pub fn counts(&self) -> &Counts {
        &self.counts
    }
    pub fn buffer_capacity(&self) -> usize {
        self.partial.capacity()
    }

    pub fn failure(&self) -> Option<&ParseFailure<P::Error>> {
        self.failure.as_ref()
    }

    pub fn advance(&mut self, bytes: &[u8], budget: StepBudget) -> Result<Step, FailedAttempt> {
        if self.failure.is_some() {
            return Err(FailedAttempt);
        }
        let mut consumed = 0;
        let mut records = 0;
        if self.partial.capacity() > budget.record_bytes {
            return Ok(Step {
                consumed,
                records,
                state: StepState::NeedsCapacity {
                    required: self.partial.capacity(),
                },
            });
        }
        while consumed < bytes.len() && consumed < budget.bytes && records < budget.records {
            let end = bytes.len().min(budget.bytes);
            let available = &bytes[consumed..end];
            let newline = available.iter().position(|&b| b == b'\n');
            let length = newline.unwrap_or(available.len());
            let required = self.partial.len().saturating_add(length);
            if required > budget.record_bytes {
                return Ok(Step {
                    consumed,
                    records,
                    state: StepState::NeedsCapacity { required },
                });
            }
            // Vec::extend alone may geometrically grow past the admitted
            // allowance. Plan exact growth, and keep the actual capacity visible.
            if (!self.partial.is_empty() || newline.is_none()) && required > self.partial.capacity()
            {
                if self
                    .partial
                    .try_reserve_exact(required - self.partial.len())
                    .is_err()
                {
                    return Ok(Step {
                        consumed,
                        records,
                        state: StepState::AllocationFailed { required },
                    });
                }
                if self.partial.capacity() > budget.record_bytes {
                    return Ok(Step {
                        consumed,
                        records,
                        state: StepState::NeedsCapacity {
                            required: self.partial.capacity(),
                        },
                    });
                }
            }
            match newline {
                Some(_) => {
                    let start = self.parsed_upto;
                    let line = &available[..length];
                    let outcome = if self.partial.is_empty() {
                        account_json(&mut self.parser, &mut self.counts, line, start)
                    } else {
                        self.partial.extend_from_slice(line);
                        account_json(&mut self.parser, &mut self.counts, &self.partial, start)
                    };
                    match outcome {
                        Ok(n) => records += n,
                        Err(error) => {
                            self.failure = Some(ParseFailure {
                                offset: start,
                                phase: if matches!(error, ParseError::Decoding(_)) {
                                    FailurePhase::Decoding
                                } else {
                                    FailurePhase::Record
                                },
                                error,
                            });
                            return Err(FailedAttempt);
                        }
                    }
                    self.partial.clear();
                    consumed += length + 1;
                    self.scanned_upto += (length + 1) as u64;
                    self.parsed_upto = self.scanned_upto;
                }
                None => {
                    self.partial.extend_from_slice(available);
                    consumed += length;
                    self.scanned_upto += length as u64;
                }
            }
        }
        Ok(Step {
            consumed,
            records,
            state: if consumed == bytes.len() {
                StepState::Exhausted
            } else {
                StepState::Yielded
            },
        })
    }

    /// Explicit end of this parser run. An incomplete last line is not decoded
    /// or counted, matching the provider append contract. EOF of a supplied
    /// window is not sufficient reason to call this method.
    pub fn finish(self) -> Framed<P::Output, P::Error> {
        self.finish_parts().0
    }

    /// End this keyed provider step, retaining an incomplete line for the next
    /// committed step. Unlike input EOF, this may happen at a scheduling yield.
    /// The provider's finish must itself qualify touched-state/output costs.
    /// Callback or provider-finalization errors drop the continuation entirely.
    pub fn checkpoint(self) -> Result<PendingJsonlStep<P::Output>, Framed<P::Output, P::Error>> {
        let (framed, continuation) = self.finish_parts();
        let Framed {
            output,
            counts,
            parsed_upto,
        } = framed;
        match output {
            Ok(output) => Ok(PendingJsonlStep {
                output,
                counts,
                continuation,
            }),
            Err(error) => Err(Framed {
                output: Err(error),
                counts,
                parsed_upto,
            }),
        }
    }

    fn finish_parts(self) -> (Framed<P::Output, P::Error>, JsonlContinuation) {
        // A callback may have changed ephemeral state before failing. Drop it
        // without finalization; reconstruct from committed state on failure.
        let framed = match self.failure {
            Some(error) => Framed {
                output: Err(error),
                counts: self.counts,
                parsed_upto: self.parsed_upto,
            },
            None => finish_framed(self.parser, self.counts, self.parsed_upto),
        };
        (
            framed,
            JsonlContinuation {
                parsed_upto: self.parsed_upto,
                scanned_upto: self.scanned_upto,
                partial: self.partial,
            },
        )
    }
}

fn account_json<P: RecordParser>(
    parser: &mut P,
    counts: &mut Counts,
    line: &[u8],
    start: u64,
) -> Result<u64, ParseError<P::Error>> {
    if line.iter().all(u8::is_ascii_whitespace) {
        return Ok(0);
    }
    let value = match Value::parse_with_repairs(line, start, &counts.text_repairs) {
        Ok(value) => value,
        Err(error @ RawJsonError::Capacity { .. }) => return Err(ParseError::Decoding(error)),
        Err(_) => {
            counts.framed += 1;
            counts.undecodable.push(start);
            return Ok(1);
        }
    };
    let rec = Record {
        bytes: line,
        value,
        start,
        end: start + line.len() as u64,
        table: "",
    };
    let outcome = parser.record(&rec).map_err(ParseError::Provider)?;
    counts.framed += 1;
    account_outcome(counts, outcome, start);
    Ok(1)
}

fn account_outcome<I: IgnoredKind>(counts: &mut Counts, outcome: RecordOutcome<I>, start: u64) {
    match outcome {
        RecordOutcome::Used => counts.used += 1,
        RecordOutcome::Ignored(k) => *counts.ignored.entry(k.name()).or_default() += 1,
        RecordOutcome::Unrecognized(what) => counts.unrecognized.push((what, start)),
    }
}

/// Whole-run wrapper over the same resumable framing route. Complete records
/// and an unfinished tail stay borrowed from this one window.
pub fn drive_jsonl<P: RecordParser>(
    parser: P,
    bytes: &[u8],
    base: u64,
) -> Framed<P::Output, P::Error> {
    whole_jsonl_driver(parser, bytes, base).finish()
}

fn whole_jsonl_driver<P: RecordParser>(parser: P, bytes: &[u8], base: u64) -> JsonlDriver<P> {
    let mut driver = JsonlDriver::new(parser, base);
    // A whole borrowed window will not resume its incomplete tail, so do not
    // make an owned copy just to drop it at finish. Interpretation still runs
    // through the same continuation driver over the complete-line prefix.
    let complete = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |p| p + 1);
    let _step = driver.advance(
        &bytes[..complete],
        StepBudget {
            bytes: complete,
            records: u64::MAX,
            record_bytes: usize::MAX,
        },
    );
    driver
}

/// Frames a JSON document that is an array of records (an account export's
/// `conversations.json`): each element is one record, with its exact byte range
/// in the input, taken from the borrowed raw value. A document that does not
/// parse as an array is one undecodable record at offset 0.
pub fn drive_json_array<P: RecordParser>(parser: P, bytes: &[u8]) -> Framed<P::Output, P::Error> {
    match serde_json::from_slice::<Vec<&serde_json::value::RawValue>>(bytes) {
        Ok(items) => drive_raw_records(parser, bytes, &items),
        Err(_) => undecodable_document(parser, bytes),
    }
}

/// Frames the records of a JSON document whose records sit in one member array
/// (a Gemini CLI session's `messages`): the provider was given the document's
/// other members when it was built; each array element is one record with its
/// exact byte range. A document without that array is one undecodable record.
pub fn drive_json_member_array<P: RecordParser>(
    parser: P,
    bytes: &[u8],
    member: &str,
) -> Framed<P::Output, P::Error> {
    let doc = serde_json::from_slice::<
        std::collections::BTreeMap<String, &serde_json::value::RawValue>,
    >(bytes);
    let items = doc.ok().and_then(|m| {
        m.get(member).and_then(|raw| {
            serde_json::from_str::<Vec<&serde_json::value::RawValue>>(raw.get()).ok()
        })
    });
    match items {
        Some(items) => drive_raw_records(parser, bytes, &items),
        None => undecodable_document(parser, bytes),
    }
}

fn undecodable_document<P: RecordParser>(parser: P, bytes: &[u8]) -> Framed<P::Output, P::Error> {
    let counts = Counts {
        framed: 1,
        undecodable: vec![0],
        ..Counts::default()
    };
    finish_framed(parser, counts, bytes.len() as u64)
}

fn finish_framed<P: RecordParser>(
    parser: P,
    counts: Counts,
    parsed_upto: u64,
) -> Framed<P::Output, P::Error> {
    let output = parser.finish().map_err(|error| ParseFailure {
        offset: parsed_upto,
        phase: FailurePhase::Finalization,
        error: ParseError::Provider(error),
    });
    Framed {
        output,
        counts,
        parsed_upto,
    }
}

/// A failure at `offset`; the failed record is not counted.
fn failed<O, E>(
    counts: Counts,
    offset: u64,
    phase: FailurePhase,
    error: ParseError<E>,
    parsed_upto: u64,
) -> Framed<O, E> {
    Framed {
        output: Err(ParseFailure {
            offset,
            phase,
            error,
        }),
        counts,
        parsed_upto,
    }
}

/// The shared loop over records borrowed from `bytes`: decodes each, keeps its
/// byte range, and counts every outcome (law 120).
fn drive_raw_records<P: RecordParser>(
    mut parser: P,
    bytes: &[u8],
    items: &[&serde_json::value::RawValue],
) -> Framed<P::Output, P::Error> {
    let mut counts = Counts::default();
    let base = bytes.as_ptr() as usize;
    for raw in items {
        let text = raw.get();
        let start = (text.as_ptr() as usize).saturating_sub(base) as u64;
        counts.framed += 1;
        let value = match Value::parse_with_repairs(text.as_bytes(), start, &counts.text_repairs) {
            Ok(value) => value,
            Err(error @ RawJsonError::Capacity { .. }) => {
                counts.framed -= 1;
                let error = ParseError::Decoding(error);
                return failed(counts, start, FailurePhase::Decoding, error, start);
            }
            Err(_) => {
                counts.undecodable.push(start);
                continue;
            }
        };
        let rec = Record {
            bytes: text.as_bytes(),
            value,
            start,
            end: start + text.len() as u64,
            table: "",
        };
        match parser.record(&rec) {
            Ok(outcome) => account_outcome(&mut counts, outcome, start),
            Err(error) => {
                counts.framed -= 1;
                let error = ParseError::Provider(error);
                return failed(counts, start, FailurePhase::Record, error, start);
            }
        }
    }
    finish_framed(parser, counts, bytes.len() as u64)
}

/// One row of a provider database snapshot, its columns as a JSON object
/// (`spec/sources/README.md`: the framework frames "SQLite rows" too).
#[derive(Debug)]
pub struct Row {
    pub table: &'static str,
    pub rowid: i64,
    pub value: Value,
}

/// Frames the rows of a provider database snapshot, read by the engine from the
/// stored snapshot table by table in rowid order. Each row is one record whose
/// position is `(table, rowid)`; the counts are kept exactly as for bytes (law 120).
pub fn drive_rows<P: RecordParser>(
    mut parser: P,
    rows: impl IntoIterator<Item = Row>,
) -> Framed<P::Output, P::Error> {
    let mut counts = Counts::default();
    let mut last = 0u64;
    for row in rows {
        counts.framed += 1;
        let at = u64::try_from(row.rowid).unwrap_or(0);
        let rec = Record {
            bytes: &[],
            value: row.value,
            start: at,
            end: at,
            table: row.table,
        };
        match parser.record(&rec) {
            Ok(outcome) => account_outcome(&mut counts, outcome, at),
            Err(error) => {
                counts.framed -= 1;
                let error = ParseError::Provider(error);
                return failed(counts, at, FailurePhase::Record, error, last);
            }
        }
        last = last.max(at);
    }
    finish_framed(parser, counts, last)
}

#[cfg(test)]
mod continuation_memory_tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Never {}
    impl IgnoredKind for Never {
        fn name(self) -> &'static str {
            match self {}
        }
        fn reason(self) -> &'static str {
            match self {}
        }
    }
    struct Counter(u64);
    impl RecordParser for Counter {
        type Error = std::convert::Infallible;
        type Ignored = Never;
        type Output = u64;
        fn record(
            &mut self,
            _: &Record<'_>,
        ) -> std::result::Result<RecordOutcome<Self::Ignored>, Self::Error> {
            Ok({
                self.0 += 1;
                RecordOutcome::Used
            })
        }
        fn finish(self) -> std::result::Result<Self::Output, Self::Error> {
            Ok(self.0)
        }
    }

    #[test]
    fn whole_window_does_not_copy_a_large_unfinished_tail() {
        let mut input = b"{}\n".to_vec();
        input.extend(std::iter::repeat_n(b'x', 1 << 20));
        let driver = whole_jsonl_driver(Counter(0), &input, 10);
        assert_eq!(
            driver.buffer_capacity(),
            0,
            "whole-run partial tail must stay borrowed"
        );
        let result = driver.finish();
        assert_eq!(result.output.unwrap(), 1);
        assert_eq!(result.parsed_upto, 13);
    }

    #[test]
    fn incremental_growth_does_not_hide_geometric_spare_capacity() {
        let mut driver = JsonlDriver::new(Counter(0), 0);
        for admitted in 10..=30 {
            let bytes = if admitted == 10 {
                &b"1234567890"[..]
            } else {
                &b"x"[..]
            };
            let step = driver
                .advance(
                    bytes,
                    StepBudget {
                        bytes: bytes.len(),
                        records: 1,
                        record_bytes: admitted,
                    },
                )
                .unwrap();
            assert_eq!(step.state, StepState::Exhausted);
            assert!(
                driver.buffer_capacity() <= admitted,
                "{} retained for {admitted} admitted",
                driver.buffer_capacity()
            );
        }
        let step = driver
            .advance(
                b"\n",
                StepBudget {
                    bytes: 1,
                    records: 1,
                    record_bytes: 1,
                },
            )
            .unwrap();
        assert_eq!(
            step.state,
            StepState::NeedsCapacity {
                required: driver.buffer_capacity()
            }
        );
        assert_eq!(
            step.consumed, 0,
            "retained capacity stays charged across a smaller allowance"
        );
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;
    #[derive(Clone, Copy, Debug)]
    enum Never {}
    impl IgnoredKind for Never {
        fn name(self) -> &'static str {
            match self {}
        }
        fn reason(self) -> &'static str {
            match self {}
        }
    }
    struct Counter {
        n: usize,
        fail_record: bool,
        fail_finish: bool,
    }
    impl Counter {
        fn new(n: usize) -> Self {
            Self {
                n,
                fail_record: false,
                fail_finish: false,
            }
        }
    }
    impl RecordParser for Counter {
        type Error = &'static str;
        type Ignored = Never;
        type Output = usize;
        fn record(&mut self, _: &Record<'_>) -> Result<RecordOutcome<Never>, Self::Error> {
            self.n += 1;
            if self.fail_record {
                Err("record failed")
            } else {
                Ok(RecordOutcome::Used)
            }
        }
        fn finish(self) -> Result<usize, Self::Error> {
            if self.fail_finish {
                Err("finish failed")
            } else {
                Ok(self.n)
            }
        }
    }
    fn budget(bytes: usize) -> StepBudget {
        StepBudget {
            bytes,
            records: 100,
            record_bytes: 1 << 20,
        }
    }
    #[test]
    fn committed_checkpoint_moves_partial_buffer_and_starts_fresh_accounting() {
        let mut driver = JsonlDriver::new(Counter::new(0), 100);
        driver.advance(b"{}\n{\"text\":\"", budget(100)).unwrap();
        let address = driver.partial.as_ptr();
        let pending = match driver.checkpoint() {
            Ok(p) => p,
            Err(_) => panic!("valid step"),
        };
        assert_eq!(pending.parsed_upto(), 103);
        assert!(pending.scanned_upto() > pending.parsed_upto());
        assert_eq!(pending.counts().used, 1);
        let (committed, continuation) = pending
            .commit_with(|output, counts, parsed, scanned| {
                assert_eq!(counts.used, 1);
                assert_eq!(parsed, 103);
                assert!(scanned > parsed);
                Ok::<_, ()>(output)
            })
            .unwrap();
        assert_eq!(
            continuation.partial.as_ptr(),
            address,
            "ownership transfer must not copy a partial line"
        );
        let mut driver = JsonlDriver::resume(Counter::new(committed), continuation);
        assert_eq!(driver.partial.as_ptr(), address);
        assert_eq!(
            driver.counts().framed,
            0,
            "committed prefix is not counted twice"
        );
        driver.advance(b"x\"}\n", budget(100)).unwrap();
        let ready = match driver.checkpoint() {
            Ok(p) => p,
            Err(_) => panic!("valid next step"),
        };
        let (committed, continuation) = ready
            .commit_with(|output, counts, _, _| {
                assert_eq!(counts.framed, 1);
                Ok::<_, ()>(output)
            })
            .unwrap();
        assert_eq!(committed, 2);
        assert_eq!(continuation.parsed_upto(), continuation.scanned_upto());
    }
    #[test]
    fn failed_writer_discards_continuation_and_replays_only_committed_state() {
        let mut driver = JsonlDriver::new(Counter::new(7), 0);
        driver.advance(b"{}\n{", budget(100)).unwrap();
        let ready = match driver.checkpoint() {
            Ok(p) => p,
            Err(_) => panic!("valid step"),
        };
        let result = ready.commit_with(|output, _, _, _| {
            assert_eq!(output, 8);
            Err::<(), _>("CAS rejected")
        });
        assert!(
            matches!(result, Err("CAS rejected")),
            "failed writer must not release a continuation"
        );
        let mut replay = JsonlDriver::new(Counter::new(7), 0);
        replay.advance(b"{}\n{}\n", budget(100)).unwrap();
        assert_eq!(
            replay.finish().output.unwrap(),
            9,
            "discarded uncommitted effects are not replayed twice"
        );
    }
    #[test]
    fn callback_and_finalization_failure_cannot_produce_checkpoint_continuations() {
        for fail_record in [false, true] {
            let mut driver = JsonlDriver::new(
                Counter {
                    n: 0,
                    fail_record,
                    fail_finish: !fail_record,
                },
                50,
            );
            let _ = driver.advance(b"{}\n{", budget(100));
            match driver.checkpoint() {
                Ok(_) => panic!("failed attempt must not expose resumable state"),
                Err(framed) => {
                    let error = framed.output.unwrap_err();
                    assert_eq!(
                        error.phase,
                        if fail_record {
                            FailurePhase::Record
                        } else {
                            FailurePhase::Finalization
                        }
                    );
                    assert_eq!(error.offset, if fail_record { 50 } else { 53 });
                }
            }
        }
    }
}
