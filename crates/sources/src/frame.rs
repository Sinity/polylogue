//! Record framing. This module splits input into records, decodes each, keeps
//! its byte range and counts what the provider did with it. A provider sees
//! one record at a time and answers with a [`RecordOutcome`]; ignoring a record
//! needs a kind from the provider's own `Ignored` enum, so every record is
//! accounted for.

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
}

/// The per-record interface every provider implements (rule R88).
pub trait RecordParser {
    type Ignored: IgnoredKind;
    type Output;
    /// A state lookup or write failure is an error, never an unknown record.
    type Error;
    fn record(&mut self, rec: &Record<'_>) -> Result<RecordOutcome<Self::Ignored>, Self::Error>;
    fn finish(self) -> Result<Self::Output, Self::Error>;
}

/// Every record framed, and what became of it.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Text repairs, counted separately from record outcomes.
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
    /// Every framed record has exactly one outcome (law 120).
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

/// The attempt failed; `finish` returns the error. Later `advance` calls do
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FailedAttempt;

pub struct Framed<O, E> {
    pub output: Result<O, ParseFailure<E>>,
    /// Counts for the records before any failure. The failed record is not
    /// counted, and a failed attempt's state must not be committed.
    pub counts: Counts,
    /// Offset just past the last complete line; a partial last line waits.
    pub parsed_upto: u64,
}

/// How much one `advance` call may do. `record_bytes` caps the buffer for one
/// record. A larger record stays pending: the caller must raise the allowance
/// or report a typed failure, never skip or cut the record.
#[derive(Clone, Copy, Debug)]
pub struct StepBudget {
    pub bytes: usize,
    pub records: u64,
    pub record_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepState {
    /// Every supplied byte was scanned (not necessarily the end of input).
    Exhausted,
    /// The byte or record budget ran out.
    Yielded,
    /// The next record needs a larger `record_bytes`. Nothing of it was parsed.
    NeedsCapacity { required: usize },
    /// Allocating the record buffer failed. Nothing of the record was parsed.
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

/// Framing state carried from one committed step to the next, including the
/// buffered partial record. It cannot be cloned. The caller keeps it with the
/// committed parser state it belongs to and counts its buffer capacity.
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

/// A finished step waiting for the writer. The continuation is released only by
/// `commit_with`; if the writer fails, the step is dropped and the next attempt
/// starts again from the last committed state.
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

    /// Transform the output and keep the continuation. A failed transform drops
    /// the step.
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

    /// `writer` must commit the parser state, output and progress in one
    /// transaction before returning `Ok`; only then is the continuation
    /// returned.
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

/// Feeds JSONL to one provider parser in budgeted steps. Each complete record
/// reaches `RecordParser::record` exactly once; a partial last line is kept in
/// a buffer whose capacity the caller can read and charge.
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

    /// Continue after a committed step, with a parser over the committed state.
    /// Counts start at zero; the partial-record buffer is moved, not copied.
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
            // Vec::extend can double capacity past the allowance; reserve exactly.
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

    /// End the run. An incomplete last line is not decoded or counted.
    pub fn finish(self) -> Framed<P::Output, P::Error> {
        self.finish_parts().0
    }

    /// End this step and keep an incomplete last line for the next one. On a
    /// parser error the continuation is dropped.
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
        // After a record error the parser state is unreliable: skip `finish`.
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

/// Parse a whole input with the same driver. An unfinished tail is ignored.
pub fn drive_jsonl<P: RecordParser>(
    parser: P,
    bytes: &[u8],
    base: u64,
) -> Framed<P::Output, P::Error> {
    whole_jsonl_driver(parser, bytes, base).finish()
}

fn whole_jsonl_driver<P: RecordParser>(parser: P, bytes: &[u8], base: u64) -> JsonlDriver<P> {
    let mut driver = JsonlDriver::new(parser, base);
    // Only pass the complete lines, so the unfinished tail is never copied.
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

/// Frames a JSON array of records (an account export's `conversations.json`).
/// Each element is one record with its byte range. A document that is not an
/// array is one undecodable record at offset 0.
pub fn drive_json_array<P: RecordParser>(parser: P, bytes: &[u8]) -> Framed<P::Output, P::Error> {
    match serde_json::from_slice::<Vec<&serde_json::value::RawValue>>(bytes) {
        Ok(items) => drive_raw_records(parser, bytes, &items),
        Err(_) => undecodable_document(parser, bytes),
    }
}

/// Frames the elements of one member array of a JSON object (a Gemini CLI
/// session's `messages`). The provider gets the other members when it is built.
/// A document without that array is one undecodable record.
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

/// Decode and count each raw record borrowed from `bytes`.
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

/// One row of a provider database snapshot, its columns as a JSON object.
#[derive(Debug)]
pub struct Row {
    pub table: &'static str,
    pub rowid: i64,
    pub value: Value,
}

/// Frames provider database rows, given table by table in rowid order. Each row
/// is one record at position `(table, rowid)`.
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
mod tests;
