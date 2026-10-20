//! The parser seam. A parser maps one record to rows, carrying a small
//! state between records; [`run`] does everything around it: framing,
//! JSON decoding, accounting and the count check.

use crate::Records;
use crate::frame::{self, Record};
use polylogue_model::Json;
use polylogue_model::{Accounting, CountMismatch, Disposition};
pub use polylogue_model::{BackgroundUpdate, InlineBlob, MessageRow, Output};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub trait Parser {
    /// The carry: state a parser keeps between records and between
    /// appends of one input (open tool calls, the current message), so an
    /// append is parsed without reading the file again. Small by design (at
    /// most a few hundred KB on the real archive).
    type Carry: Default + Clone + Serialize + DeserializeOwned;

    /// The kind name accounting uses for a record (`user`, `progress`).
    fn kind(&self, value: &Json) -> String;

    fn parse_record(
        &self,
        carry: &mut Self::Carry,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> Disposition;

    /// Called when no complete record remains. Sends anything still open
    /// (a message whose continuation may arrive later) as an upsert, and
    /// keeps it open in the carry.
    fn flush(&self, carry: &mut Self::Carry, out: &mut Output);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub output: Output,
    pub accounting: Accounting,
    /// Bytes of the input the step covered; the next step starts there.
    pub consumed: u64,
    /// Start of this step within the supplied range. Normal parses start at 0.
    pub start: u64,
    pub snapshot: Option<polylogue_model::Snapshot>,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("a document must be parsed from its first byte, got offset {0}")]
    NotWhole(u64),
    #[error("array exports require the retained array reader")]
    Array,
    #[error("invalid snapshot stream: {0}")]
    Snapshot(String),
    #[error(transparent)]
    Counts(#[from] CountMismatch),
    #[error("carried state is unreadable: {0}")]
    Carry(serde_json::Error),
}

/// Parses the complete records in `bytes`, which start at `base` in the
/// input, continuing from `carry`.
pub fn run<P: Parser>(
    parser: &P,
    carry: &mut P::Carry,
    bytes: &[u8],
    base: u64,
) -> Result<Step, ParseError> {
    let (records, consumed) = frame::lines(bytes, base);
    let mut output = Output::default();
    let mut sink = Records::lines(records.len() as u64);
    for record in records {
        match serde_json::from_slice::<Json>(record.bytes) {
            Ok(value) => {
                let kind = parser.kind(&value);
                let disposition =
                    parser.parse_record(carry, record, &value, &mut output, &mut sink);
                sink.record(&kind, &disposition);
            }
            Err(e) => sink.record("?", &Disposition::refused(format!("malformed JSON: {e}"))),
        }
    }
    parser.flush(carry, &mut output);
    let accounting = sink.finish()?;
    Ok(Step {
        output,
        accounting,
        consumed: consumed as u64,
        start: 0,
        snapshot: None,
    })
}

/// A parser with its carry stored as JSON, so the engine can drive any
/// origin without knowing its carry type.
pub trait StepParser: Send {
    fn step(
        &self,
        carry: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError>;
}

impl<P: Parser + Send> StepParser for P {
    fn step(
        &self,
        carry: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError> {
        let mut state = match carry {
            Some(c) => P::Carry::deserialize(c).map_err(ParseError::Carry)?,
            None => P::Carry::default(),
        };
        let step = run(self, &mut state, bytes, base)?;
        let carry = serde_json::to_value(&state).map_err(ParseError::Carry)?;
        Ok((step, carry))
    }
}

/// A parser for an input read whole (architecture 5.3): an export, or a
/// capture file the extension rewrites in place. Records are the
/// document's own items (one conversation, one span); the parser frames
/// them and submits each disposition to the shared accounting sink.
pub trait DocumentParser {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records);

    fn snapshot_complete(&self, _bytes: &[u8]) -> bool {
        false
    }
}

/// Parses a complete document. Each step covers the whole input, so there
/// is no carry: a changed document is parsed again from its first byte.
pub fn run_document<D: DocumentParser + ?Sized>(
    parser: &D,
    bytes: &[u8],
) -> Result<Step, ParseError> {
    let mut output = Output::default();
    let mut sink = Records::default();
    parser.document(bytes, &mut output, &mut sink);
    let accounting = sink.finish()?;
    Ok(Step {
        output,
        accounting,
        consumed: bytes.len() as u64,
        start: 0,
        snapshot: None,
    })
}

/// The parser an origin opens for one input. Which variant it is decides
/// how the input's bytes divide into records, so the framing and the
/// parser cannot disagree.
pub enum Parse {
    /// One record per line; appends are parsed from where the last step
    /// stopped.
    Lines(Box<dyn StepParser>),
    /// One retained export array, externally ordered and delivered per element.
    Array(Box<dyn crate::array::ArrayParser>),
    /// One document (or archive) read whole: an export, a capture the
    /// extension rewrites. Any change parses the whole input again.
    Document(Box<dyn DocumentParser + Send>),
    /// Ordered, complete documents in an append-only snapshot log.
    Snapshots(Box<dyn DocumentParser + Send>),
}

impl Parse {
    pub fn is_snapshots(&self) -> bool {
        matches!(self, Parse::Snapshots(_))
    }

    pub fn is_document(&self) -> bool {
        matches!(self, Parse::Document(_))
    }

    /// One step over `bytes`, which start at `base` in the input. A
    /// document is given whole, from its first byte, and has no carry.
    pub fn step(
        &self,
        carry: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError> {
        match self {
            Parse::Array(_) => Err(ParseError::Array),
            Parse::Lines(p) => p.step(carry, bytes, base),
            Parse::Snapshots(_) => {
                let (mut steps, carry) = self.steps(carry, bytes, base)?;
                if steps.len() != 1 {
                    return Err(ParseError::Snapshot(
                        "use steps for a multi-snapshot range".into(),
                    ));
                }
                Ok((steps.remove(0), carry))
            }
            Parse::Document(d) => {
                if base != 0 {
                    return Err(ParseError::NotWhole(base));
                }
                Ok((run_document(&**d, bytes)?, Json::Null))
            }
        }
    }

    /// A range can contain several complete snapshots. Each remains a separate
    /// writer step, so coalesced reads preserve every intermediate version.
    pub fn steps(
        &self,
        carry: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Vec<Step>, Json), ParseError> {
        match self {
            Parse::Snapshots(parser) => Ok((snapshots(&**parser, bytes, base)?, Json::Null)),
            _ => self
                .step(carry, bytes, base)
                .map(|(step, carry)| (vec![step], carry)),
        }
    }
}

fn snapshots(
    parser: &dyn DocumentParser,
    bytes: &[u8],
    base: u64,
) -> Result<Vec<Step>, ParseError> {
    use polylogue_model::snapshot_log::{HEADER_LEN, Header, MAGIC};
    let mut position = 0usize;
    if base == 0 {
        if MAGIC.starts_with(bytes) && bytes.len() < MAGIC.len() {
            return empty_steps();
        }
        if bytes.starts_with(MAGIC) {
            position = MAGIC.len();
        } else {
            // Existing single-envelope spools keep their original input identity.
            match serde_json::from_slice::<serde::de::IgnoredAny>(bytes) {
                Err(error) if error.is_eof() => return empty_steps(),
                _ => {}
            }
            let mut step = run_document(parser, bytes)?;
            mark_snapshot(
                &mut step,
                0,
                bytes.len() as u64,
                parser.snapshot_complete(bytes),
            );
            return Ok(vec![step]);
        }
    }
    let mut steps = Vec::new();
    while bytes.len().saturating_sub(position) >= HEADER_LEN {
        let header_start = position;
        let header = Header::decode(&bytes[position..position + HEADER_LEN])
            .map_err(ParseError::Snapshot)?;
        position += HEADER_LEN;
        let length = usize::try_from(header.length)
            .map_err(|_| ParseError::Snapshot("frame length exceeds address space".into()))?;
        let Some(end) = position
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
        else {
            break;
        };
        if !header.verifies(&bytes[position..end]) {
            return Err(ParseError::Snapshot(
                "snapshot payload digest mismatch".into(),
            ));
        }
        let mut step = run_document(parser, &bytes[position..end])?;
        mark_snapshot(
            &mut step,
            base + position as u64,
            base + end as u64,
            parser.snapshot_complete(&bytes[position..end]),
        );
        step.start = if base == 0 && steps.is_empty() {
            0
        } else {
            header_start as u64
        };
        step.consumed = end as u64;
        steps.push(step);
        position = end;
    }
    if steps.is_empty() {
        return empty_steps();
    }
    Ok(steps)
}

fn mark_snapshot(step: &mut Step, start: u64, end: u64, complete: bool) {
    for row in &mut step.output.messages {
        row.offset = start;
        row.end = end;
    }
    step.snapshot = Some(polylogue_model::Snapshot {
        start,
        end,
        complete,
        covered: step
            .output
            .sessions
            .iter()
            .map(|session| session.id.clone())
            .collect(),
    });
}

fn empty_steps() -> Result<Vec<Step>, ParseError> {
    Ok(vec![Step {
        output: Output::default(),
        accounting: Accounting::default(),
        consumed: 0,
        start: 0,
        snapshot: None,
    }])
}

#[cfg(test)]
mod tests;
