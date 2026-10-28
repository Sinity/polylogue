//! The parser interface. A parser maps one record to rows, keeping a small
//! state between records; [`run`] does everything around it: splitting into records,
//! JSON decoding, record counts and their check.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use crate::Records;
use crate::lines::{self, Record};
use polylogue_model::Json;
pub use polylogue_model::{BackgroundUpdate, DocumentRow, ExtractedBlob, MessageRow, Output};
use polylogue_model::{CountMismatch, RecordCounts, RecordOutcome};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub trait Parser {
    /// The parser state: state a parser keeps between records and between
    /// appends of one input (open tool calls, the current message), so an
    /// append is parsed without reading the file again. Small by design (at
    /// most a few hundred KB on the real archive).
    type ParserState: Default + Clone + Serialize + DeserializeOwned;

    /// The kind name the record counts use for a record (`user`, `progress`).
    fn kind(&self, value: &Json) -> String;

    fn parse_record(
        &self,
        parser_state: &mut Self::ParserState,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        records: &mut Records,
    ) -> RecordOutcome;

    /// Called when no complete record remains. Sends anything still open
    /// (a message whose continuation may arrive later) as an upsert, and
    /// keeps it open in the parser state.
    fn flush(&self, parser_state: &mut Self::ParserState, out: &mut Output);
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub output: Output,
    pub counts: RecordCounts,
    /// Bytes of the input the step covered; the next step starts there.
    pub consumed: u64,
    /// Start of this step within the supplied range. Normal parses start at 0.
    pub start: u64,
    /// Each record repeats whole conversations, so a later record's
    /// version of a message replaces an earlier one in the same chain.
    pub repeats_whole_conversations: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("a document must be parsed from its first byte, got offset {0}")]
    NotWhole(u64),
    #[error("exports require the export item reader")]
    Export,
    #[error("invalid capture log: {0}")]
    CaptureLog(String),
    #[error(transparent)]
    Counts(#[from] CountMismatch),
    #[error("carried state is unreadable: {0}")]
    ParserState(serde_json::Error),
}

/// Parses the complete records in `bytes`, which start at `base` in the
/// input, continuing from `parser_state`.
pub fn run<P: Parser>(
    parser: &P,
    parser_state: &mut P::ParserState,
    bytes: &[u8],
    base: u64,
) -> Result<Step, ParseError> {
    let (records, consumed) = lines::lines(bytes, base);
    let mut output = Output::default();
    let mut sink = Records::lines(records.len() as u64);
    for record in records {
        match serde_json::from_slice::<Json>(record.bytes) {
            Ok(value) => {
                let kind = parser.kind(&value);
                let field_checks = sink.field_checks();
                let event_start = output.events.len();
                let outcome =
                    parser.parse_record(parser_state, record, &value, &mut output, &mut sink);
                if sink.field_checks() == field_checks {
                    let events = &output.events[event_start..];
                    let root_kept = events.iter().any(|event| event.data == value);
                    let payload_kept = value
                        .get("payload")
                        .is_some_and(|payload| events.iter().any(|event| &event.data == payload));
                    sink.remaining_fields(&kind, &value, root_kept, payload_kept);
                }
                sink.record(&kind, &outcome);
            }
            Err(e) => sink.record(
                "?",
                &RecordOutcome::unreadable(format!("malformed JSON: {e}")),
            ),
        }
    }
    parser.flush(parser_state, &mut output);
    let counts = sink.finish()?;
    Ok(Step {
        output,
        counts,
        consumed: consumed as u64,
        start: 0,
        repeats_whole_conversations: false,
    })
}

/// A parser with its parser state stored as JSON, so the engine can drive any
/// origin without knowing its parser state type.
pub trait StepParser: Send {
    fn step(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError>;
}

impl<P: Parser + Send> StepParser for P {
    fn step(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError> {
        let mut state = match parser_state {
            Some(c) => P::ParserState::deserialize(c).map_err(ParseError::ParserState)?,
            None => P::ParserState::default(),
        };
        let step = run(self, &mut state, bytes, base)?;
        let parser_state = serde_json::to_value(&state).map_err(ParseError::ParserState)?;
        Ok((step, parser_state))
    }
}

/// A parser for an input read whole: an export, or a
/// capture file the extension rewrites in place. Records are the
/// document's own items (one conversation, one span); the parser frames
/// them and records each outcome in the shared record counts.
pub trait DocumentParser {
    fn document(&self, bytes: &[u8], out: &mut Output, records: &mut Records);
}

/// Parses a complete document. Each step covers the whole input, so there
/// is no parser state: a changed document is parsed again from its first byte.
pub fn run_document<D: DocumentParser + ?Sized>(
    parser: &D,
    bytes: &[u8],
) -> Result<Step, ParseError> {
    let mut output = Output::default();
    let mut sink = Records::default();
    parser.document(bytes, &mut output, &mut sink);
    // Document parsers emit messages in reading order, not at byte offsets.
    for row in &mut output.messages {
        row.offset = 0;
        row.end = bytes.len() as u64;
    }
    for row in &mut output.documents {
        row.offset = 0;
        row.end = bytes.len() as u64;
    }
    for event in &mut output.events {
        event.range.get_or_insert((0, bytes.len() as u64));
    }
    let counts = sink.finish()?;
    Ok(Step {
        output,
        counts,
        consumed: bytes.len() as u64,
        start: 0,
        repeats_whole_conversations: false,
    })
}

/// The parser an origin opens for one input. Which variant it is decides
/// how the input's bytes divide into records, so the split and the
/// parser cannot disagree.
pub enum Parse {
    /// One record per line; appends are parsed from where the last step
    /// stopped.
    Lines(Box<dyn StepParser>),
    /// Stored export records, externally ordered and delivered one at a time.
    Export(Box<dyn crate::export::ExportParser>),
    /// One document (or archive) read whole: an export, a capture the
    /// extension rewrites. Any change parses the whole input again.
    Document(Box<dyn DocumentParser + Send>),
    /// Ordered, complete documents in an append-only capture log, one per
    /// frame.
    CaptureLog(Box<dyn DocumentParser + Send>),
}

impl Parse {
    pub fn is_capture_log(&self) -> bool {
        matches!(self, Parse::CaptureLog(_))
    }

    pub fn is_document(&self) -> bool {
        matches!(self, Parse::Document(_))
    }

    /// One step over `bytes`, which start at `base` in the input. A
    /// document is given whole, from its first byte, and has no parser state.
    pub fn step(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError> {
        match self {
            Parse::Export(_) => Err(ParseError::Export),
            Parse::Lines(p) => p.step(parser_state, bytes, base),
            Parse::CaptureLog(_) => {
                let (mut steps, parser_state) = self.steps(parser_state, bytes, base)?;
                if steps.len() != 1 {
                    return Err(ParseError::CaptureLog(
                        "use steps for a range holding several captures".into(),
                    ));
                }
                Ok((steps.remove(0), parser_state))
            }
            Parse::Document(d) => {
                if base != 0 {
                    return Err(ParseError::NotWhole(base));
                }
                Ok((run_document(&**d, bytes)?, Json::Null))
            }
        }
    }

    /// A range can hold several complete captures. Each stays a separate
    /// writer step, so reading them together keeps every intermediate version.
    pub fn steps(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Vec<Step>, Json), ParseError> {
        match self {
            Parse::CaptureLog(parser) => Ok((capture_steps(&**parser, bytes, base)?, Json::Null)),
            _ => self
                .step(parser_state, bytes, base)
                .map(|(step, parser_state)| (vec![step], parser_state)),
        }
    }
}

fn capture_steps(
    parser: &dyn DocumentParser,
    bytes: &[u8],
    base: u64,
) -> Result<Vec<Step>, ParseError> {
    use polylogue_model::capture_log::{HEADER_LEN, Header, MAGIC};
    let mut position = 0usize;
    if base == 0 {
        if MAGIC.starts_with(bytes) && bytes.len() < MAGIC.len() {
            return empty_steps();
        }
        if bytes.starts_with(MAGIC) {
            position = MAGIC.len();
        } else {
            // An unframed file (an imported capture) is one capture.
            match serde_json::from_slice::<serde::de::IgnoredAny>(bytes) {
                Err(error) if error.is_eof() => return empty_steps(),
                _ => {}
            }
            let mut step = run_document(parser, bytes)?;
            mark_whole_capture(&mut step, 0, bytes.len() as u64);
            return Ok(vec![step]);
        }
    }
    let mut steps = Vec::new();
    while bytes.len().saturating_sub(position) >= HEADER_LEN {
        let header_start = position;
        let header = Header::decode(&bytes[position..position + HEADER_LEN])
            .map_err(ParseError::CaptureLog)?;
        position += HEADER_LEN;
        let length = usize::try_from(header.length)
            .map_err(|_| ParseError::CaptureLog("frame length exceeds address space".into()))?;
        let Some(end) = position
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
        else {
            break;
        };
        if !header.verifies(&bytes[position..end]) {
            return Err(ParseError::CaptureLog("capture digest mismatch".into()));
        }
        let mut step = run_document(parser, &bytes[position..end])?;
        mark_whole_capture(&mut step, base + position as u64, base + end as u64);
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

fn mark_whole_capture(step: &mut Step, start: u64, end: u64) {
    for row in &mut step.output.messages {
        row.offset = start;
        row.end = end;
    }
    for row in &mut step.output.documents {
        row.offset = start;
        row.end = end;
    }
    for event in &mut step.output.events {
        event.range = Some((start, end));
    }
    step.repeats_whole_conversations = true;
}

fn empty_steps() -> Result<Vec<Step>, ParseError> {
    Ok(vec![Step {
        output: Output::default(),
        counts: RecordCounts::default(),
        consumed: 0,
        start: 0,
        repeats_whole_conversations: false,
    }])
}

#[cfg(test)]
mod tests;
