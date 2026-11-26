//! The parser interface. A parser maps one record to rows, keeping between
//! records only what is still open; [`run`] does everything around it:
//! splitting into records, JSON decoding, record counts and their check.
//! The rows a record yields depend on the bytes of its chain up to the end
//! of that record and on nothing else: names that belong to the chain are
//! settled afterwards from its rows ([`crate::chain_names`]).

/// Version of the rows this module produces.
pub const VERSION: u32 = 3;

parser_modules! {}

use crate::Records;
use crate::lines::{self, Record};
use polylogue_model::Json;
pub use polylogue_model::{BackgroundUpdate, DocumentRow, ExtractedBlob, MessageRow, Output};
use polylogue_model::{CountMismatch, RecordCounts, RecordOutcome};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub trait Parser {
    /// The parser state: what a parser carries from one record to the next
    /// and between appends of one input, so an append is parsed without
    /// reading the file again. It holds only what is still open (the
    /// message before this one, the open turn's tool calls), never anything
    /// that grows with the bytes read.
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
    fn flush(&self, _parser_state: &mut Self::ParserState, _out: &mut Output) {}
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
    #[error("parser state is unreadable: {0}")]
    ParserState(serde_json::Error),
}

impl From<ParseError> for polylogue_model::ChainFailure {
    fn from(error: ParseError) -> Self {
        match error {
            ParseError::Counts(mismatch) => mismatch.into(),
            error => Self::malformed(error.to_string()),
        }
    }
}

/// Parses the complete records in `bytes`, which start at `base` in an input
/// that may still grow, continuing from `parser_state`.
pub fn run<P: Parser>(
    parser: &P,
    parser_state: &mut P::ParserState,
    bytes: &[u8],
    base: u64,
) -> Result<Step, ParseError> {
    run_to(parser, parser_state, bytes, base, false)
}

/// [`run`], where `finished` says the bytes reach the end of an input that
/// will not grow, so its last line is a record even without a newline.
pub fn run_to<P: Parser>(
    parser: &P,
    parser_state: &mut P::ParserState,
    bytes: &[u8],
    base: u64,
    finished: bool,
) -> Result<Step, ParseError> {
    let (records, consumed) = lines::lines(bytes, base, finished);
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
/// `finished` says the bytes reach the end of an input that will not grow.
pub trait StepParser: Send {
    fn step(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
    ) -> Result<(Step, Json), ParseError>;
}

impl<P: Parser + Send> StepParser for P {
    fn step(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
    ) -> Result<(Step, Json), ParseError> {
        let mut state = match parser_state {
            Some(c) => P::ParserState::deserialize(c).map_err(ParseError::ParserState)?,
            None => P::ParserState::default(),
        };
        let step = run_to(self, &mut state, bytes, base, finished)?;
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
    parser.document(
        crate::document::after_byte_order_mark(bytes),
        &mut output,
        &mut sink,
    );
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

    /// Whether ranges are read in order, so a parse can stop at a record's
    /// end and continue from there (lines or capture frames).
    pub fn reads_in_order(&self) -> bool {
        matches!(self, Parse::Lines(_) | Parse::CaptureLog(_))
    }

    pub fn is_document(&self) -> bool {
        matches!(self, Parse::Document(_))
    }

    /// One step over `bytes`, which start at `base` in an input that may
    /// still grow. A document is given whole, from its first byte, and has
    /// no parser state.
    pub fn step(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
    ) -> Result<(Step, Json), ParseError> {
        self.step_to(parser_state, bytes, base, false)
    }

    /// [`Parse::step`], where `finished` says the bytes reach the end of an
    /// input that will not grow (an import, an export member).
    fn step_to(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
    ) -> Result<(Step, Json), ParseError> {
        match self {
            Parse::Export(_) => Err(ParseError::Export),
            Parse::Lines(p) => p.step(parser_state, bytes, base, finished),
            Parse::CaptureLog(_) => {
                let (mut steps, parser_state) = self.steps(parser_state, bytes, base, finished)?;
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
    /// `finished` says the bytes reach the end of an input that will not grow.
    pub fn steps(
        &self,
        parser_state: Option<&Json>,
        bytes: &[u8],
        base: u64,
        finished: bool,
    ) -> Result<(Vec<Step>, Json), ParseError> {
        match self {
            Parse::CaptureLog(parser) => {
                Ok((capture_steps(&**parser, bytes, base, finished)?, Json::Null))
            }
            _ => self
                .step_to(parser_state, bytes, base, finished)
                .map(|(step, parser_state)| (vec![step], parser_state)),
        }
    }
}

/// The captures in `bytes`, one step each. A frame whose bytes fail its
/// digest is counted unreadable and the next frame read; a header that does
/// not decode ends the framing, so the rest is counted unreadable; at the
/// end of an input that will not grow, an incomplete last frame is counted
/// unreadable. The captures before any of these stay.
fn capture_steps(
    parser: &dyn DocumentParser,
    bytes: &[u8],
    base: u64,
    finished: bool,
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
            match serde_json::from_slice::<serde::de::IgnoredAny>(
                crate::document::after_byte_order_mark(bytes),
            ) {
                Err(error) if error.is_eof() => return empty_steps(),
                _ => {}
            }
            let mut step = run_document(parser, bytes)?;
            mark_whole_capture(&mut step, 0, bytes.len() as u64);
            return Ok(vec![step]);
        }
    }
    let mut steps = Vec::new();
    let start_of = |steps: &Vec<Step>, header_start: usize| {
        if base == 0 && steps.is_empty() {
            0
        } else {
            header_start as u64
        }
    };
    while bytes.len().saturating_sub(position) >= HEADER_LEN {
        let header_start = position;
        let header = match Header::decode(&bytes[position..position + HEADER_LEN]) {
            Ok(header) => header,
            Err(reason) => {
                let start = start_of(&steps, header_start);
                steps.push(unreadable_capture(&reason, start, bytes.len())?);
                return Ok(steps);
            }
        };
        position += HEADER_LEN;
        let end = usize::try_from(header.length)
            .ok()
            .and_then(|length| position.checked_add(length));
        let Some(end) = end.filter(|end| *end <= bytes.len()) else {
            position = header_start;
            break;
        };
        let start = start_of(&steps, header_start);
        if !header.verifies(&bytes[position..end]) {
            steps.push(unreadable_capture("capture digest mismatch", start, end)?);
            position = end;
            continue;
        }
        let mut step = run_document(parser, &bytes[position..end])?;
        mark_whole_capture(&mut step, base + position as u64, base + end as u64);
        step.start = start;
        step.consumed = end as u64;
        steps.push(step);
        position = end;
    }
    if finished && position < bytes.len() {
        let start = start_of(&steps, position);
        steps.push(unreadable_capture(
            "the capture log ends inside a frame",
            start,
            bytes.len(),
        )?);
    }
    if steps.is_empty() {
        return empty_steps();
    }
    Ok(steps)
}

/// A step over capture-log bytes that hold no readable capture: one record,
/// counted unreadable with its reason, and no rows.
fn unreadable_capture(reason: &str, start: u64, end: usize) -> Result<Step, ParseError> {
    let mut records = Records::default();
    records.record(
        "capture",
        &RecordOutcome::unreadable(format!("invalid capture log: {reason}")),
    );
    Ok(Step {
        output: Output::default(),
        counts: records.finish()?,
        consumed: end as u64,
        start,
        repeats_whole_conversations: false,
    })
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
