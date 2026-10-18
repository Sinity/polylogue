//! The parser seam. A parser maps one record to rows, carrying a small
//! state between records; [`run`] does everything around it: framing,
//! JSON decoding, accounting and the count check.

use crate::account::{Accounting, CountMismatch, Disposition};
use crate::frame::{self, Record};
use polylogue_model::Json;
pub use polylogue_model::{BackgroundUpdate, InlineBlob, MessageRow, Output};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

pub trait Parser {
    /// State carried between records and between appends. Small by design
    /// (at most a few hundred KB on the real archive).
    type Carry: Default + Clone + Serialize + DeserializeOwned;

    /// The kind name accounting uses for a record (`user`, `progress`).
    fn kind(&self, value: &Json) -> String;

    fn record(
        &self,
        carry: &mut Self::Carry,
        record: Record<'_>,
        value: &Json,
        out: &mut Output,
        acc: &mut Accounting,
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
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
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
    let mut acc = Accounting {
        framed: records.len() as u64,
        ..Accounting::default()
    };
    for record in records {
        match serde_json::from_slice::<Json>(record.bytes) {
            Ok(value) => {
                let kind = parser.kind(&value);
                let disposition = parser.record(carry, record, &value, &mut output, &mut acc);
                acc.add(&kind, &disposition);
            }
            Err(e) => acc.add("?", &Disposition::refused(format!("malformed JSON: {e}"))),
        }
    }
    parser.flush(carry, &mut output);
    acc.check()?;
    Ok(Step {
        output,
        accounting: acc,
        consumed: consumed as u64,
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
