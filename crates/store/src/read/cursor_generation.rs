//! Opening a sealed read cursor, and the index generation a cursor names,
//! read without its key.
//!
//! The wire form is the shared cursor library's: hex of one version byte, the
//! JSON continuation, and a 32-byte keyed signature. A generation read here is
//! unauthenticated and only distinguishes a cursor sealed by an earlier index
//! generation from one that was never a cursor.
use query_language::cursor::{Continuation, CursorContext, CursorSeal, SealedCursor};
use query_language::identity::QuestionId;
use query_language::time::ClockWire;
use query_language::{Error, ErrorKind};

const VERSION: u8 = 1;
const SIGNATURE_BYTES: usize = 32;

/// Opens `cursor` with the seal of the index generation `current`. A cursor
/// that does not authenticate is `stale_cursor` when its payload names
/// another index generation (that generation's key went with it), and
/// `invalid_query` otherwise (docs/query.md, Paging, Refusals). The
/// unauthenticated generation only chooses between the two refusals.
pub fn open_cursor(
    seal: &CursorSeal,
    current: &str,
    cursor: &SealedCursor,
    question: QuestionId,
    resolve_zone: impl FnOnce(&str) -> Result<jiff::tz::TimeZone, Error>,
    available: impl FnOnce(&CursorContext) -> bool,
) -> Result<Continuation, Error> {
    match seal.open(cursor, question, resolve_zone, available) {
        Err(error) if error.kind == ErrorKind::InvalidQuery => match sealed_generation(cursor) {
            Some(generation) if generation != current => Err(Error::new(
                ErrorKind::StaleCursor,
                "the cursor is from an earlier index generation",
                0,
            )
            .hint("restart the question without its cursor")),
            _ => Err(error),
        },
        opened => opened,
    }
}

fn sealed_generation(cursor: &SealedCursor) -> Option<String> {
    let bytes = unhex(cursor.as_str())?;
    let payload = bytes.get(..bytes.len().checked_sub(SIGNATURE_BYTES)?)?;
    let (&version, body) = payload.split_first()?;
    if version != VERSION {
        return None;
    }
    let continuation: Continuation<ClockWire> = serde_json::from_slice(body).ok()?;
    continuation
        .context
        .read_positions
        .get("generation")
        .cloned()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    text.as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[high, low]| {
            let digit = |byte: u8| (byte as char).to_digit(16);
            Some(((digit(high)? << 4) | digit(low)?) as u8)
        })
        .collect()
}
