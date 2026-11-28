//! Opening a sealed read cursor, and the index instance a cursor names,
//! read without its key.
//!
//! The wire form is the shared cursor library's: hex of one version byte, the
//! JSON continuation, and a 32-byte keyed signature. An index instance read here is
//! unauthenticated and only distinguishes a cursor sealed by an earlier index
//! instance from one that was never a cursor.
use query_language::cursor::{
    AuthenticatedCursor, Continuation, CursorContext, CursorSeal, SealedCursor,
};
use query_language::identity::QuestionId;
use query_language::time::ClockWire;
use query_language::{Error, ErrorKind};

const VERSION: u8 = 1;
const SIGNATURE_BYTES: usize = 32;

/// Opens `cursor` with the seal of the index instance `current`. A cursor
/// that does not authenticate is `stale_cursor` when its payload names
/// another index instance (that instance's key went with it), and
/// `invalid_query` otherwise (docs/query.md, Paging, Refusals). The
/// unauthenticated index instance only chooses between the two refusals.
pub fn open_cursor(
    seal: &CursorSeal,
    current: &str,
    cursor: &SealedCursor,
    question: QuestionId,
    resolve_zone: impl FnOnce(&str) -> Result<jiff::tz::TimeZone, Error>,
    available: impl FnOnce(&CursorContext) -> bool,
) -> Result<Continuation, Error> {
    authenticate(seal, current, cursor)?.bind(question, resolve_zone, available)
}

/// Authenticate a cursor before resolving the run that owns its question.
pub fn authenticate(
    seal: &CursorSeal,
    current: &str,
    cursor: &SealedCursor,
) -> Result<AuthenticatedCursor, Error> {
    match seal.authenticate(cursor) {
        Err(error) if error.kind == ErrorKind::InvalidQuery => match sealed_index_instance(cursor) {
            Some(instance) if instance != current => Err(Error::new(
                ErrorKind::StaleCursor,
                "the index was rebuilt since this cursor was issued; run the query again without its cursor",
                0,
            )
            .hint("run the query again without its cursor")),
            _ => Err(error),
        },
        opened => opened,
    }
}

fn sealed_index_instance(cursor: &SealedCursor) -> Option<String> {
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
        .get("index_instance")
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
