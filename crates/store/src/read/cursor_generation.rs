//! The index generation a sealed cursor names, read without its key.
//!
//! The wire form is the shared cursor library's: hex of one version byte, the
//! JSON continuation, and a 32-byte keyed signature. A generation read here is
//! unauthenticated and only distinguishes a cursor sealed by an earlier index
//! generation from one that was never a cursor.
use query_language::cursor::{Continuation, SealedCursor};
use query_language::time::ClockWire;

const VERSION: u8 = 1;
const SIGNATURE_BYTES: usize = 32;

pub(crate) fn sealed_generation(cursor: &SealedCursor) -> Option<String> {
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
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| (byte as char).to_digit(16);
            Some(((digit(pair[0])? << 4) | digit(pair[1])?) as u8)
        })
        .collect()
}
