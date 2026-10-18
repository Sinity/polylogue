//! Storing input bytes before anything refers to them (architecture 5.2).
//! Hashing and compression run on the parse pool; the writer only records
//! rows.

use polylogue_store::blobs::{Change, FileFacts, Piece};
use std::path::Path;

/// Reads what changed in a file since its last observation: nothing when
/// its facts are unchanged, only the appended bytes when it grew.
pub fn read_change(
    _path: &Path,
    _last: Option<FileFacts>,
) -> std::io::Result<(Change, Option<Piece>)> {
    // Architecture 4.2: stat key, tail window check, appended bytes only.
    todo!()
}
