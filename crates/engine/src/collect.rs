//! Finding inputs at the origins' fixed locations (architecture 5.1).
//! Exactly the layout each tool writes; a recognizer runs before the origin
//! is decided. An unreadable root is a state, never an empty backlog.

use polylogue_sources::OriginSpec;
use std::path::{Path, PathBuf};

pub struct Found {
    pub path: PathBuf,
    pub origin: &'static OriginSpec,
}

/// Every file under the live and spool locations that an origin
/// recognizes, and the roots that could not be read.
pub fn scan(_home: &Path) -> (Vec<Found>, Vec<PathBuf>) {
    // Architecture 5.1: glob locations, recognize by path and head bytes.
    todo!()
}

/// Watches the locations with `notify`; the loop also rescans once a minute.
pub struct Watcher {
    pub inner: notify::RecommendedWatcher,
}
