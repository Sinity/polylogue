//! Context and briefings (architecture 7, 12). One scheduler is the only
//! admission point for automatic context. The session-start brief is
//! injected by default and covers recent sessions and handoffs, with the
//! delta since the last brief on resume. Agent-written notes are never
//! injected. Every delivery records the exact bytes delivered.

use polylogue_model::SessionId;
use std::path::Path;

pub struct Brief {
    pub text: String,
    pub delivery_id: String,
}

pub fn session_start(_cwd: &Path, _session: Option<&SessionId>) -> Brief {
    // Architecture 7: one scheduler, delivery receipts, no size cap.
    todo!()
}
