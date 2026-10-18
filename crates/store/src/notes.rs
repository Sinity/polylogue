//! Notes and assertions in `archive.db`: one store with revisions and
//! relations. Agent-written notes are stored with their author and never
//! selected for automatic context.

use crate::Result;
use polylogue_model::{Note, Target};

pub fn add(_note: &Note) -> Result<i64> {
    // Architecture 7: notes and assertions; request IDs make a resend a no-op.
    todo!()
}

pub fn for_target(_reader: &crate::Reader<'_>, _target: &Target) -> Result<Vec<Note>> {
    // Architecture 7: current revision of each note on a target.
    todo!()
}
