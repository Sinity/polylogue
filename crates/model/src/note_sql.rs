//! Predicates over the current note revisions in an attached archive.
/// A note that is its family's current revision and not withdrawn.
pub fn live_note(n: &str) -> String {
    format!("{} AND NOT {}", current_note(n), withdrawn_note(n))
}

pub fn current_note(n: &str) -> String {
    format!(
        "NOT EXISTS (SELECT 1 FROM a.note nr WHERE nr.first_note = {n}.first_note AND nr.commit_seq > {n}.commit_seq)"
    )
}

/// Whether note `n`'s family is withdrawn: a withdrawal names its first note.
pub fn withdrawn_note(n: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM a.note_relation nw WHERE nw.kind = 'withdraws' AND nw.other = 'note:' || {n}.first_note)"
    )
}
