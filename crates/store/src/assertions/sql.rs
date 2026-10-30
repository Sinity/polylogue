//! SQL consumes the pinned Reader's pure Graph projection; it never computes
//! a competing revision or withdrawal predicate.

pub fn current_assertion(row: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM temp.assertion_currency ac WHERE ac.id={row}.id AND ac.root={row}.root AND ac.current=1)"
    )
}

pub fn withdrawn_assertion(row: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM temp.assertion_currency ac WHERE ac.id={row}.id AND ac.root={row}.root AND ac.withdrawn=1)"
    )
}

pub fn live_assertion(row: &str) -> String {
    current_assertion(row)
}
