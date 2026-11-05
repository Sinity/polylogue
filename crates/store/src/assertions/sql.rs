//! SQL over the stored currency index (`revision_state`), which the writer
//! keeps equal to the descendant rule for every root it touches.

pub fn current_assertion(row: &str) -> String {
    format!("EXISTS (SELECT 1 FROM source_revision_state rs WHERE rs.id={row}.id AND rs.current=1)")
}

pub fn withdrawn_assertion(row: &str) -> String {
    format!(
        "EXISTS (SELECT 1 FROM source_revision_state rs WHERE rs.id={row}.id AND rs.withdrawn=1)"
    )
}

pub fn live_assertion(row: &str) -> String {
    current_assertion(row)
}
