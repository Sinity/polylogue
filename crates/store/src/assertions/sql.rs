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

/// Whether assertion `row` is in conflict: its family has several current
/// revisions, or another family holds one of its single-value slots with a
/// current revision (each written first where the other was not yet seen,
/// then merged by the stream or an import).
pub fn in_conflict(row: &str) -> String {
    let single_valued: Vec<String> = polylogue_model::assertion::AssertionKind::ALL
        .iter()
        .filter(|kind| {
            kind.spec()
                .uniqueness
                .is_some_and(|uniqueness| uniqueness.when_held == authored::WhenHeld::Revise)
        })
        .map(|kind| format!("'{}'", kind.as_str()))
        .collect();
    format!(
        "((SELECT count(*) FROM source_revision_state ac WHERE ac.root={row}.root AND ac.current=1)>1 \
         OR ({row}.kind IN ({kinds}) AND EXISTS (SELECT 1 FROM source_revision_state oh \
           JOIN source_assertion_slot os ON os.id = oh.id \
           JOIN source_assertion_slot ts ON ts.slot = os.slot \
           JOIN source_assertion ta ON ta.id = ts.id AND ta.root <> {row}.root \
           JOIN source_revision_state th ON th.id = ts.id AND th.current = 1 \
           WHERE oh.root = {row}.root AND oh.current = 1)))",
        kinds = single_valued.join(",")
    )
}
