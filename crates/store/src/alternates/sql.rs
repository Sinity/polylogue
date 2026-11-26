//! SQL lowering of the same revision rule used by `heads` and
//! authored::Currency: an unwithdrawn revision is current only when no
//! transitive reviser is unwithdrawn. A withdrawn intermediate is transparent.

/// Withdrawal annotations have their own computed state. A superseded but
/// unwithdrawn withdrawal still has effect, so `current` is deliberately absent.
pub(crate) fn withdrawn_sql(alternate: &str) -> String {
    format!(
        "EXISTS(SELECT 1 FROM source_annotation_link alternate_withdrawal \
         JOIN source_revision_state alternate_withdrawal_state ON alternate_withdrawal_state.id=alternate_withdrawal.annotation AND alternate_withdrawal_state.withdrawn=0 \
         WHERE alternate_withdrawal.role='withdraws' AND \
         (alternate_withdrawal.ref='alternate:'||{alternate}.root OR alternate_withdrawal.ref='alternate:'||{alternate}.root||'@'||{alternate}.id))"
    )
}

/// Both recursive arms are bounded by the existing indexed alternate root.
/// UNION and the final self exclusion mirror Currency's visited-revision set.
pub(crate) fn current_sql(alternate: &str) -> String {
    format!(
        "(NOT ({}) AND NOT EXISTS( \
         WITH RECURSIVE alternate_revisers(id) AS ( \
           SELECT next.id FROM source_alternate next WHERE next.root={alternate}.root AND next.base={alternate}.id \
           UNION SELECT next.id FROM source_alternate next JOIN alternate_revisers prior ON next.base=prior.id WHERE next.root={alternate}.root \
         ) SELECT 1 FROM alternate_revisers descendant JOIN source_alternate alternate_candidate ON alternate_candidate.id=descendant.id \
         WHERE alternate_candidate.id<>{alternate}.id AND NOT ({})))",
        withdrawn_sql(alternate),
        withdrawn_sql("alternate_candidate")
    )
}
