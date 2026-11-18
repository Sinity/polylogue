//! SQL lowering of the same revision rule used by `heads` and
//! authored::Currency: an unwithdrawn revision is current only when no
//! transitive reviser is unwithdrawn. A withdrawn intermediate is transparent.

/// Withdrawal assertions have their own computed state. A superseded but
/// unwithdrawn withdrawal still has effect, so `current` is deliberately absent.
pub(crate) fn withdrawn_sql(variant: &str) -> String {
    format!(
        "EXISTS(SELECT 1 FROM source_link variant_withdrawal \
         JOIN source_revision_state variant_withdrawal_state ON variant_withdrawal_state.id=variant_withdrawal.record AND variant_withdrawal_state.withdrawn=0 \
         WHERE variant_withdrawal.role='withdraws' AND \
         (variant_withdrawal.ref='variant:'||{variant}.root OR variant_withdrawal.ref='variant:'||{variant}.root||'@'||{variant}.id))"
    )
}

/// Both recursive arms are bounded by the existing indexed variant root.
/// UNION and the final self exclusion mirror Currency's visited-revision set.
pub(crate) fn current_sql(variant: &str) -> String {
    format!(
        "(NOT ({}) AND NOT EXISTS( \
         WITH RECURSIVE variant_revisers(id) AS ( \
           SELECT next.id FROM source_variant next WHERE next.root={variant}.root AND next.base={variant}.id \
           UNION SELECT next.id FROM source_variant next JOIN variant_revisers prior ON next.base=prior.id WHERE next.root={variant}.root \
         ) SELECT 1 FROM variant_revisers descendant JOIN source_variant variant_candidate ON variant_candidate.id=descendant.id \
         WHERE variant_candidate.id<>{variant}.id AND NOT ({})))",
        withdrawn_sql(variant),
        withdrawn_sql("variant_candidate")
    )
}
