//! One read of a session produces its clocks, tool unions and time lanes.
use super::{Derivation, Version};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "session_span",
    products: &[polylogue_model::derived::SESSION_SPAN,polylogue_model::derived::SESSION_TIME],
    version: Version::Rules(crate::measure::time::RULE_VERSION),
    after: &["duplicates", "active_path", "tool_calls"],
    run: |tx, pk, _| {
        let (span,lanes)=crate::measure::time::compute(tx,pk,polylogue_model::derived::CLUSTER_GAP_MS)?;
        crate::measure::time::write(tx,pk,&span,&lanes)
    },
    finish: None,
};
