//! One read of a session produces its clocks, tool unions and time lanes.
//! An append reads only the messages after the last settled one
//! ([`crate::measure::time::derive`]).
use super::{Class, Derivation, Version};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "session_span",
    class: Class::Session,
    products: &[
        polylogue_model::derived::SESSION_SPAN,
        polylogue_model::derived::SESSION_TIME,
    ],
    version: Version::Rules(crate::measure::time::RULE_VERSION),
    after: &["duplicates", "active_path", "tool_calls"],
    dirtied_by: &[],
    run: |tx, pk, inputs| {
        crate::measure::time::derive(tx, pk, inputs.changes, inputs.from_start).map(|_| ())
    },
};
