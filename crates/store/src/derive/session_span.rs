//! One read of a session produces its clocks, tool unions and time lanes.
//! An append reads only the messages after the last settled one
//! ([`crate::measure::time::derive`]).
use super::{Class, Derivation, Output, Version};

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
    compute: |conn, pk, inputs| {
        crate::measure::time::derive(conn, pk, inputs.from_start)
            .map(|(derived, _)| Output::Span(Box::new(derived)))
    },
};
