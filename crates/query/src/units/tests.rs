use super::*;
use std::collections::HashSet;

/// Every variant has exactly one declaration, and names are distinct, so
/// `from_name` has one answer. The exhaustive match fails to compile when a
/// variant is added without being listed here.
#[test]
fn every_unit_is_declared_once_under_its_own_name() {
    let listed = [
        Unit::Sessions,
        Unit::Documents,
        Unit::Messages,
        Unit::Blocks,
        Unit::ToolCalls,
        Unit::Files,
        Unit::Events,
        Unit::Annotations,
        Unit::Alternates,
        Unit::Queries,
        Unit::Runs,
        Unit::Results,
        Unit::SessionLinks,
        Unit::Turns,
        Unit::Deliveries,
        Unit::Feed,
        Unit::LiveIntervals,
        Unit::Processes,
    ];
    for unit in listed {
        match unit {
            Unit::Sessions
            | Unit::Documents
            | Unit::Messages
            | Unit::Blocks
            | Unit::ToolCalls
            | Unit::Files
            | Unit::Events
            | Unit::Annotations
            | Unit::Alternates
            | Unit::Queries
            | Unit::Runs
            | Unit::Results
            | Unit::SessionLinks
            | Unit::Turns
            | Unit::Deliveries
            | Unit::Feed
            | Unit::LiveIntervals
            | Unit::Processes => {}
        }
        assert_eq!(
            UNITS.iter().filter(|s| s.unit == unit).count(),
            1,
            "{unit:?}"
        );
    }
    assert_eq!(UNITS.len(), listed.len());
    let names: HashSet<_> = UNITS.iter().map(|s| s.name).collect();
    assert_eq!(names.len(), UNITS.len());
}
