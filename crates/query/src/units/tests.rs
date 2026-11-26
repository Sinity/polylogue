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
        Unit::ImportJobs,
        Unit::ChatActions,
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
            | Unit::Processes
            | Unit::ImportJobs
            | Unit::ChatActions => {}
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

/// docs/query.md#units-and-fields lists every unit; the registry declares
/// exactly those, built or designed.
#[test]
fn designed_units_match_the_query_page() {
    let page = include_str!("../../../../docs/query.md");
    let table = page
        .split("| Unit | Rows |")
        .nth(1)
        .expect("the page has its unit table");
    let mut listed: Vec<&str> = table
        .lines()
        .skip(2)
        .take_while(|line| line.starts_with('|'))
        .flat_map(|line| line.split('|').nth(1).unwrap().split(','))
        .map(|name| name.trim().trim_matches('`'))
        .collect();
    listed.sort_unstable();
    let mut declared: Vec<&str> = UNITS
        .iter()
        .map(|unit| unit.name)
        .chain(DESIGNED_UNITS.iter().map(|unit| unit.name))
        .collect();
    declared.sort_unstable();
    assert_eq!(listed, declared);
}
