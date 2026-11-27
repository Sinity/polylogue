use super::*;

/// Each stage's time is kept in the slot its position in `Stage::ALL` names.
#[test]
fn stage_slots_follow_the_stage_list() {
    for (slot, stage) in Stage::ALL.iter().enumerate() {
        assert_eq!(stage.index(), slot, "{}", stage.name());
    }
    assert_eq!(Stage::ALL.len(), NANOS.len());
}

#[test]
fn stage_seconds_keeps_its_names_and_order() {
    let names: Vec<&str> = stage_seconds()
        .into_iter()
        .map(|(stage, _)| stage.name())
        .collect();
    assert_eq!(
        names,
        [
            "read",
            "database",
            "store",
            "parse",
            "rows",
            "writer",
            "commit",
            "search_index",
            "trigram_index",
            "secondary_indexes",
            "sync",
            "derive",
        ]
    );
}
