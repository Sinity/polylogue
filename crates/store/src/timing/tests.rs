use super::*;

/// Each stage's time is kept in the slot its position in `STAGES` names.
#[test]
fn stage_slots_follow_the_stage_list() {
    for (slot, stage) in STAGES.iter().enumerate() {
        assert_eq!(stage.index(), slot, "{}", stage.name());
    }
    assert_eq!(STAGES.len(), NANOS.len());
}
