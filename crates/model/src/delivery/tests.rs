use super::*;

#[test]
fn delivery_states_are_prepared_then_the_outcome_names() {
    assert_eq!(
        DELIVERY_STATES,
        ["prepared", "delivered", "failed", "not_sent", "used"]
    );
    assert_eq!(&DELIVERY_STATES[1..], DeliveryOutcome::NAMES);
}
