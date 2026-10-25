use super::*;

#[test]
fn a_delivery_is_prepared_until_an_outcome_is_recorded() {
    assert_eq!(DELIVERY_STATES[0], "prepared");
    assert_eq!(&DELIVERY_STATES[1..], DeliveryOutcome::NAMES);
}

#[test]
fn stored_names_are_the_serde_names() {
    for e in DeliveryEvent::ALL {
        assert_eq!(serde_json::to_value(e).unwrap(), e.as_str());
    }
    for s in DeliverySender::ALL {
        assert_eq!(serde_json::to_value(s).unwrap(), s.as_str());
    }
    for o in DeliveryOutcome::ALL {
        assert_eq!(serde_json::to_value(o).unwrap(), o.as_str());
    }
}
