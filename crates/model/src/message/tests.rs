use super::*;

#[test]
fn response_boundary_updates_preserve_absent_and_supplied_states() {
    let mut production = Production {
        response_boundary_unknown: Some(true),
        ..Production::default()
    };
    production.merge(Production::default());
    assert_eq!(production.response_boundary_unknown, Some(true));
    production.merge(Production {
        response_boundary_unknown: Some(false),
        ..Production::default()
    });
    production.merge(Production::default());
    assert_eq!(production.response_boundary_unknown, Some(false));
    let encoded = serde_json::to_value(&production).unwrap();
    assert_eq!(encoded["response_boundary_unknown"], false);
    assert_eq!(
        serde_json::from_value::<Production>(encoded).unwrap(),
        production
    );
    assert!(
        serde_json::to_value(Production::default())
            .unwrap()
            .get("response_boundary_unknown")
            .is_none()
    );
}
