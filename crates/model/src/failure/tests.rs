use super::*;
#[test]
fn codes_round_trip_the_wire_names() {
    for &code in Code::ALL {
        let text = serde_json::to_value(code).unwrap();
        assert_eq!(text.as_str(), Some(code.as_str()));
        assert_eq!(Code::from_name(code.as_str()), Some(code));
        assert_eq!(serde_json::from_value::<Code>(text).unwrap(), code);
    }
}
