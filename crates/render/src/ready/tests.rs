use super::*;
use serde_json::json;

#[test]
fn explicit_null_is_a_wire_value_and_absence_alone_omits_a_field() {
    let supplied = HttpCall {
        method: "GET".into(),
        path: "/api/v1/read".into(),
        arguments: json!({"deadline":null,"nullable":null,"flag":false,"zero":0}),
    };
    let url = http_url(&supplied).unwrap();
    for field in ["deadline=null", "nullable=null", "flag=false", "zero=0"] {
        assert!(
            url.split_once('?')
                .unwrap()
                .1
                .split('&')
                .any(|part| part == field),
            "{url}"
        );
    }
    let mut omitted = supplied;
    omitted
        .arguments
        .as_object_mut()
        .unwrap()
        .remove("deadline");
    let url = http_url(&omitted).unwrap();
    assert!(!url.contains("deadline="), "{url}");
    assert!(url.contains("nullable=null"), "{url}");
}
