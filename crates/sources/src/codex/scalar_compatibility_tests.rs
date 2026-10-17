use super::*;
#[test]
fn scalar_scope_refuses_native_surrogates_without_repair() {
    for source in [
        r#"{"id":"\ud800"}"#,
        r#"{"\ud800":"ordinary"}"#,
        r#"["\ud800"]"#,
    ] {
        let value = Value::parse(source.as_bytes()).unwrap();
        assert!(require_scalar_json(&value).is_err());
    }
    let ordinary = Value::parse(br#"{"id":"literal:~u:eda080","n":1e+02}"#).unwrap();
    require_scalar_json(&ordinary).unwrap();
    assert!(
        String::from_utf8(ordinary.canonical())
            .unwrap()
            .contains("1e+02")
    );
}
#[test]
fn embedded_json_keeps_the_existing_scalar_only_decode_scope() {
    assert!(scalar_json(r#"{"s":"\ud800"}"#).is_none());
    let ordinary = scalar_json(r#"{"s":"ordinary","n":1e+02}"#).unwrap();
    assert_eq!(ordinary["s"].as_str(), Some("ordinary"));
    assert!(
        String::from_utf8(ordinary.canonical())
            .unwrap()
            .contains("1e+02")
    );
}
