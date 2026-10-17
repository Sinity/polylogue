use super::*;
use std::time::Duration;
fn until() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
fn raw(source: &str) -> RetainedJson {
    RetainedJson::parse(source.as_bytes()).unwrap()
}

#[test]
fn envelopes_preserve_numbers_surrogates_keys_and_presence() {
    for source in [
        "null",
        "{}",
        r#"{"n":1e+02,"s":"\ud800","\ud801":-0}"#,
        r#"{"literal":"~ud800","escaped":"\\ud800"}"#,
    ] {
        let value = raw(source);
        let page = envelope(&value, until(), false).unwrap();
        assert_eq!(page["encoding"], "canonical-json-v1");
        assert_eq!(page["masked"], false);
        assert_eq!(raw(page["json"].as_str().unwrap()), value);
        let decoded: Value = serde_json::from_slice(&serde_json::to_vec(&page).unwrap()).unwrap();
        assert_eq!(decoded, page);
    }
}
#[test]
fn masks_logical_strings_once_without_crossing_structured_boundaries() {
    let token = format!("ghp_{}", "A".repeat(36));
    let value = RetainedJson::parse(
        serde_json::to_vec(&json!({"a":token,"b":"ordinary"}))
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    let page = envelope(&value, until(), false).unwrap();
    assert_eq!(page["masked"], true);
    let parsed: Value = serde_json::from_str(page["json"].as_str().unwrap()).unwrap();
    assert_eq!(parsed["a"], "[credential: GitHub token, masked]");
    assert_eq!(parsed["b"], "ordinary");
    let parts = raw(r#"["-----BEGIN PRIVATE KEY-----","public next field"]"#);
    let page = envelope(&parts, until(), false).unwrap();
    assert!(page["json"].as_str().unwrap().contains("public next field"));
}
#[test]
fn witnesses_and_structural_keys_refuse_instead_of_changing() {
    let token = format!("ghp_{}", "A".repeat(36));
    let witness = RetainedJson::from(token.clone());
    assert!(envelope(&witness, until(), true).is_err());
    let key = RetainedJson::parse(
        serde_json::to_vec(&json!({token:"value"}))
            .unwrap()
            .as_slice(),
    )
    .unwrap();
    assert!(envelope(&key, until(), false).is_err());
    let witness = raw(r#"{"field":"blocks[0].input","origin":{"path":["\ud800"],"n":1e+02}}"#);
    assert_eq!(
        raw(envelope(&witness, until(), true).unwrap()["json"]
            .as_str()
            .unwrap()),
        witness
    );
}
#[test]
fn surrogate_assessment_never_commits_repaired_or_uncertain_masked_values() {
    let token = format!("ghp_{}", "A".repeat(36));
    let source = format!(r#""{token}\ud800""#);
    assert!(envelope(&raw(&source), until(), false).is_err());
    // A logical surrogate is an actual delimiter, never the source spelling's
    // six ordinary characters. PEM state still spans the assessment boundary.
    let source = raw(r#""-----BEGIN PRIVATE KEY-----\ud800secret\n-----END PRIVATE KEY-----""#);
    assert!(envelope(&source, until(), false).is_err());
    let clean = raw(r#""ghp_\ud800AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA""#);
    let page = envelope(&clean, until(), false).unwrap();
    assert_eq!(page["masked"], false);
    assert_eq!(raw(page["json"].as_str().unwrap()), clean);
}
#[test]
fn retained_encoder_streams_original_value_and_envelope_cost_is_final() {
    let value = raw(r#"{"n":1e+02,"s":"\ud800"}"#);
    let mut encoder = crate::output_json::Encoder::new(Vec::new(), until());
    encoder.retained(&value, false).unwrap();
    assert_eq!(encoder.finish().unwrap(), value.canonical());
    let mut encoder = crate::output_json::Encoder::new(Vec::new(), until());
    encoder.opaque(&value).unwrap();
    let bytes = encoder.finish().unwrap();
    let page = envelope(&value, until(), false).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), page);
    assert!(bytes.len() > value.canonical().len());
}
