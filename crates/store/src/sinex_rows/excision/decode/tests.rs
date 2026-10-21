use super::*;
use crate::sinex_rows::{Replacement, Unit};

fn manifest() -> ExcisionPublication {
    let mut publication = ExcisionPublication {
        version: 1,
        replacements: vec![Replacement {
            unit: Unit {
                commit_seq: 1,
                kind: "note".into(),
                key: "a".into(),
            },
            batch: None,
            bytes: 0,
            rows: 0,
            blobs: vec![],
        }],
        retired_batches: vec![],
        digest: String::new(),
    };
    publication.digest = publication.computed_digest().unwrap();
    publication
}

#[test]
fn only_missing_fields_are_legacy_and_last_outer_value_still_wins() {
    for old in ["[]", "null", "false", "123", r#""legacy""#] {
        assert!(publication(old).unwrap().is_none());
    }
    assert!(
        publication(r#"{"ignored":{"nested":[1,2]}}"#)
            .unwrap()
            .is_none()
    );
    let expected = manifest();
    let json = serde_json::to_string(&expected).unwrap();
    for data in [
        format!(r#"{{"publication":{json}}}"#),
        format!(r#"{{"publication":null,"publication":{json}}}"#),
        format!(r#"{{"publ\u0069cation":{json}}}"#),
    ] {
        assert_eq!(publication(&data).unwrap(), Some(expected.clone()));
    }
    for data in [
        r#"{"publication":null}"#.into(),
        format!(r#"{{"publication":{json},"publication":null}}"#),
        format!(r#"{{"publication":{json}}} garbage"#),
        r#"{"publication":{},"ignored":[1,}"#.into(),
    ] {
        assert!(publication(&data).is_err(), "accepted {data}");
    }
}

#[test]
fn borrowed_manifest_still_checks_version_digest_and_identity() {
    for change in 0..3 {
        let mut value = manifest();
        match change {
            0 => value.version += 1,
            1 => value.digest = "wrong".into(),
            _ => {
                value.replacements[0].unit.key.clear();
                value.digest = value.computed_digest().unwrap();
            }
        }
        let data = serde_json::json!({"publication":value}).to_string();
        assert!(publication(&data).is_err());
    }
}
