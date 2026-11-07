use super::*;
#[test]
fn request_uses_the_supplied_clock_and_machine() {
    let metadata = RequestMetadata::new(
        "synthetic-machine".into(),
        crate::WallClock::fixed("2026-03-08T12:34:56.123456789Z".parse().unwrap()),
    );
    let request = metadata.request(
        "synthetic-request".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
        Authorship {
            author: crate::Author::Operator,
            via: crate::authorship::Channel::Cli,
        },
    );
    assert_eq!(request.machine, "synthetic-machine");
    assert_eq!(
        jiff::Timestamp::from(request.at),
        "2026-03-08T12:34:56.123456789Z"
            .parse::<jiff::Timestamp>()
            .unwrap()
    );
}

#[test]
fn write_intent_refuses_caller_supplied_identity() {
    for write in [
        serde_json::json!({"write":"withdraw", "note":"root", "author":"operator"}),
        serde_json::json!({"write":"set_setting", "key":"browser.actions", "value":"on", "machine":null, "author":"operator"}),
        serde_json::json!({"write":"add_note", "note":{"kind":"note", "value":"text", "targets":["repo:synthetic"], "author":"operator"}}),
    ] {
        assert!(serde_json::from_value::<UserWrite>(write).is_err());
    }
}

#[test]
fn revision_bases_use_only_exact_natural_assertion_references() {
    let write = UserWrite::ReviseAssertions {
        bases: vec![crate::RevisionReference::exact(
            "family".into(),
            "revision".into(),
        )],
        patch: Default::default(),
    };
    let encoded = serde_json::to_value(&write).unwrap();
    assert_eq!(
        encoded["bases"],
        serde_json::json!(["assertion:family@revision"])
    );
    assert_eq!(
        serde_json::from_value::<UserWrite>(encoded.clone()).unwrap(),
        write
    );
    for invalid in [
        serde_json::json!(["assertion:family"]),
        serde_json::json!(["message:codex:message"]),
        serde_json::json!([{"root":"family","revision":"revision"}]),
    ] {
        let mut invalid_write = encoded.clone();
        invalid_write["bases"] = invalid;
        assert!(serde_json::from_value::<UserWrite>(invalid_write).is_err());
    }
}
