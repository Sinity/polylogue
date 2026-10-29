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
