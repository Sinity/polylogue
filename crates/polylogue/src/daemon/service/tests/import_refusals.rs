use crate::test_daemon::{self, Call};
use serde_json::json;

fn refusal(arguments: serde_json::Value) -> (String, String) {
    let d = test_daemon::start("import refusals");
    let answer = Call::new("import", arguments).answer(&d.daemon);
    let value = serde_json::to_value(&answer).unwrap();
    let error = &value["error"];
    (
        error["kind"]
            .as_str()
            .unwrap_or_else(|| panic!("{value}"))
            .to_owned(),
        error["message"].as_str().unwrap_or_default().to_owned(),
    )
}

/// A provider import's options are refused on files, naming the option.
#[test]
fn a_provider_import_option_on_files_is_refused() {
    let (kind, message) = refusal(json!({"paths": ["/home/op/export.json"], "cancel": true}));
    assert_eq!(kind, "bad_request");
    assert!(
        message.starts_with("--cancel is for a provider import"),
        "{message}"
    );
}
