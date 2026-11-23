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

/// An option of the provider import is refused as not built, naming it.
#[test]
fn a_provider_import_option_is_refused_as_not_built() {
    let (kind, message) = refusal(json!({"cancel": true}));
    assert_eq!(kind, "not_yet_available");
    assert!(message.contains("--cancel"), "{message}");
}
