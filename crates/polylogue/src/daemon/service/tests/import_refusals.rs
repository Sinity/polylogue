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

/// `--dry-run` belongs to `--from-archive` alone.
#[test]
fn dry_run_without_an_archive_is_a_bad_request() {
    let (kind, message) = refusal(json!({"dry_run": true}));
    assert_eq!(kind, "bad_request");
    assert!(message.contains("--dry-run"), "{message}");
}

/// An archive import names its own structure, so an origin is refused.
#[test]
fn an_archive_import_takes_no_origin() {
    let (kind, message) = refusal(json!({"from_archive": "/home/op/older", "origin": "codex"}));
    assert_eq!(kind, "bad_request");
    assert!(message.contains("--origin"), "{message}");
}
