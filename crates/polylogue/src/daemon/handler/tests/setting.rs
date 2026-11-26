use super::*;

#[test]
fn set_key_reads_declaration_choices_without_changing_write_or_scope_policy() {
    let daemon = crate::test_daemon::start("synthetic setting reader");
    let (_, _, arguments) =
        crate::cli::request_arguments(&["polylogue", "set", "browser.actions"]).unwrap();
    let answer = crate::client::call_ready(
        &daemon.daemon.archive,
        &request("set", Value::Object(arguments), "setting-details-cli"),
    )
    .unwrap();
    let answer = crate::test_answers::ok(answer);
    let details: polylogue_model::settings::Details = serde_json::from_value(answer).unwrap();
    assert_eq!(details.current.key, "browser.actions");
    assert_eq!(details.allowed_values, ["on", "off"]);
    assert_eq!(
        details.current.value,
        Some(polylogue_model::settings::Value::Boolean(false))
    );
    assert!(details.history.is_empty());
    let row = crate::commands::command("set").unwrap();
    for surface in [Surface::Cli, Surface::Mcp, Surface::Http] {
        let caller = Caller { surface, cwd: None };
        let policy = row
            .registration()
            .unwrap()
            .admit(&json!({"key":"browser.actions"}), row, &caller)
            .unwrap();
        assert!(policy.read_only);
        let write = row
            .registration()
            .unwrap()
            .admit(&json!({"key":"browser.actions","value":"on"}), row, &caller)
            .unwrap();
        assert!(!write.read_only);
    }
    for arguments in [
        json!({"key":"browser.actions","machine":"other-host"}),
        json!({"key":"not-a-declared-setting"}),
        json!({"key":"signal.interrupts_per_hour","machine":""}),
    ] {
        let answer = crate::test_daemon::Call::new("set", arguments).answer(&daemon.daemon);
        let failure = answer.failure().unwrap();
        assert_eq!(failure.code.as_deref(), Some("bad_value"));
    }
    let listed = crate::test_daemon::Call::new("set", json!({})).answer(&daemon.daemon);
    let catalog: crate::protocol::setting::Answer =
        serde_json::from_value(listed.value().unwrap().clone()).unwrap();
    let crate::protocol::setting::Answer::Catalog(catalog) = catalog else {
        panic!("set without a key lists its catalog");
    };
    let browser = catalog
        .iter()
        .find(|current| current.key == "browser.actions")
        .unwrap();
    assert!(browser.authored.is_none());
}
