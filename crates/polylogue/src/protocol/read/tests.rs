use super::*;
use serde_json::json;

#[test]
fn page_options_are_not_operation_arguments() {
    for value in [
        json!({"target":"session:test:one","cursor":"sealed"}),
        json!({"target":"sessions","root_run":"019a0000-0000-7000-8000-000000000001"}),
        json!({"cursor":"no target"}),
    ] {
        assert!(serde_json::from_value::<Arguments>(value).is_err());
    }
}

#[test]
fn a_read_keeps_the_request_cursor_beside_its_reading_options() {
    let arguments: Arguments =
        serde_json::from_value(json!({"target":"session:test:one","view":"dialogue"})).unwrap();
    let request =
        crate::daemon::service::read::request_of(arguments, Some("sealed".into())).unwrap();
    assert_eq!(request.options.view, polylogue_model::read::View::Dialogue);
    assert_eq!(request.cursor.as_deref(), Some("sealed"));
    let round = crate::daemon::service::read::arguments_of(&request).unwrap();
    assert!(serde_json::to_value(round).unwrap().get("cursor").is_none());
    // The line owns its read cursor in the typed read step.
    let words = [
        "session:test:one",
        "read",
        "--view",
        "dialogue",
        "--cursor",
        "sealed",
    ]
    .map(str::to_owned);
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let call = crate::cli::run_line::request(&parsed).unwrap();
    assert_eq!(call.operation, "rows");
    assert!(call.cursor.is_none());
    let rows: crate::protocol::line::Rows = serde_json::from_value(call.arguments).unwrap();
    let [crate::protocol::line::Step::Read(step)] = rows.steps.as_slice() else {
        panic!("one read")
    };
    assert_eq!(step.view, polylogue_model::read::View::Dialogue);
    assert_eq!(step.cursor.as_deref(), Some("sealed"));
}

#[test]
fn read_budget_keeps_default_zero_and_unlimited_through_the_existing_inverse() {
    use polylogue_model::read::ReadBudget;
    for (value, expected) in [
        (json!({"target":"session:test:one"}), ReadBudget::Default),
        (
            json!({"target":"session:test:one","page_tokens":null}),
            ReadBudget::Unlimited,
        ),
        (
            json!({"target":"session:test:one","page_tokens":0}),
            ReadBudget::Tokens(0),
        ),
        (
            json!({"target":"session:test:one","page_tokens":u64::MAX}),
            ReadBudget::Tokens(u64::MAX),
        ),
    ] {
        let args: Arguments = serde_json::from_value(value.clone()).unwrap();
        let request =
            crate::daemon::service::read::request_of(args, Some("cursor".into())).unwrap();
        assert_eq!(request.budget, expected);
        let args = crate::daemon::service::read::arguments_of(&request).unwrap();
        assert_eq!(args.page_tokens, expected);
        let step = crate::daemon::verbs::read::step_of(args, request.cursor).unwrap();
        assert_eq!(step.budget, expected);
        assert_eq!(step.cursor.as_deref(), Some("cursor"));
        let wire = serde_json::to_value(step).unwrap();
        match expected {
            ReadBudget::Default => assert!(wire.get("budget").is_none()),
            ReadBudget::Unlimited => assert_eq!(wire.get("budget"), Some(&serde_json::Value::Null)),
            ReadBudget::Tokens(n) => assert_eq!(wire["budget"], json!(n)),
        }
    }
}

#[test]
fn read_budget_none_parses_without_changing_its_physical_option() {
    use polylogue_model::read::ReadBudget;
    let words = [
        "session:test:one",
        "read",
        "--physical",
        "--budget",
        "none",
        "--show",
        "tools:full",
        "--show",
        "thinking:full",
    ]
    .map(str::to_owned);
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let request = crate::cli::run_line::request(&parsed).unwrap();
    let rows: crate::protocol::line::Rows = serde_json::from_value(request.arguments).unwrap();
    let [crate::protocol::line::Step::Read(step)] = rows.steps.as_slice() else {
        panic!("one read");
    };
    assert!(step.physical);
    assert_eq!(step.budget, ReadBudget::Unlimited);
    assert_eq!(step.show, ["tools:full", "thinking:full"]);
}
