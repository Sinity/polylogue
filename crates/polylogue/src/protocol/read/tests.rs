use super::*;
use serde_json::json;

#[test]
fn page_options_are_not_operation_arguments() {
    for value in [
        json!({"target":"session:test:one","cursor":"sealed"}),
        json!({"target":"sessions","page_tokens":0}),
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
