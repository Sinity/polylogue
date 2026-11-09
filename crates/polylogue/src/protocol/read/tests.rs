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
        crate::daemon::service::read::ReadRequest::from_arguments(arguments, Some("sealed".into()))
            .unwrap();
    assert_eq!(request.options.view, polylogue_model::read::View::Dialogue);
    assert_eq!(request.cursor.as_deref(), Some("sealed"));
    let round = Arguments::from_request(&request).unwrap();
    assert!(serde_json::to_value(round).unwrap().get("cursor").is_none());
    // The command line sends the cursor beside the arguments too.
    let (_, values, arguments) = crate::cli::request_arguments(
        &[
            "polylogue",
            "read",
            "session:test:one",
            "--view",
            "dialogue",
            "--cursor",
            "sealed",
        ],
        std::path::Path::new("/"),
    )
    .unwrap();
    assert!(arguments.get("cursor").is_none());
    assert_eq!(values.text("cursor").as_deref(), Some("sealed"));
}
