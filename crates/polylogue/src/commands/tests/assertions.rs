use super::*;
fn accepts(arguments: &Value) -> bool {
    let row = command("rows").unwrap();
    row.registration()
        .unwrap()
        .admit(
            arguments,
            row,
            &Caller {
                surface: Surface::Mcp,
                cwd: None,
            },
        )
        .is_ok()
}
#[test]
fn authored_steps_take_their_declared_fields_and_transport_identity() {
    for step in [
        json!({"verb":"note","text":"Synthetic finding","as":"decision"}),
        json!({"verb":"tag","name":"reviewed"}),
        json!({"verb":"edit","text":"Updated finding"}),
        json!({"verb":"withdraw","reason":"Superseded"}),
        json!({"verb":"judge","verdict":"accept","reason":"Matches the test"}),
        json!({"verb":"save","name":"reviewed"}),
    ] {
        let arguments = json!({"selection":"session:claude-code:s1","steps":[step]});
        assert!(accepts(&arguments), "{arguments}");
        for name in [
            "author",
            "via",
            "machine",
            "resolved",
            "seen_hash",
            "request_id",
        ] {
            let mut invalid = arguments.clone();
            invalid["steps"][0][name] = json!("caller-controlled");
            assert!(!accepts(&invalid), "{invalid}");
        }
    }
}
#[test]
fn authored_line_steps_keep_their_values_on_the_cli() {
    let words = [
        "session:claude-code:s1",
        "note",
        "Synthetic decision",
        "--as",
        "decision",
        "tag",
        "reviewed",
        "--request-id",
        "same-id",
    ];
    let parsed = crate::cli::line::parse(
        &words.map(str::to_owned),
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let request = crate::cli::run_line::request(&parsed).unwrap();
    assert_eq!(request.request_id, "same-id");
    assert_eq!(request.arguments["selection"], "session:claude-code:s1");
    assert_eq!(
        request.arguments["steps"],
        json!([{"verb":"note","text":"Synthetic decision","as":"decision"},{"verb":"tag","name":"reviewed"}])
    );
}
#[test]
fn rows_declares_closed_authored_steps_and_uses_the_write_tool() {
    let row = command("rows").unwrap();
    assert!(row.http && row.writes);
    let steps = declarations().steps("rows", "steps").unwrap();
    assert!(
        steps
            .validate_step(&json!({"verb":"note","text":"hello","as":"decision"}))
            .is_ok()
    );
    assert!(
        steps
            .validate_step(&json!({"verb":"note","text":"hello","author":"operator"}))
            .is_err()
    );
    assert!(
        steps
            .validate_step(&json!({"verb":"edit","text":"hello","bases":["annotation:a@b"]}))
            .is_err()
    );
    let request = crate::client::request(
        "rows",
        json!({"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"hello"}]}),
        "authored".into(),
    );
    let parsed = declarations()
        .parse_mcp_call(
            "polylogue_write",
            json!({"arguments":request.arguments,"request_id":request.request_id}),
        )
        .unwrap();
    assert_eq!(parsed.arguments, request.arguments);
    assert_eq!(parsed.request_id, "authored");
}
