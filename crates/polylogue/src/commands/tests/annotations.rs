use super::*;
#[test]
fn authored_steps_are_admitted_and_transport_identity_fields_are_refused() {
    for step in [
        json!({"verb":"note","text":"Synthetic finding","as":"decision"}),
        json!({"verb":"tag","name":"reviewed"}),
        json!({"verb":"edit","text":"Updated finding"}),
        json!({"verb":"withdraw","reason":"Superseded"}),
        json!({"verb":"judge","verdict":"accept","reason":"Matches the test"}),
        json!({"verb":"save","name":"reviewed"}),
    ] {
        let arguments = json!({"selection":"session:claude-code:s1","steps":[step]});
        let policy = crate::daemon::handler::policy(
            command("rows").unwrap(),
            &arguments,
            polylogue_model::authorship::Channel::Mcp,
        )
        .unwrap();
        assert!(policy.changes_archive, "{arguments}");
        let declarations = declarations();
        let steps = declarations.steps("rows", "steps").unwrap();
        steps.validate_step(&arguments["steps"][0]).unwrap();
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
            let refused = crate::daemon::handler::policy(
                command("rows").unwrap(),
                &invalid,
                polylogue_model::authorship::Channel::Mcp,
            )
            .unwrap_err();
            assert_eq!(refused.code(), "bad_request", "{invalid}");
            let failure = steps.validate_step(&invalid["steps"][0]).unwrap_err();
            assert_eq!(failure.kind, command_surface::FailureKind::BadRequest);
            assert_eq!(
                failure.message,
                format!(
                    "{}: unknown argument {name}",
                    arguments["steps"][0]["verb"].as_str().unwrap()
                )
            );
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
        &crate::cli::line::SavedNames::default(),
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
    steps
        .validate_step(&json!({"verb":"edit","text":"hello"}))
        .unwrap();
    let refusal = steps
        .validate_step(&json!({"verb":"edit","text":"hello","bases":["annotation:a@b"]}))
        .unwrap_err();
    assert_eq!(refusal.kind, command_surface::FailureKind::BadRequest);
    assert_eq!(refusal.message, "edit: unknown argument bases");
    let request = crate::client::request(
        "rows",
        json!({"selection":"session:claude-code:s1","steps":[{"verb":"note","text":"hello"}]}),
        "authored".into(),
    );
    let call = crate::mcp::tests::tool_call("rows", request.arguments.clone(), &request.request_id);
    assert_eq!(call["name"], "polylogue_write");
    // The shared schema is declared at the row's home; the product's access
    // owner selects the tool that executes these particular arguments.
    let parsed = declarations()
        .parse_mcp_call(row.tool.unwrap().name(), call["arguments"].clone())
        .unwrap();
    assert_eq!(parsed.arguments, request.arguments);
    assert_eq!(parsed.request_id, "authored");
}
