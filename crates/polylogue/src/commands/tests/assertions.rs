use super::*;

fn statement() -> Value {
    json!({"kind":"note","key":null,"value":"Synthetic decision","valid_until":null,
        "links":[{"role":"about","reference":"repo:example"}]})
}

/// Whether the operation's typed arguments accept `args`.
fn decodes(name: &str, args: &Value) -> bool {
    let row = command(name).unwrap();
    row.registration()
        .unwrap()
        .admit(
            args,
            row,
            &Caller {
                surface: Surface::Mcp,
                cwd: None,
            },
        )
        .is_ok()
}

fn schemas(name: &str) -> (Value, Value) {
    let description = declarations()
        .descriptions()
        .iter()
        .find(|description| description.spec.name == name)
        .unwrap();
    (
        description.input_schema.clone(),
        description.output_schema.clone(),
    )
}

#[test]
fn assertion_operations_take_typed_arguments_and_keep_identity_beside_them() {
    for (name, args) in [
        ("assert", json!({"assertions":[statement()]})),
        (
            "assert-query",
            json!({"query":"sessions","assertion":{
                "kind":"note","value":"Synthetic finding","key":null,"valid_until":null,"links":[]
            }}),
        ),
        (
            "revise",
            json!({"bases":["assertion:root@revision"],"patch":{"value":{"change":"set","value":"Updated"}}}),
        ),
        (
            "withdraw",
            json!({"targets":["assertion:root@revision"],"reason":"Superseded"}),
        ),
    ] {
        let command = crate::commands::command(name).unwrap();
        assert!(
            command.tool == Some(crate::commands::Tool::Note) && command.http && command.writes
        );
        assert!(decodes(name, &args), "{name}");
        for field in [
            "author",
            "via",
            "machine",
            "resolved",
            "seen_hash",
            "request_id",
        ] {
            let mut invalid = args.clone();
            invalid[field] = json!("caller-controlled");
            assert!(!decodes(name, &invalid), "{name}: {field}");
        }
    }
}

#[test]
fn structured_cli_and_http_values_retain_assertion_data_without_string_wrappers() {
    let args = json!({"assertions":[statement()]});
    let encoded = args["assertions"].to_string();
    let (_, _, cli) = crate::cli::request_arguments(
        &["polylogue", "assert", &encoded],
        std::path::Path::new("/"),
    )
    .unwrap();
    let http = declarations()
        .parse_http_query("assert", &[("assertions".into(), encoded)], "same-id")
        .unwrap();
    assert_eq!(Value::Object(cli), args);
    assert_eq!(http.arguments, args);
    assert!(crate::cli::parse(&["polylogue", "assert", "broken JSON"]).is_err());
}

#[test]
fn assertion_arguments_refuse_caller_fields_and_nonexact_revision_bases() {
    for (name, args) in [
        (
            "assert",
            json!({"assertions":[{"kind":"note","value":"text","links":[],"author":"operator"}]}),
        ),
        (
            "assert-query",
            json!({"query":"sessions","assertion":statement(),"resolved":["session:claude-code:s1"]}),
        ),
        ("revise", json!({"bases":["assertion:root"],"patch":{}})),
        (
            "revise",
            json!({"bases":["variant:root@revision"],"patch":{}}),
        ),
        (
            "revise",
            json!({"bases":["assertion:root@revision"],"patch":{"seen_hash":"invented"}}),
        ),
        ("withdraw", json!({"targets":["bare-predecessor-id"]})),
    ] {
        assert!(!decodes(name, &args), "{name}: {args}");
    }
}

#[test]
fn assertion_operations_refuse_empty_work() {
    let daemon = crate::test_daemon::start("empty assertion work");
    for (name, args) in [
        ("assert", json!({"assertions":[]})),
        (
            "assert-query",
            json!({"query":"  ","assertion":statement()}),
        ),
        ("revise", json!({"bases":[],"patch":{}})),
        ("withdraw", json!({"targets":[]})),
    ] {
        let answer = crate::test_daemon::call(&daemon.daemon, name, args);
        assert_eq!(
            answer.failure().and_then(|failure| failure.code.as_deref()),
            Some("bad_request"),
            "{name}: {answer:?}"
        );
    }
}

#[test]
fn assertion_operations_are_registered_on_every_surface() {
    for name in ["assert", "assert-query", "revise", "withdraw"] {
        let command = crate::commands::command(name).unwrap();
        assert!(command.registration().is_some());
        assert_eq!(command.http_path(), format!("/{name}"));
        assert_eq!(command.tool, Some(crate::commands::Tool::Note));
    }
}

#[test]
fn assertion_operation_schemas_expose_closed_domain_arguments_and_writer_receipts() {
    for name in ["assert", "assert-query", "revise", "withdraw"] {
        let (arguments, answer) = schemas(name);
        assert_eq!(arguments["additionalProperties"], false, "{name}");
        assert!(arguments["properties"].get("request_id").is_none());
        assert!(arguments["properties"].get("author").is_none());
        let wire = answer.to_string();
        let receipt = if name == "assert-query" {
            "AssertQueryOutcome"
        } else {
            "UserOutcome"
        };
        assert!(
            wire.contains(receipt) && wire.contains("repeat") && wire.contains("pending"),
            "{name}: {answer}"
        );
    }
    let arguments = schemas("assert").0;
    assert_eq!(arguments["properties"]["assertions"]["type"], "array");
    assert_eq!(arguments["properties"]["assertions"]["minItems"], 1);
    assert!(arguments.to_string().contains("NewAssertion"));
}

#[test]
fn the_bulk_assertion_answer_declares_its_exact_population_count() {
    type BulkAnswer = polylogue_model::user::AssertQueryOutcome;
    let (_, schema) = schemas("assert-query");
    assert!(
        schema["$defs"]["AssertQueryOutcome"]["properties"]
            .get("targets")
            .is_some(),
        "{schema}"
    );
    let value = json!({"id":"selected","repeat":false,"pending":false,"targets":3});
    let answer: BulkAnswer = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(serde_json::to_value(answer).unwrap(), value);
    let mut unknown = value;
    unknown["caller_count"] = json!(3);
    assert!(serde_json::from_value::<BulkAnswer>(unknown).is_err());
}
