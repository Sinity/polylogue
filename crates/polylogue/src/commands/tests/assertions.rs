use super::*;

fn statement() -> Value {
    json!({"kind":"note","key":null,"value":"Synthetic decision","valid_until":null,
        "links":[{"role":"about","reference":"repo:example"}]})
}

#[test]
fn canonical_assertion_operations_share_typed_arguments_and_envelope_identity() {
    for (name, args, expected) in [
        ("assert", json!({"assertions":[statement()]}), "assert"),
        (
            "assert-query",
            json!({"query":"sessions","assertion":{
                "kind":"note","value":"Synthetic finding","key":null,"valid_until":null,"links":[]
            }}),
            "assert_query",
        ),
        (
            "revise",
            json!({"bases":["assertion:root@revision"],"patch":{"value":{"change":"set","value":"Updated"}}}),
            "revise",
        ),
        (
            "withdraw",
            json!({"targets":["assertion:root@revision"],"reason":"Superseded"}),
            "withdraw",
        ),
    ] {
        let command = crate::commands::command(name).unwrap();
        assert!(
            command.tool == Some(crate::commands::Tool::Note) && command.http && command.writes
        );
        let build = command.build_typed.unwrap();
        let cli = build(args.clone(), &cli_caller(), "same-id").unwrap();
        let mcp = build(args.clone(), &mcp_caller(), "same-id").unwrap();
        let http = build(
            args.clone(),
            &Caller {
                surface: Surface::Http,
                ..cli_caller()
            },
            "same-id",
        )
        .unwrap();
        let encoded = serde_json::to_value(cli).unwrap();
        assert_eq!(encoded, serde_json::to_value(mcp).unwrap());
        assert_eq!(encoded, serde_json::to_value(http).unwrap());
        assert_eq!(encoded["op"], expected);
        assert_eq!(encoded["request_id"], "same-id");
        assert!(encoded.get("author").is_none() && encoded.get("via").is_none());
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
            assert!(
                build(invalid, &mcp_caller(), "same-id").is_err(),
                "{name}: {field}"
            );
        }
    }
}

#[test]
fn structured_cli_and_http_values_retain_assertion_data_without_string_wrappers() {
    let args = json!({"assertions":[statement()]});
    let encoded = args["assertions"].to_string();
    let (command, cli) = crate::cli::parse(&["polylogue", "assert", &encoded]).unwrap();
    let http =
        Values::from_pairs(command, Surface::Http, vec![("assertions".into(), encoded)]).unwrap();
    let mcp = Values::from_json(command, Surface::Mcp, args.as_object().unwrap().clone()).unwrap();
    let expected = request_json(command, &mcp, &mcp_caller());
    assert_eq!(request_json(command, &cli, &cli_caller()), expected);
    assert_eq!(request_json(command, &http, &cli_caller()), expected);
    assert_eq!(expected["assertions"][0], statement());
    assert!(crate::cli::parse(&["polylogue", "assert", "broken JSON"]).is_err());
}

#[test]
fn canonical_assertion_arguments_refuse_empty_work_and_nonexact_revision_bases() {
    for (name, args) in [
        ("assert", json!({"assertions":[]})),
        (
            "assert",
            json!({"assertions":[{"kind":"note","value":"text","links":[],"author":"operator"}]}),
        ),
        (
            "assert-query",
            json!({"query":"  ","assertion":statement()}),
        ),
        (
            "assert-query",
            json!({"query":"sessions","assertion":statement(),"resolved":["session:claude-code:s1"]}),
        ),
        ("revise", json!({"bases":[],"patch":{}})),
        ("revise", json!({"bases":["assertion:root"],"patch":{}})),
        (
            "revise",
            json!({"bases":["variant:root@revision"],"patch":{}}),
        ),
        (
            "revise",
            json!({"bases":["assertion:root@revision"],"patch":{"seen_hash":"invented"}}),
        ),
        ("withdraw", json!({"targets":[]})),
        ("withdraw", json!({"targets":["bare-predecessor-id"]})),
    ] {
        assert!(
            command(name).unwrap().build_typed.unwrap()(args, &mcp_caller(), "invalid").is_err(),
            "{name}"
        );
    }
}

#[test]
fn assertion_operations_are_registered_on_every_surface() {
    for name in ["assert", "assert-query", "revise", "withdraw"] {
        let command = crate::commands::command(name).unwrap();
        assert!(command.register.is_some() && command.typed_schemas.is_some());
        assert_eq!(command.http_path(), format!("/{name}"));
        assert_eq!(command.tool, Some(crate::commands::Tool::Note));
    }
}

#[test]
fn assertion_operation_schemas_expose_closed_domain_arguments_and_writer_receipts() {
    for name in ["assert", "assert-query", "revise", "withdraw"] {
        let command = crate::commands::command(name).unwrap();
        let (arguments, answer) = command.typed_schemas.unwrap()();
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
    let arguments = command("assert").unwrap().typed_schemas.unwrap()().0;
    assert_eq!(arguments["properties"]["assertions"]["type"], "array");
    assert_eq!(arguments["properties"]["assertions"]["minItems"], 1);
    assert!(arguments.to_string().contains("NewAssertion"));
}

#[test]
fn the_bulk_assertion_answer_declares_its_exact_population_count() {
    type BulkAnswer = polylogue_model::user::AssertQueryOutcome;
    let (_, schema) = command("assert-query").unwrap().typed_schemas.unwrap()();
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
