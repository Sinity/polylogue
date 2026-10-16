use super::*;

fn reason(value: Value) -> String {
    match Request::decode(&value) {
        Ok(_) => panic!("expected refused request"),
        Err(error) => error["error"]["reason"].as_str().unwrap().to_owned(),
    }
}

#[test]
fn invalid_wire_shapes_keep_specific_refusals_and_order() {
    for (value, expected) in [
        (json!([]), "unsupported_option"),
        (json!({"extra":true}), "unsupported_option"),
        (
            json!({"command":"other","presentation":false}),
            "invalid_command",
        ),
        (json!({"presentation":null}), "unsupported_presentation"),
        (json!({"view":"future"}), "unsupported_view"),
        (
            json!({"view":"dialogue","thinking":"full"}),
            "unsupported_selection_option",
        ),
        (
            json!({"view":"transcript","tools":"full"}),
            "transcript_requires_full_tools_thinking_harness",
        ),
        (json!({"branches":false}), "unsupported_branches_selection"),
        (json!({}), "invalid_reference"),
        (json!({"reference":""}), "invalid_reference"),
        (json!({"reference":"a".repeat(16385)}), "invalid_reference"),
        (
            json!({"reference":"claude-code:opaque","continuation":false}),
            "invalid_continuation",
        ),
        (
            json!({"reference":"claude-code:opaque","continuation":"bad","budget_tokens":0}),
            "invalid_continuation",
        ),
        (
            json!({"reference":"claude-code:opaque","budget_tokens":0}),
            "invalid_budget",
        ),
        (
            json!({"reference":"claude-code:opaque","budget_tokens":-1}),
            "invalid_budget",
        ),
        (
            json!({"reference":"claude-code:opaque","budget_tokens":1.5}),
            "invalid_budget",
        ),
        (
            json!({"reference":"claude-code:opaque","budget_tokens":MAX_BUDGET+1}),
            "invalid_budget",
        ),
    ] {
        assert_eq!(reason(value), expected);
    }
}

#[test]
fn default_and_authenticated_inherited_budgets_remain_distinct() {
    let value = json!({"reference":"claude-code:opaque"});
    let first = Request::decode(&value).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(first.budget, 4000);
    let token = seal(
        &[7; 32],
        first.reference,
        &Position {
            block: 3,
            scalar: 9,
            budget: 6000,
        },
        first.key,
    );
    let inherited = json!({"reference":first.reference,"continuation":token});
    let next = Request::decode(&inherited).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(next.budget, 6000);
    assert_eq!(next.continuation.as_ref().unwrap().2.scalar, 9);
    let explicit = json!({"reference":first.reference,"continuation":token,"budget_tokens":128});
    assert_eq!(
        Request::decode(&explicit)
            .unwrap_or_else(|e| panic!("{e}"))
            .budget,
        128
    );
    assert_eq!(
        reason(json!({"reference":"claude-code:other","continuation":token})),
        "cross_reference_continuation"
    );
}

#[test]
fn follow_up_preserves_selection_and_wire_envelope() {
    for mcp in [false, true] {
        for view in ["native", "dialogue", "transcript"] {
            for branches in [None, Some(false), Some(true)] {
                if view != "transcript" && branches.is_some() {
                    continue;
                }
                let mut value = json!({"reference":"claude-code:opaque","budget_tokens":6000});
                if mcp {
                    value["presentation"] = json!("mcp");
                }
                if view != "native" {
                    value["view"] = json!(view);
                }
                if view == "transcript" {
                    for option in ["tools", "thinking", "harness"] {
                        value[option] = json!("full");
                    }
                    if let Some(branches) = branches {
                        value["branches"] = json!(branches);
                    }
                }
                let decoded = Request::decode(&value).unwrap_or_else(|e| panic!("{e}"));
                let next = decoded.follow_up(
                    &[7; 32],
                    &Position {
                        block: 2,
                        scalar: 13,
                        budget: 6000,
                    },
                );
                let args = if mcp {
                    assert_eq!(next["tool"], "read");
                    &next["arguments"]
                } else {
                    &next
                };
                assert_eq!(args["ref"], value["reference"]);
                assert_eq!(args.get("view"), value.get("view"));
                assert_eq!(args.get("branches"), value.get("branches"));
                for option in ["tools", "thinking", "harness"] {
                    assert_eq!(args.get(option), value.get(option));
                }
                assert_eq!(args.get("budget_tokens").is_some(), mcp);
                let (_, _, position) =
                    open(args["continuation"].as_str().unwrap(), decoded.key).unwrap();
                assert_eq!(
                    (position.block, position.scalar, position.budget),
                    (2, 13, 6000)
                );
                let page = json!({"shown":[{"text":"東京🙂"}],"next":next});
                let expected = if mcp {
                    super::super::cost(
                        &json!({"structuredContent":page,"content":[{"type":"text","text":MCP_COMPANION}]}),
                    )
                } else {
                    super::super::cost(&page)
                };
                assert_eq!(decoded.cost(&page), expected);
            }
        }
    }
}
