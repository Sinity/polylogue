use super::*;

#[test]
fn response_boundary_updates_preserve_absent_and_supplied_states() {
    let mut production = Production {
        response_boundary_unknown: Some(true),
        ..Production::default()
    };
    production.merge(Production::default());
    assert_eq!(production.response_boundary_unknown, Some(true));
    production.merge(Production {
        response_boundary_unknown: Some(false),
        ..Production::default()
    });
    production.merge(Production::default());
    assert_eq!(production.response_boundary_unknown, Some(false));
    let encoded = serde_json::to_value(&production).unwrap();
    assert_eq!(encoded["response_boundary_unknown"], false);
    assert_eq!(
        serde_json::from_value::<Production>(encoded).unwrap(),
        production
    );
    assert!(
        serde_json::to_value(Production::default())
            .unwrap()
            .get("response_boundary_unknown")
            .is_none()
    );
}

#[test]
fn partial_production_readings_keep_other_reported_server_counts() {
    let mut first = Production {
        server_tool_use: Some(ServerToolUse {
            web_search_requests: Some(2),
            web_fetch_requests: None,
        }),
        phase: Some("commentary".into()),
        ..Production::default()
    };
    first.merge(Production {
        server_tool_use: Some(ServerToolUse {
            web_search_requests: None,
            web_fetch_requests: Some(0),
        }),
        phase: Some("final".into()),
        iterations: Some(3),
        ..Production::default()
    });
    assert_eq!(
        first.server_tool_use,
        Some(ServerToolUse {
            web_search_requests: Some(2),
            web_fetch_requests: Some(0)
        })
    );
    assert_eq!(first.phase.as_deref(), Some("final"));
    assert_eq!(first.iterations, Some(3));
}
