use super::*;
use crate::daemon::{Request, service::handle};

fn data(response: Response) -> Value {
    match response {
        Response::Ok { data } => data,
        other => panic!("{other:?}"),
    }
}

#[test]
fn catalogs_advance_and_preserve_every_declaration() {
    for subject in &SUBJECTS[1..] {
        let expected = catalog(subject, None);
        let mut seen = Vec::new();
        let mut continuation = None;
        loop {
            let page = data(answer(
                subject,
                None,
                None,
                Some(1),
                continuation.as_deref(),
            ));
            assert_eq!(page["subject"], *subject);
            assert_eq!(page["total"], expected.len());
            seen.extend(page["entries"].as_array().unwrap().iter().cloned());
            let next = page["next"].as_str().map(str::to_owned);
            assert!(next.is_none() || next != continuation);
            continuation = next;
            if continuation.is_none() {
                break;
            }
            assert!(seen.len() <= expected.len());
        }
        assert_eq!(seen, expected);
    }
    let fields = data(answer(
        "fields",
        None,
        Some("messages"),
        Some(u64::MAX),
        None,
    ));
    assert_eq!(
        fields["entries"].as_array().unwrap().len(),
        fields::names(Unit::Messages).count()
    );
    assert!(matches!(
        answer("fields", None, None, Some(0), None),
        Response::Error { .. }
    ));
    assert!(matches!(
        answer("fields", None, None, None, Some("units//1")),
        Response::Error { .. }
    ));
    assert!(matches!(
        answer("fields", None, Some("bananas"), None, None),
        Response::Error { .. }
    ));
    let capabilities = catalog("capabilities", None);
    let views: Vec<_> = capabilities
        .iter()
        .filter(|v| v["kind"] == "read_view")
        .map(|v| v["name"].as_str().unwrap())
        .collect();
    assert_eq!(views, polylogue_render::View::NAMES);
}

#[test]
fn query_explanation_validates_without_acquiring_a_reader() {
    let w = crate::test_daemon::start("synthetic query explanation");
    let position = w.daemon.reads.reader().unwrap().change_position().unwrap();
    for expression in [
        "tool_calls where outcome:error | group by session | count",
        "sessions where exists tool_call(outcome:error)",
        "messages where written_by:human chrono",
        "sessions where cost:unknown",
        "tool_calls | union (outcome:error)",
        "sessions | tool_calls | union (outcome:error)",
    ] {
        let result = data(handle(
            Request::Explain {
                subject: "query".into(),
                expression: Some(expression.into()),
                unit: None,
                limit: None,
                continuation: None,
            },
            &w.daemon,
        ));
        assert_eq!(result["valid"], true);
        assert!(!result["explanation"]["sql"].as_str().unwrap().is_empty());
        assert!(result["explanation"]["parsed"].is_object());
        if expression.starts_with("tool_calls") {
            assert_eq!(
                result["explanation"]["selected_units"],
                json!(["tool_calls"])
            );
        }
        if expression.contains("exists") || expression.starts_with("sessions |") {
            assert_eq!(
                result["explanation"]["selected_units"],
                json!(["sessions", "tool_calls"])
            );
        }
    }
    assert_eq!(
        w.daemon.reads.reader().unwrap().change_position().unwrap(),
        position
    );
    for expression in [
        None,
        Some(""),
        Some("messages where nonsense:x"),
        Some("bananas where role:user"),
    ] {
        assert!(
            matches!(
                answer("query", expression, None, None, None),
                Response::Error { .. }
            ),
            "{expression:?}"
        );
    }
    w.daemon.reads.close();
    assert!(matches!(
        handle(
            Request::Explain {
                subject: "query".into(),
                expression: Some("sessions".into()),
                unit: None,
                limit: None,
                continuation: None
            },
            &w.daemon
        ),
        Response::Ok { .. }
    ));
}
