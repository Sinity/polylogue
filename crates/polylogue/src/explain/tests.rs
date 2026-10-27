use super::*;
use crate::daemon::{Request, Response, service::handle};

fn data(response: Response) -> Value {
    match response {
        Response::Ok { data } => data,
        other => panic!("{other:?}"),
    }
}

fn explain(expression: Option<&str>) -> Request {
    Request::Explain {
        subject: "query".into(),
        expression: expression.map(str::to_owned),
        unit: None,
        page_size: None,
        cursor: None,
        session: Some("claude-code:s1".into()),
    }
}

#[test]
fn declaration_pages_preserve_every_entry() {
    for subject in &SUBJECTS[1..] {
        let expected = entries(subject, None).unwrap();
        let mut seen = Vec::new();
        let mut cursor = None;
        loop {
            let page = answer(subject, None, None, Some(1), cursor.as_deref()).unwrap();
            assert_eq!(page["subject"], *subject);
            assert_eq!(page["total"], expected.len());
            assert!(page.get("syntax").is_none());
            seen.extend(page["entries"].as_array().unwrap().iter().cloned());
            let next = page["next"].as_str().map(str::to_owned);
            assert!(next.is_none() || next != cursor);
            cursor = next;
            if cursor.is_none() {
                break;
            }
            assert!(seen.len() <= expected.len());
        }
        assert_eq!(seen, expected);
    }
    let page = answer("fields", None, Some("messages"), Some(u64::MAX), None).unwrap();
    assert_eq!(
        page["entries"].as_array().unwrap().len(),
        fields::names(Unit::Messages).count()
    );
    assert!(answer("fields", None, None, Some(0), None).is_err());
    assert!(answer("fields", None, None, None, Some("units//1")).is_err());
    assert!(answer("fields", None, Some("bananas"), None, None).is_err());
    let capabilities = entries("capabilities", None).unwrap();
    let views: Vec<_> = capabilities
        .iter()
        .filter(|v| v["kind"] == "read_view")
        .map(|v| v["name"].as_str().unwrap())
        .collect();
    assert_eq!(views, polylogue_render::View::NAMES);
}

#[test]
fn every_grammar_keyword_is_discoverable() {
    let page = answer("syntax", None, None, None, None).unwrap();
    let keywords = page["entries"][0]["keywords"].as_array().unwrap();
    for line in include_str!("../../../query/src/query.pest")
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
    {
        for (i, text) in line.split('"').enumerate() {
            if i % 2 == 1
                && text.chars().any(|c| c.is_ascii_alphabetic())
                && text
                    .chars()
                    .all(|c| c.is_ascii_alphabetic() || c == '_' || c == ':')
            {
                assert!(keywords.contains(&json!(text)), "missing {text}");
            }
        }
    }
    assert!(
        page["entries"][0]["stages"]
            .as_array()
            .unwrap()
            .contains(&json!("sort by"))
    );
    assert!(
        page["entries"][0]["aggregates"]
            .as_array()
            .unwrap()
            .contains(&json!("count_distinct"))
    );
}

#[test]
fn query_explanation_uses_archive_definitions_session_and_day_without_writes() {
    let w = crate::test_daemon::start("synthetic query explanation");
    w.set("day.zone", "Pacific/Kiritimati");
    w.set("day.logical_start", "07:30");
    data(handle(
        Request::Save {
            request_id: "save-explain".into(),
            name: "mine".into(),
            text: "self".into(),
            author: polylogue_model::Author::Operator,
        },
        &w.daemon,
    ));
    let kept = data(handle(
        Request::Keep {
            request_id: "keep-explain".into(),
            text: "sessions".into(),
            name: None,
        },
        &w.daemon,
    ));
    let position = w.daemon.reads.reader().unwrap().change_position().unwrap();
    for expression in [
        "tool_calls where outcome:error | group by session | count".to_owned(),
        "sessions where exists tool_call(outcome:error)".into(),
        "messages where written_by:human chrono".into(),
        "sessions where cost:unknown".into(),
        "tool_calls | union (outcome:error)".into(),
        "sessions | tool_calls | union (outcome:error)".into(),
        "query:mine".into(),
        format!("result:{}", kept["id"].as_str().unwrap()),
    ] {
        let result = data(handle(explain(Some(&expression)), &w.daemon));
        let explanation = &result["explanation"];
        let program = &explanation["program"];
        assert!(!program["sql"].as_str().unwrap().is_empty());
        assert!(explanation["parsed"].is_object());
        assert_eq!(program["zone"], "Pacific/Kiritimati");
        assert_eq!(program["day_start_minutes"], 450);
        assert!(program["incomplete"].is_array());
        assert!(program["params"].is_array());
        if expression == "query:mine" {
            assert_eq!(program["definitions"][0]["name"], "mine");
            assert!(program["params"].to_string().contains("claude-code:s1"));
        }
        if expression.starts_with("tool_calls") {
            assert_eq!(explanation["units"], json!(["tool_calls"]));
        }
        if expression.contains("exists") || expression.starts_with("sessions |") {
            assert_eq!(explanation["units"], json!(["sessions", "tool_calls"]));
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
                handle(explain(expression), &w.daemon),
                Response::Error { .. }
            ),
            "{expression:?}"
        );
    }
}
