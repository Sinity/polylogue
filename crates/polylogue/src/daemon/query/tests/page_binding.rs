use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

fn page(home: &ClaudeHome, cursor: Option<&str>) -> Result<Value, Failure> {
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let mut request = ask("sessions | sort by id");
    request.page_size = Some(2);
    request.cursor = cursor;
    run(&reader, request)
}

#[test]
fn an_insert_before_the_page_position_refuses_instead_of_repeating_a_row() {
    let home = ClaudeHome::new();
    for session in ["a", "b", "c"] {
        home.append(
            &format!("{session}.jsonl"),
            &claude_user(&format!("{session}1"), session, "synthetic prompt"),
        );
    }
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    let first = page(&home, None).unwrap();
    assert_eq!(first["count"], 3);
    assert_eq!(first["rows"][0]["id"], "claude-code:a");
    assert_eq!(first["rows"][1]["id"], "claude-code:b");
    let cursor = first["next"].as_str().unwrap();
    home.append("aa.jsonl", &claude_user("aa1", "aa", "new earlier prompt"));
    assert!(engine.converge().unwrap().errors.is_empty());
    let error = page(&home, Some(cursor)).unwrap_err();
    assert_eq!(error.kind, "stale_cursor");
}
