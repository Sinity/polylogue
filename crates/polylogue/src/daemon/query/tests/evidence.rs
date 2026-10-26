//! `first_seen` and `evidence_until`: what Polylogue had, and when.

use super::*;
use polylogue_testkit::{ClaudeHome, claude_user};

fn query(home: &ClaudeHome, text: &str) -> Result<Value, Failure> {
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let mut request = ask(text);
    request.page_size = Some(100);
    run(&reader, request)
}

#[test]
fn evidence_until_reads_the_rows_polylogue_had_then() {
    let home = ClaudeHome::new();
    home.append("a.jsonl", &claude_user("a1", "a", "first prompt"));
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let first = query(&home, "messages | sort by first_seen").unwrap();
    let then = first["rows"][0]["first_seen"].as_str().unwrap().to_owned();
    // Observations are timed in milliseconds: the next one is later.
    std::thread::sleep(std::time::Duration::from_millis(5));
    home.append("a.jsonl", &claude_user("a2", "a", "second prompt"));
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let all = query(&home, "messages | sort by first_seen").unwrap();
    assert_eq!(all["count"], 2);
    assert_eq!(all["rows"][0]["first_seen"], then.as_str(), "{all}");
    assert_ne!(all["rows"][1]["first_seen"], then.as_str(), "{all}");
    let cut = query(&home, &format!("messages where evidence_until:{then}")).unwrap();
    assert_eq!(cut["count"], 1, "{cut}");
    assert_eq!(cut["rows"][0]["id"], first["rows"][0]["id"]);
    let session = query(&home, "sessions | sort by first_seen").unwrap();
    assert_eq!(session["rows"][0]["first_seen"], then.as_str());
    let refused = query(&home, &format!("sessions where evidence_until:{then}")).unwrap_err();
    assert_eq!(refused.kind, "invalid_query");
}
