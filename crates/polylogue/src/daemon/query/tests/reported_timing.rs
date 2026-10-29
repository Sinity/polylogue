//! Native tool timing facts reach the public registry through the real loop.
use super::super::*;
use polylogue_testkit::ClaudeHome;

#[test]
fn reported_duration_and_result_time_preserve_their_native_meaning() {
    let home = ClaudeHome::new();
    home.append("timing.jsonl", concat!(
        "{\"type\":\"assistant\",\"uuid\":\"call-message\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:00Z\",\"message\":{\"id\":\"api-call\",\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"call-a\",\"name\":\"Bash\",\"input\":{\"command\":\"printf synthetic\"}}]}}\n",
        "{\"type\":\"user\",\"uuid\":\"result-message\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:02Z\",\"toolUseResult\":{\"durationMs\":1500,\"exitCode\":0},\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"call-a\",\"content\":\"synthetic\"}]}}\n"
    ));
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let pool = home.reads();
    let fields = vec!["reported_duration".to_owned(), "result_at".to_owned()];
    let answer = run(&pool.reader().unwrap(), Ask { text: "tool_calls where reported_duration >= 1s", cursor: None, page_size: Some(1), why: false, session: None, now: 0, day: Some(LogicalDay::named("UTC", 240).unwrap()), embeddings: false, near: None, fields: &fields }).unwrap();
    assert_eq!(answer["count"], 1);
    assert_eq!(answer["rows"][0]["reported_duration"], 1500);
    assert_eq!(answer["rows"][0]["result_at"], "2026-10-07T10:00:02Z");
    let catalog = polylogue_query::catalog::catalog(&[], &[]);
    assert_eq!(polylogue_query::parse::with_catalog("tool_calls where duration >= 1s", &catalog).unwrap_err().kind(), "unknown_field");
}
