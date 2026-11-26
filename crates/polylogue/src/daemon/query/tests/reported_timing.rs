//! Tool timing the source reports reaches the public registry through the real loop.
use super::super::*;
use super::{fixture_context, run};
use polylogue_testkit::Home;

#[test]
fn reported_duration_and_result_time_keep_meaning() {
    let home = Home::new();
    home.append("timing.jsonl", concat!(
        "{\"type\":\"assistant\",\"uuid\":\"call-message\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:00Z\",\"message\":{\"id\":\"api-call\",\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"call-a\",\"name\":\"Bash\",\"input\":{\"command\":\"printf synthetic\"}}]}}\n",
        "{\"type\":\"user\",\"uuid\":\"result-message\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:02Z\",\"toolUseResult\":{\"durationMs\":1500,\"exitCode\":0},\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"call-a\",\"content\":\"synthetic\"}]}}\n",
        "{\"type\":\"assistant\",\"uuid\":\"call-message-b\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:03Z\",\"message\":{\"id\":\"api-call-b\",\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"call-b\",\"name\":\"Bash\",\"input\":{\"command\":\"printf synthetic\"}}]}}\n",
        "{\"type\":\"user\",\"uuid\":\"result-message-b\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:04Z\",\"toolUseResult\":{\"durationMs\":999,\"exitCode\":0},\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"call-b\",\"content\":\"synthetic\"}]}}\n",
        "{\"type\":\"assistant\",\"uuid\":\"call-message-c\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:05Z\",\"message\":{\"id\":\"api-call-c\",\"role\":\"assistant\",\"content\":[{\"type\":\"tool_use\",\"id\":\"call-c\",\"name\":\"Bash\",\"input\":{\"command\":\"printf synthetic\"}}]}}\n",
        "{\"type\":\"user\",\"uuid\":\"result-message-c\",\"sessionId\":\"timing\",\"timestamp\":\"2026-10-07T10:00:06Z\",\"toolUseResult\":{\"durationMs\":1000,\"exitCode\":0},\"message\":{\"role\":\"user\",\"content\":[{\"type\":\"tool_result\",\"tool_use_id\":\"call-c\",\"content\":\"synthetic\"}]}}\n"
    ));
    let (mut engine, _lock) = home.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let pool = home.reads();
    let fields = vec!["reported_duration".to_owned(), "result_at".to_owned()];
    let answer = run(
        &pool.reader().unwrap(),
        Ask {
            text: "tool_calls where reported_duration >= 1s",
            cursor: None,
            page_size: Some(10),
            why: false,
            session: None,
            context: fixture_context(0, "UTC", 240),
            embeddings: false,
            ranking_identity: None,
            near: None,
            fields: &fields,
        },
    )
    .unwrap();
    // The 999 ms call is below the threshold; 1000 ms is at it.
    assert_eq!(answer["count"], 2);
    let mut durations: Vec<i64> = answer["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["reported_duration"].as_i64().unwrap())
        .collect();
    durations.sort_unstable();
    assert_eq!(durations, [1000, 1500]);
    let slow = answer["rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["reported_duration"] == 1500)
        .unwrap();
    assert_eq!(slow["result_at"], "2026-10-07T10:00:02Z");
    let catalog = polylogue_query::catalog::catalog(&[]);
    assert_eq!(
        polylogue_query::parse::with_catalog("tool_calls where duration >= 1s", &catalog)
            .unwrap_err()
            .kind(),
        polylogue_model::Code::UnknownField
    );
}
