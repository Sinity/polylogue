use crate::test_daemon::{self, Call};
use serde_json::json;

#[test]
fn resume_in_hands_off() {
    let d = test_daemon::start("continue the clock work");
    let answer = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s1", "dry_run": true,
                "steps": [{"verb": "resume", "in": "codex"}],
            }),
        )
        .id("resume-dry")
        .answer(&d.daemon),
    );
    let outcome = &answer["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "would_do", "{answer}");
    assert_eq!(outcome["refusal"], serde_json::Value::Null, "{answer}");
    let would = outcome["would"].as_str().unwrap();
    let tokens = would
        .strip_prefix("would continue claude-code:s1 in codex with a brief of ")
        .unwrap()
        .strip_suffix(" tokens")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let invocation =
        d.daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "resume-test",
            ));
    let reader = d.daemon.reads.reader().unwrap();
    let context = d.daemon.resolve_operation_context(invocation).unwrap();
    let source = polylogue_model::SessionId::parse("claude-code:s1").unwrap();
    let prepared = crate::briefing::prepare(
        crate::briefing::BriefContext {
            reader: &reader,
            delivery_id: "resume-dry-1",
            context: &context,
            ranking_identity: None,
            near: None,
        },
        &d.daemon.scope,
        None,
        &[],
        &crate::briefing::Ask::Takeover {
            session: source,
            for_session: None,
        },
        command_surface::TokenBudget::Default,
    )
    .unwrap();
    assert_eq!(
        tokens,
        polylogue_model::read::estimate_tokens(&format!(
            "Hand-off from claude-code:s1:\n{}",
            prepared.text
        ))
    );
    drop(reader);
    assert_eq!(d.daemon.acts.hosted_sessions().len(), 0);
    let deliveries = crate::test_answers::empty(
        Call::new("rows", json!({"selection": "deliveries"})).answer(&d.daemon),
    );
    assert_eq!(deliveries["reply"]["count"], 0);
}

#[test]
fn resume_over_several_is_ambiguous_with_candidates() {
    let d = test_daemon::start_with_sources_and_workers(
        "first session",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/s2.jsonl"),
                polylogue_testkit::claude_user("u2", "s2", "second session"),
            )
            .unwrap();
        },
        1,
        2,
    );
    let answer = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "references": ["session:claude-code:s1", "session:claude-code:s2"],
                "steps": [{"verb": "resume"}],
            }),
        )
        .answer(&d.daemon),
    );
    let outcome = &answer["reply"]["steps"][0]["outcome"];
    assert_eq!(outcome["status"], "failed");
    assert_eq!(outcome["error"]["kind"], "ambiguous");
    assert_eq!(
        outcome["error"]["candidates"],
        json!(["claude-code:s1", "claude-code:s2"])
    );
}

#[test]
fn resume_dry_run_prints_harness_command() {
    let d = test_daemon::start("resume this work");
    let answer = crate::test_answers::ok(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s1", "dry_run": true,
                "steps": [{"verb": "resume"}],
            }),
        )
        .answer(&d.daemon),
    );
    assert_eq!(
        answer["reply"]["steps"][0]["outcome"]["would"],
        "would resume claude-code:s1 with claude --resume s1"
    );
}
