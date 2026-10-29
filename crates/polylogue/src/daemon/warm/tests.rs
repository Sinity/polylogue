use super::*;
use serde_json::json;

#[test]
fn recent_keeps_the_most_recently_used() {
    let mut r = Recent::new(2);
    r.put("a".into(), 1);
    r.put("b".into(), 2);
    assert_eq!(r.get("a"), Some(1));
    r.put("c".into(), 3);
    assert_eq!(r.get("b"), None, "b was the least recently used");
    assert_eq!(r.get("a"), Some(1));
    assert_eq!(r.get("c"), Some(3));
}

#[test]
fn an_answer_is_reused_only_at_its_read_position() {
    let warm = Warm::default();
    warm.keep_answer("q".into(), "g:1:1".into(), false, &json!({"rows": [1]}));
    assert_eq!(warm.answer("q", "g:1:1"), Some(json!({"rows": [1]})));
    assert_eq!(
        warm.answer("q", "g:2:1"),
        None,
        "a write moved the position"
    );
    assert_eq!(warm.answer("other", "g:1:1"), None);
}

#[test]
fn a_first_page_is_reused_only_briefly() {
    let warm = Warm::default();
    warm.keep_answer("q".into(), "p".into(), true, &json!(1));
    assert!(warm.answer("q", "p").is_some());
    if let Ok(mut answers) = warm.answers.lock()
        && let Some(entry) = answers.entries.back_mut()
    {
        entry.1.made -= FIRST_PAGE_FOR;
    }
    assert!(warm.answer("q", "p").is_none(), "the clock moved on");
}

#[test]
fn count_failures_are_visible_and_a_success_clears_them() {
    let daemon = crate::test_daemon::start("count worker failure");
    let warm = Warm::default();
    warm.refresh_counts(&daemon.daemon.reads);
    assert!(warm.count_error().is_none());
    let before = warm.archive_counts.lock().unwrap().clone().unwrap();
    let closed = polylogue_store::ReadPool::open(&daemon.daemon.archive).unwrap();
    closed.close();
    warm.refresh_counts(&closed);
    assert!(warm.count_error().is_some());
    assert_eq!(
        warm.archive_counts.lock().unwrap().as_ref().unwrap().status,
        before.status
    );
    warm.archive_counts.lock().unwrap().as_mut().unwrap().took = Duration::ZERO;
    warm.recount_wanted.store(false, Ordering::Release);
    let reader = daemon.daemon.reads.reader().unwrap();
    let cached = warm.counts(&reader, "later position").unwrap();
    assert_eq!(cached.position, before.position);
    assert!(warm.count_error().is_some());
    assert!(warm.recount_wanted.load(Ordering::Acquire));
    drop(reader);
    warm.refresh_counts(&daemon.daemon.reads);
    assert!(warm.count_error().is_none());
}

#[test]
fn completion_reads_declared_values_and_refreshes_observed_models() {
    let d = crate::test_daemon::start("synthetic prompt");
    let values = |word: &str| match crate::daemon::service::handle(
        crate::daemon::Request::Complete {
            text: word.into(),
            position: word.len(),
        },
        &d.daemon,
    ) {
        crate::daemon::Response::Ok { data, .. } => data,
        other => panic!("{other:?}"),
    };
    assert!(
        values("live:")
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "waiting"
                && v["replace"]["start"] == 5
                && v["replace"]["end"] == 5)
    );
    assert!(
        !values("model:")
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value["value"] == "\"model with spaces\"")
    );
    let mut message: serde_json::Value = serde_json::from_str(
        &polylogue_testkit::claude_assistant("a1", "s1", "synthetic answer"),
    )
    .unwrap();
    message["message"]["model"] = json!("model with spaces");
    std::fs::write(
        d.home.join(".claude/projects/-p/s1.jsonl"),
        format!(
            "{}{}\n",
            polylogue_testkit::claude_user("u1", "s1", "synthetic prompt"),
            message
        ),
    )
    .unwrap();
    d.wait_for(|s| {
        s["store"]["messages"] == 2
            && values("model:model")
                .as_array()
                .unwrap()
                .iter()
                .any(|value| value["value"] == "\"model with spaces\"")
    });
    assert!(
        values("model:model")
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "\"model with spaces\"")
    );
}

#[test]
fn completion_request_preserves_full_text_offsets_and_reports_invalid_boundaries() {
    let daemon = crate::test_daemon::start("synthetic prompt");
    let text = "tool_calls where outcome:err | count";
    let position = text.find(" | ").unwrap();
    match crate::daemon::service::handle(
        crate::daemon::Request::Complete {
            text: text.into(),
            position,
        },
        &daemon.daemon,
    ) {
        crate::daemon::Response::Ok { data, .. } => {
            let candidate = data
                .as_array()
                .unwrap()
                .iter()
                .find(|candidate| candidate["value"] == "error")
                .unwrap();
            let start = candidate["replace"]["start"].as_u64().unwrap() as usize;
            let end = candidate["replace"]["end"].as_u64().unwrap() as usize;
            assert_eq!(candidate["kind"], "value");
            assert_eq!(
                format!(
                    "{}{}{}",
                    &text[..start],
                    candidate["value"].as_str().unwrap(),
                    &text[end..]
                ),
                "tool_calls where outcome:error | count"
            );
        }
        other => panic!("{other:?}"),
    }
    for position in [6, 8] {
        match crate::daemon::service::handle(
            crate::daemon::Request::Complete {
                text: "repo:λ".into(),
                position,
            },
            &daemon.daemon,
        ) {
            crate::daemon::Response::Error { kind, .. } => assert_eq!(kind, "bad_request"),
            other => panic!("{other:?}"),
        }
    }
}
