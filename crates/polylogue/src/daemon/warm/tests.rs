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
fn caught_up_failures_are_visible_and_a_success_clears_them() {
    let daemon = crate::test_daemon::start("caught-up failure");
    let warm = Warm::default();
    warm.refresh_caught_up(&daemon.daemon.reads);
    assert!(warm.caught_up_error().is_none());
    let closed = polylogue_store::ReadPool::open(&daemon.daemon.archive).unwrap();
    closed.close();
    warm.refresh_caught_up(&closed);
    assert!(warm.caught_up_error().is_some());
    warm.refresh_caught_up(&daemon.daemon.reads);
    assert!(warm.caught_up_error().is_none());
}

#[test]
fn a_read_at_a_newer_position_has_the_worker_check_at_it() {
    let daemon = crate::test_daemon::start("caught-up at a newer position");
    let warm = Warm::default();
    warm.refresh_caught_up(&daemon.daemon.reads);
    warm.recheck_wanted.store(false, Ordering::Release);
    let before = warm.caught_up.lock().unwrap().clone().unwrap().0;
    daemon.set("browser.actions", "on");
    let reader = daemon.daemon.reads.reader().unwrap();
    let position = reader.change_position().unwrap().to_string();
    assert_ne!(position, before);
    // The read answers with what was last found and asks for a newer answer.
    warm.caught_up_for_read(&reader, &position).unwrap();
    assert!(warm.recheck_wanted.load(Ordering::Acquire));
    drop(reader);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| warm.caught_up_worker(&daemon.daemon.reads, None));
        while warm.caught_up.lock().unwrap().as_ref().unwrap().0 != position {
            std::thread::sleep(Duration::from_millis(10));
        }
        warm.stop_checking();
        worker.join().unwrap();
    });
}

#[test]
fn completion_reads_declared_values_and_refreshes_observed_models() {
    let d = crate::test_daemon::start("synthetic prompt");
    let values = |word: &str| {
        crate::test_daemon::Call::new(
            "complete",
            serde_json::json!({"text": word, "position": word.len()}),
        )
        .answer(&d.daemon)
    };
    assert!(
        crate::test_answers::ok(values("live:"))
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "waiting"
                && v["replace"]["start"] == 5
                && v["replace"]["end"] == 5)
    );
    // No session names a model yet: the field's existence predicates are offered.
    assert_eq!(
        crate::test_answers::ok(values("model:"))
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value["value"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["*", "known", "unknown"]
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
            && matches!(
                values("model:model"),
                command_surface::Answer { outcome: command_surface::Outcome::Ok(value), .. } if value
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|value| value["value"] == "\"model with spaces\"")
            )
    });
    assert!(
        crate::test_answers::ok(values("model:model"))
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
    match crate::test_daemon::Call::new(
        "complete",
        serde_json::json!({"text": text, "position": position}),
    )
    .answer(&daemon.daemon)
    {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(data),
            ..
        } => {
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
        match crate::test_daemon::Call::new(
            "complete",
            serde_json::json!({"text": "repo:λ", "position": position}),
        )
        .answer(&daemon.daemon)
        {
            command_surface::Answer {
                outcome: command_surface::Outcome::Error(error),
                ..
            } => {
                assert_eq!(error.code.as_deref(), Some("bad_request"))
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn vector_loading_leaves_the_cache_unlocked_and_clear_prevents_publication() {
    let daemon = crate::test_daemon::start("vector cache ownership");
    let warm = Warm::default();
    let (started, entered) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let warm = &warm;
        let daemon = &daemon;
        let worker = scope.spawn(move || {
            let reader = daemon.daemon.reads.reader().unwrap();
            warm.cached_vectors(&reader, "synthetic/empty", 0, || {
                started.send(()).unwrap();
                released.recv().unwrap();
                reader
                    .vectors()
                    .unwrap()
                    .load("synthetic/empty")
                    .map_err(|e| e.to_string())
            })
            .unwrap()
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        let unlocked = warm.vectors.try_lock().is_ok();
        if unlocked {
            warm.clear();
        }
        release.send(()).unwrap();
        let result = worker.join().unwrap();
        assert!(unlocked, "loading must not hold the shared cache mutex");
        assert_eq!(result.vectors(), 0);
        assert!(
            warm.vectors.lock().unwrap().loaded.is_none(),
            "clear invalidates an in-flight publication"
        );
    });
}

#[test]
fn older_vector_snapshot_does_not_replace_newer_cached_values() {
    use polylogue_store::vectors::Vector;
    let daemon = crate::test_daemon::start("vector snapshot ordering");
    let warm = Warm::default();
    let reader = daemon.daemon.reads.reader().unwrap();
    let key = "synthetic/model/2/document/unit";
    let mut writer = reader.vectors_writer().unwrap();
    writer
        .put(
            key,
            &[Vector {
                text_hash: vec![1],
                chunk: 0,
                values: vec![1., 0.],
            }],
        )
        .unwrap();
    let first = writer.load(key).unwrap();
    writer
        .put(
            key,
            &[Vector {
                text_hash: vec![1],
                chunk: 0,
                values: vec![0., 1.],
            }],
        )
        .unwrap();
    let second = writer.load(key).unwrap();
    drop(writer);
    drop(reader);
    let older = first.revision;
    let newer = second.revision;
    let (started, entered) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let warm = &warm;
        let daemon = &daemon;
        let worker = scope.spawn(move || {
            let reader = daemon.daemon.reads.reader().unwrap();
            warm.cached_vectors(&reader, key, older, || {
                started.send(()).unwrap();
                released.recv().unwrap();
                Ok(first)
            })
            .unwrap()
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        let reader = daemon.daemon.reads.reader().unwrap();
        let current = warm
            .cached_vectors(&reader, key, newer, || Ok(second))
            .unwrap();
        release.send(()).unwrap();
        let old = worker.join().unwrap();
        assert_eq!(old.revision, older);
        assert_eq!(current.revision, newer);
        assert_eq!(
            warm.vectors
                .lock()
                .unwrap()
                .loaded
                .as_ref()
                .unwrap()
                .revision,
            newer
        );
        let reused = warm
            .cached_vectors(&reader, key, newer, || {
                panic!("current snapshot should reuse the cache")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&current, &reused));
        assert!(
            warm.cached_vectors(&reader, "synthetic/failure", 0, || Err(
                "cancelled load".into()
            ))
            .is_err()
        );
        assert_eq!(
            warm.vectors
                .lock()
                .unwrap()
                .loaded
                .as_ref()
                .unwrap()
                .revision,
            newer
        );
    });
}

#[test]
fn summary_does_not_wait_for_a_load() {
    let daemon = crate::test_daemon::start("status during a vector load");
    let warm = Warm::default();
    let before = serde_json::to_value(warm.summary()).unwrap();
    let (started, entered) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let warm = &warm;
        let daemon = &daemon;
        let loading = scope.spawn(move || {
            let reader = daemon.daemon.reads.reader().unwrap();
            warm.load_vectors_until(&reader, started, released);
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        let (summary, summarized) = std::sync::mpsc::channel();
        scope.spawn(move || summary.send(serde_json::to_value(warm.summary()).unwrap()));
        let during = summarized.recv_timeout(Duration::from_secs(5));
        release.send(()).unwrap();
        loading.join().unwrap();
        assert_eq!(during.expect("status waited for the vector load"), before);
    });
}
