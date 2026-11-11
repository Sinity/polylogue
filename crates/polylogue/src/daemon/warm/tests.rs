use super::*;
use serde_json::json;
use std::time::Duration;

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

fn put_vector(daemon: &crate::test_daemon::TestDaemon, key: &str, hash: u8, values: [f32; 2]) {
    let reader = daemon.daemon.reads.reader().unwrap();
    reader
        .vectors_writer()
        .unwrap()
        .put(
            key,
            &[polylogue_store::vectors::Vector {
                text_hash: vec![hash],
                chunk: 0,
                values: values.to_vec(),
            }],
        )
        .unwrap();
}

#[test]
fn vector_loading_leaves_the_cache_unlocked_and_clear_prevents_publication() {
    let daemon = crate::test_daemon::start("vector cache ownership");
    let warm = Warm::new(&daemon.daemon.archive);
    let (started, entered) = std::sync::mpsc::channel();
    let (release, released) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let warm = &warm;
        let daemon = &daemon;
        let reading = scope.spawn(move || {
            let reader = daemon.daemon.reads.reader().unwrap();
            warm.load_vectors_until(&reader, started, released);
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        let unlocked = warm.vectors.cache.try_lock().is_ok();
        if unlocked {
            warm.clear();
        }
        release.send(()).unwrap();
        reading.join().unwrap();
        assert!(unlocked, "loading must not hold the shared cache mutex");
        let cache = warm.vectors.cache.lock().unwrap();
        // The cleared pass published nothing; the read asked again.
        assert_eq!(
            cache.passes, 2,
            "clear invalidates an in-flight publication"
        );
        assert_eq!(cache.sets.len(), 1);
        assert_eq!(cache.sets[0].vectors(), 0);
    });
}

#[test]
fn older_vector_snapshot_does_not_replace_newer_cached_values() {
    let daemon = crate::test_daemon::start("vector snapshot ordering");
    let warm = Warm::new(&daemon.daemon.archive);
    let key = "synthetic/model/2/document/unit";
    put_vector(&daemon, key, 1, [1., 0.]);
    // A read whose snapshot predates the next write.
    let older = daemon.daemon.reads.reader().unwrap();
    let older_revision = older.vectors().unwrap().revision(key).unwrap();
    put_vector(&daemon, key, 2, [0., 1.]);
    let reader = daemon.daemon.reads.reader().unwrap();
    let current = warm.vectors_of(&reader, key).unwrap();
    assert!(current.revision > older_revision);
    assert_eq!(current.vectors(), 2);
    // The older read loads its own set, which serves it and is not kept.
    let old = warm.vectors_of(&older, key).unwrap();
    assert_eq!(old.revision, older_revision);
    assert_eq!(old.vectors(), 1);
    let cache = warm.vectors.cache.lock().unwrap();
    assert_eq!(cache.sets.len(), 1);
    assert!(Arc::ptr_eq(&cache.sets[0], &current));
}

#[test]
fn a_read_after_a_revision_commit_never_extends_inline() {
    let daemon = crate::test_daemon::start("vectors worker");
    let warm = Warm::new(&daemon.daemon.archive);
    let key = "synthetic/model/2/document/unit";
    put_vector(&daemon, key, 1, [1., 0.]);
    let read = || {
        let reader = daemon.daemon.reads.reader().unwrap();
        warm.vectors_of(&reader, key).unwrap()
    };
    assert_eq!(read().vectors(), 1);
    for hash in 2..=5 {
        let held = warm.summary().vectors.unwrap().vectors;
        put_vector(&daemon, key, hash, [0., 1.]);
        // The worker extends the set on the commit, before any read asks.
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while warm.summary().vectors.unwrap().vectors == held {
            assert!(
                std::time::Instant::now() < deadline,
                "the worker did not extend the set"
            );
            std::thread::sleep(Duration::from_millis(2));
        }
        let loads = super::loads_here();
        let current = read();
        assert_eq!(
            super::loads_here(),
            loads,
            "the read loaded or extended vectors itself"
        );
        assert_eq!(current.vectors(), usize::from(hash));
        assert_eq!(current.rows_read, 1, "the worker read only the new vector");
        // The set the read got ranks like a full load of its snapshot.
        let reader = daemon.daemon.reads.reader().unwrap();
        let whole = Arc::new(reader.vectors().unwrap().load(key).unwrap());
        assert_eq!(current.revision, whole.revision);
        let ranked = current.similarity(&[0.6, 0.8]).unwrap();
        let loaded = whole.similarity(&[0.6, 0.8]).unwrap();
        for text in 1..=hash {
            assert_eq!(ranked.of(&[text]), loaded.of(&[text]), "text {text}");
        }
    }
    assert_eq!(warm.counts.vector_loads.load(Ordering::Relaxed), 1);
}

#[test]
fn vector_write_appends_without_reload() {
    use polylogue_store::vectors::Vector;
    let daemon = crate::test_daemon::start("vector appends");
    let warm = Warm::new(&daemon.daemon.archive);
    let key = "synthetic/model/2/document/unit";
    let put = |hash: u8, values: [f32; 2]| {
        let reader = daemon.daemon.reads.reader().unwrap();
        reader
            .vectors_writer()
            .unwrap()
            .put(
                key,
                &[Vector {
                    text_hash: vec![hash],
                    chunk: 0,
                    values: values.to_vec(),
                }],
            )
            .unwrap();
    };
    put(1, [1., 0.]);
    let read = || {
        let reader = daemon.daemon.reads.reader().unwrap();
        warm.vectors_of(&reader, key).unwrap()
    };
    // What a full load of the same snapshot ranks: the appended set must
    // rank every text the same.
    let agree = |current: Loaded, texts: u8| {
        let reader = daemon.daemon.reads.reader().unwrap();
        let whole = Arc::new(reader.vectors().unwrap().load(key).unwrap());
        assert_eq!(current.vectors(), whole.vectors());
        let appended = Arc::new(current).similarity(&[0.6, 0.8]).unwrap();
        let loaded = whole.similarity(&[0.6, 0.8]).unwrap();
        for hash in 1..=texts {
            assert!(appended.of(&[hash]).is_some(), "text {hash}");
            assert_eq!(appended.of(&[hash]), loaded.of(&[hash]), "text {hash}");
        }
        assert_eq!(appended.of(&[texts + 1]), None);
    };
    let first = read();
    assert_eq!(first.vectors(), 1);
    // Enough appends to merge runs: every read sees each vector once.
    for hash in 2..=9 {
        put(hash, [0., 1.]);
        let current = read();
        assert_eq!(current.vectors(), usize::from(hash));
        assert_eq!(current.rows_read, 1, "only the new vector is read");
        agree((*current).clone(), hash);
    }
    // A text's second chunk, stored later, ranks it by its best chunk.
    {
        let reader = daemon.daemon.reads.reader().unwrap();
        reader
            .vectors_writer()
            .unwrap()
            .put(
                key,
                &[Vector {
                    text_hash: vec![1],
                    chunk: 1,
                    values: vec![0., 1.],
                }],
            )
            .unwrap();
    }
    let current = read();
    assert_eq!(current.vectors(), 10);
    assert_eq!(current.rows_read, 1);
    let similarity = Arc::new((*current).clone()).similarity(&[0., 1.]).unwrap();
    assert_eq!(similarity.of(&[1]), similarity.of(&[2]));
    agree((*current).clone(), 9);
    assert_eq!(
        warm.counts.vector_loads.load(Ordering::Relaxed),
        1,
        "every later revision extended the held vectors"
    );
    // The first read's vectors are as they were.
    assert_eq!(first.vectors(), 1);
}

#[test]
fn summary_does_not_wait_for_a_load() {
    let daemon = crate::test_daemon::start("status during a vector load");
    let warm = Warm::new(&daemon.daemon.archive);
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

#[test]
fn one_provider_call_per_query_text_in_flight() {
    use std::sync::atomic::AtomicUsize;
    let daemon = crate::test_daemon::start("parser text");
    let listener = daemon.embedding_endpoint();
    let d = &daemon.daemon;
    let provider = {
        let reader = d.reads.reader().unwrap();
        let settings = reader.settings_snapshot(&daemon.config.machine).unwrap();
        crate::embeddings::Provider::from_settings(&settings, &d.config.embedding_credentials)
            .unwrap()
            .expect("the local provider is configured")
    };
    let calls = AtomicUsize::new(0);
    // Each read holds a reader slot while it waits, so this stays within the pool.
    let readers = 2;
    let answers: Vec<_> = std::thread::scope(|scope| {
        let reads: Vec<_> = (0..readers)
            .map(|_| {
                scope.spawn(|| {
                    let reader = d.reads.reader().unwrap();
                    d.warm.similarity(&reader, &provider, "parser").map(drop)
                })
            })
            .collect();
        // The provider answers its first call only once every other read of
        // the same text waits for that call.
        let (peer, _) = listener.accept().unwrap();
        calls.fetch_add(1, Ordering::Relaxed);
        let until = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let waiting = d
                .warm
                .in_flight
                .lock()
                .unwrap()
                .values()
                .map(|flight| flight.waiting.load(Ordering::Relaxed))
                .sum::<usize>();
            if waiting == readers - 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < until,
                "the other reads never waited"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        crate::test_daemon::embeddings::response(
            peer,
            200,
            r#"{"data":[{"index":0,"embedding":[1.0,0.0]}]}"#,
        );
        reads.into_iter().map(|read| read.join().unwrap()).collect()
    });
    listener.set_nonblocking(true).unwrap();
    assert!(
        listener.accept().is_err(),
        "a second provider call was made"
    );
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(answers, vec![Ok(()); readers]);
}

#[test]
fn slow_provider_calls_cannot_fill_the_read_pool() {
    let daemon = crate::test_daemon::start("parser text");
    let listener = daemon.embedding_endpoint();
    let d = &daemon.daemon;
    // More slow calls in flight than the pool has readers.
    let calls = d.reads.max_readers() + 1;
    let answers: Vec<_> = std::thread::scope(|scope| {
        let queries: Vec<_> = (0..calls)
            .map(|n| {
                scope.spawn(move || {
                    crate::test_daemon::call(
                        d,
                        "query",
                        serde_json::json!({"query": format!("messages where similar:\"slow {n}\"")}),
                    )
                })
            })
            .collect();
        // The provider answers nothing until every call is waiting on it; a
        // call that holds a reader slot while it waits keeps the last one out.
        let until = std::time::Instant::now() + Duration::from_secs(10);
        listener.set_nonblocking(true).unwrap();
        let mut peers = Vec::new();
        while peers.len() < calls {
            match listener.accept() {
                Ok((peer, _)) => {
                    peer.set_nonblocking(false).unwrap();
                    peers.push(peer);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < until,
                        "{} of {calls} calls reached the provider",
                        peers.len()
                    );
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("{error}"),
            }
        }
        while d.reads.checked_out() > 0 {
            assert!(
                std::time::Instant::now() < until,
                "reads waiting on the provider hold reader slots"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        drop(d.reads.reader().unwrap());
        for peer in peers {
            crate::test_daemon::embeddings::response(
                peer,
                200,
                r#"{"data":[{"index":0,"embedding":[1.0,0.0]}]}"#,
            );
        }
        queries
            .into_iter()
            .map(|query| query.join().unwrap())
            .collect()
    });
    for answer in answers {
        crate::test_answers::ok(answer);
    }
}

#[test]
fn only_an_executed_ranking_asks_the_provider() {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    let daemon = crate::test_daemon::start("parser text");
    let listener = daemon.embedding_endpoint();
    listener.set_nonblocking(true).unwrap();
    let calls = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let d = &daemon.daemon;
    std::thread::scope(|scope| {
        scope.spawn(|| {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((peer, _)) => {
                        peer.set_nonblocking(false).unwrap();
                        calls.fetch_add(1, Ordering::AcqRel);
                        crate::test_daemon::embeddings::response(
                            peer,
                            200,
                            r#"{"data":[{"index":0,"embedding":[1.0,0.0]}]}"#,
                        );
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("{error}"),
                }
            }
        });
        // The provider stops serving when the checks end, failed or not.
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _stop = Stop(&stop);
        // Explaining a ranking runs nothing.
        crate::test_answers::ok(crate::test_daemon::call(
            d,
            "explain",
            json!({"subject": "query", "expression": "messages where similar:\"explained\""}),
        ));
        assert_eq!(calls.load(Ordering::Acquire), 0, "explain");
        // A query that parses but is refused runs nothing.
        let refused = crate::test_daemon::call(
            d,
            "query",
            json!({"query": "messages where no_such_field:1 similar:\"refused\""}),
        );
        assert!(
            matches!(refused.outcome, command_surface::Outcome::Error(_)),
            "{refused:?}"
        );
        assert_eq!(calls.load(Ordering::Acquire), 0, "refused query");
        // An executed ranking asks once: the run uses the vector asked for.
        crate::test_answers::ok(crate::test_daemon::call(
            d,
            "query",
            json!({"query": "messages where similar:\"executed\""}),
        ));
        assert_eq!(calls.load(Ordering::Acquire), 1, "executed query");
    });
}
