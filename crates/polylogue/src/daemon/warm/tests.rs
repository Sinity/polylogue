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
            "internal complete",
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
        "internal complete",
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
            "internal complete",
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
            &polylogue_model::WallClock::new(jiff::Timestamp::now),
        )
        .unwrap();
}

fn wait_for_vector_wait(limit: &polylogue_store::read::limit::ReadLimit) {
    let until = std::time::Instant::now() + Duration::from_secs(10);
    while limit.stage() != Some("vector wait") {
        assert!(
            std::time::Instant::now() < until,
            "request did not wait for vectors"
        );
        std::thread::yield_now();
    }
}

#[test]
fn two_requests_for_newer_vectors_load_them_once() {
    use crate::test_daemon::embeddings::{VectorLoad, vector_query};
    let daemon = crate::test_daemon::start("parser text");
    let provider = daemon.local_embeddings();
    let before = crate::test_answers::ok(crate::test_answers::listing(
        vector_query().answer(&daemon.daemon),
    ));
    provider.join().unwrap();
    put_vector(&daemon, "local/synthetic/2/document/unit", 255, [0., 1.]);
    let loading = VectorLoad::start(&daemon);
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
    let answer = std::thread::scope(|scope| {
        let reading = scope.spawn(|| vector_query().within(&limit).answer(&daemon.daemon));
        wait_for_vector_wait(&limit);
        crate::test_answers::ok(crate::test_answers::listing(loading.finish()));
        reading.join().unwrap()
    });
    let rows = crate::test_answers::ok(crate::test_answers::listing(answer));
    assert_eq!(rows["rows"][0]["id"], "claude-code:u1");
    assert_eq!(rows["count"], 1);
    assert_eq!(rows["rows"], before["rows"]);
    let status = crate::test_answers::ok(crate::test_daemon::call(
        &daemon.daemon,
        "status",
        json!({}),
    ));
    assert_eq!(status["warm"]["vector_loads"], 1);
    assert_eq!(status["warm"]["vectors"]["vectors"], 2);
}

#[test]
fn similar_query_on_morrow_keeps_its_rows_after_vectors_are_extended() {
    let daemon = crate::test_daemon::start_with_sources_and_workers(
        "parser text",
        &|home| {
            let name = "7f3a2c10-0001-4000-8000-000000000001.jsonl";
            std::fs::copy(
                polylogue_testkit::morrow_dir()
                    .join("claude-code/projects/-home-op-morrow")
                    .join(name),
                home.join(".claude/projects/-p").join(name),
            )
            .unwrap();
        },
        12,
        2,
    );
    let provider = daemon.local_embeddings();
    let query = || {
        crate::test_answers::ok(crate::test_answers::listing(crate::test_daemon::call(
            &daemon.daemon,
            "rows",
            json!({"selection": "messages where session:claude-code:7f3a2c10-0001-4000-8000-000000000001 similar:parser"}),
        )))
    };
    let before = query();
    provider.join().unwrap();
    assert_eq!(before["count"], 12);
    let ids = before["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    // Messages 4 and 8 contain only tool calls, so they follow ranked prose.
    let expected = [12, 11, 10, 9, 7, 6, 5, 3, 2, 1, 8, 4]
        .into_iter()
        .map(|n| format!("claude-code:c1000000-0000-4000-8000-{n:012}"))
        .collect::<Vec<_>>();
    assert_eq!(ids, expected);
    assert_eq!(before["unknown"]["similar"], 2);
    put_vector(&daemon, "local/synthetic/2/document/unit", 255, [0., 1.]);
    let after = query();
    assert_eq!(after["rows"], before["rows"]);
    assert_eq!(after["count"], 12);
    let status = crate::test_answers::ok(crate::test_daemon::call(
        &daemon.daemon,
        "status",
        json!({}),
    ));
    assert_eq!(status["warm"]["vector_loads"], 1);
}

#[test]
fn cancelled_request_leaves_while_another_request_loads_vectors() {
    use crate::test_daemon::embeddings::{VectorLoad, vector_query};
    use polylogue_store::{cancellation::Cancellation, read::limit::ReadLimit};
    let daemon = crate::test_daemon::start("parser text");
    let loading = VectorLoad::start(&daemon);
    let cancel = Cancellation::default();
    let limit = ReadLimit::without_deadline(cancel.clone());
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            done.send(vector_query().within(&limit).answer(&daemon.daemon))
                .unwrap()
        });
        wait_for_vector_wait(&limit);
        cancel.cancel();
        let answer = finished.recv_timeout(Duration::from_secs(10));
        crate::test_answers::ok(crate::test_answers::listing(loading.finish()));
        let answer = answer.expect("cancelled request waited for the other load");
        match answer.outcome {
            command_surface::Outcome::Error(error) => {
                assert_eq!(error.code.as_deref(), Some("cancelled"))
            }
            other => panic!("{other:?}"),
        }
    });
    let rows = crate::test_answers::ok(crate::test_answers::listing(
        vector_query().answer(&daemon.daemon),
    ));
    assert_eq!(rows["rows"][0]["id"], "claude-code:u1");
}

#[test]
fn a_request_deadline_ends_its_vector_wait() {
    use crate::test_daemon::embeddings::{VectorLoad, vector_query};
    let daemon = crate::test_daemon::start_with_sources_and_workers(
        "parser text",
        &|home| {
            let name = "7f3a2c10-0001-4000-8000-000000000001.jsonl";
            std::fs::copy(
                polylogue_testkit::morrow_dir()
                    .join("claude-code/projects/-home-op-morrow")
                    .join(name),
                home.join(".claude/projects/-p").join(name),
            )
            .unwrap();
        },
        12,
        2,
    );
    let loading = VectorLoad::start(&daemon);
    let limit =
        polylogue_store::read::limit::ReadLimit::new(Duration::from_secs(1), Default::default());
    let answer = vector_query().within(&limit).answer(&daemon.daemon);
    crate::test_answers::ok(crate::test_answers::listing(loading.finish()));
    assert_eq!(answer.exit_code(), 7, "{answer:?}");
    assert_eq!(limit.stage(), Some("vector wait"));
    match answer.outcome {
        command_surface::Outcome::Error(error) => {
            assert_eq!(error.code.as_deref(), Some("deadline_exceeded"));
            assert_eq!(error.kind.name(), "timeout");
            assert_eq!(error.message, "the read deadline was exceeded");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn stopping_the_daemon_ends_a_request_waiting_for_vectors() {
    use crate::test_daemon::embeddings::{VectorLoad, vector_query};
    let daemon = crate::test_daemon::start("parser text");
    let loading = VectorLoad::start(&daemon);
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(
        daemon.daemon.archive.stop().child(),
    );
    let (done, finished) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            done.send(vector_query().within(&limit).answer(&daemon.daemon))
                .unwrap()
        });
        wait_for_vector_wait(&limit);
        daemon.daemon.archive.stop().cancel();
        let waiting = finished.recv_timeout(Duration::from_secs(10));
        let loading = loading.finish();
        for answer in [waiting.expect("daemon stop did not end the wait"), loading] {
            match answer.outcome {
                command_surface::Outcome::Error(error) => {
                    assert_eq!(error.code.as_deref(), Some("shutting_down"))
                }
                other => panic!("{other:?}"),
            }
        }
    });
}

#[test]
fn clearing_during_a_load_does_not_publish_its_vectors() {
    use crate::test_daemon::embeddings::{VectorLoad, vector_query};
    let daemon = crate::test_daemon::start("parser text");
    let loading = VectorLoad::start(&daemon);
    daemon.daemon.warm.clear();
    crate::test_answers::ok(crate::test_answers::listing(loading.finish()));
    let status = crate::test_answers::ok(crate::test_daemon::call(
        &daemon.daemon,
        "status",
        json!({}),
    ));
    assert_eq!(status["warm"]["vectors"], serde_json::Value::Null);
    crate::test_answers::listing(vector_query().answer(&daemon.daemon));
    let status = crate::test_answers::ok(crate::test_daemon::call(
        &daemon.daemon,
        "status",
        json!({}),
    ));
    assert_eq!(status["warm"]["vector_loads"], 2);
}

#[test]
fn older_vector_snapshot_does_not_replace_newer_cached_values() {
    // The overlapping snapshots use a pool independent of daemon workers.
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = polylogue_store::Archive::open_or_create(dir.path()).unwrap();
    installed(&archive);
    let clock = polylogue_model::WallClock::new(jiff::Timestamp::now);
    let warm = Warm::new(&archive, clock.clone());
    let reads = polylogue_store::ReadPool::open(&archive).unwrap();
    let mut store = polylogue_store::vectors::Vectors::writer(&archive, 256).unwrap();
    let key = "synthetic/model/2/document/unit";
    let mut put = |hash, values: [f32; 2]| {
        store
            .put(
                key,
                &[polylogue_store::vectors::Vector {
                    text_hash: vec![hash],
                    chunk: 0,
                    values: values.to_vec(),
                }],
                &clock,
            )
            .unwrap();
    };
    put(1, [1., 0.]);
    // A read whose snapshot predates the next write.
    let older = reads.reader().unwrap();
    let older_revision = older.vectors().unwrap().revision(key).unwrap();
    put(2, [0., 1.]);
    let reader = reads.reader().unwrap();
    let current = warm.vectors_of(&reader, key).unwrap();
    assert!(current.revision > older_revision);
    assert_eq!(current.vectors(), 2);
    // The older read loads its own set, which serves it and is not kept.
    let old = warm.vectors_of(&older, key).unwrap();
    assert_eq!(old.revision, older_revision);
    assert_eq!(old.vectors(), 1);
    let cache = warm.vectors.lock().unwrap();
    assert_eq!(cache.sets.len(), 1);
    assert!(Arc::ptr_eq(&cache.sets[0], &current));
}

#[test]
fn vector_write_appends_without_reload() {
    use polylogue_store::vectors::Vector;
    let daemon = crate::test_daemon::start("vector appends");
    let warm = Warm::new(&daemon.daemon.archive, daemon.daemon.clock.clone());
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
                &polylogue_model::WallClock::new(jiff::Timestamp::now),
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
                &polylogue_model::WallClock::new(jiff::Timestamp::now),
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
        warm.summary().vector_loads,
        1,
        "every later revision extended the held vectors"
    );
    // The first read's vectors are as they were.
    assert_eq!(first.vectors(), 1);
}

#[test]
fn summary_does_not_wait_for_a_load() {
    let daemon = crate::test_daemon::start("status during a vector load");
    let loading = crate::test_daemon::embeddings::VectorLoad::start(&daemon);
    let status = crate::test_answers::ok(crate::test_daemon::call(
        &daemon.daemon,
        "status",
        json!({}),
    ));
    assert_eq!(status["warm"]["vector_loads"], 1);
    assert_eq!(status["warm"]["vectors"], serde_json::Value::Null);
    crate::test_answers::ok(crate::test_answers::listing(loading.finish()));
}

#[test]
fn one_provider_call_per_query_text_in_flight() {
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
        let until = std::time::Instant::now() + Duration::from_secs(10);
        loop {
            let waiting = d.warm.summary().query_vectors_reused as usize;
            if waiting == readers - 1 {
                break;
            }
            assert!(
                std::time::Instant::now() < until,
                "the other reads never waited"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
        crate::test_daemon::embeddings::serve_request(
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
                        "rows",
                        serde_json::json!({"selection": format!("messages where similar:\"slow {n}\"")}),
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
            crate::test_daemon::embeddings::serve_request(
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
                        crate::test_daemon::embeddings::serve_request(
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
            "rows",
            json!({"selection": "messages where similar:\"explained\"", "explain":true}),
        ));
        assert_eq!(calls.load(Ordering::Acquire), 0, "explain");
        // A query that parses but is refused runs nothing.
        let refused = crate::test_daemon::call(
            d,
            "rows",
            json!({"selection": "messages where no_such_field:1 similar:\"refused\""}),
        );
        assert!(
            matches!(refused.outcome, command_surface::Outcome::Error(_)),
            "{refused:?}"
        );
        assert_eq!(calls.load(Ordering::Acquire), 0, "refused query");
        // An executed ranking asks once: the run uses the vector asked for.
        crate::test_answers::ok(crate::test_daemon::call(
            d,
            "rows",
            json!({"selection": "messages where similar:\"executed\""}),
        ));
        assert_eq!(calls.load(Ordering::Acquire), 1, "executed query");
    });
}

#[test]
fn ranking_with_model_calls_off_uses_stored_vectors() {
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    let daemon = crate::test_daemon::start("parser text");
    let listener = daemon.embedding_endpoint();
    listener.set_nonblocking(true).unwrap();
    let calls = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let d = &daemon.daemon;
    let query = |text: &str| {
        crate::test_answers::listing(crate::test_daemon::call(
            d,
            "rows",
            json!({ "selection": text }),
        ))
    };
    std::thread::scope(|scope| {
        // A provider that counts the calls it answers.
        scope.spawn(|| {
            while !stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((peer, _)) => {
                        peer.set_nonblocking(false).unwrap();
                        calls.fetch_add(1, Ordering::AcqRel);
                        crate::test_daemon::embeddings::serve_request(
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
        struct Stop<'a>(&'a AtomicBool);
        impl Drop for Stop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }
        let _stop = Stop(&stop);
        // With calls on, a new text is asked for once and stored.
        let asked = crate::test_answers::ok(query("messages where similar:\"parser\""));
        assert_eq!(calls.load(Ordering::Acquire), 1);
        // Calls off, and nothing kept in memory: the stored vector ranks the
        // same rows in the same order without a call.
        daemon.set("embed.calls", "off");
        d.warm.clear();
        let stored = crate::test_answers::ok(query("messages where similar:\"parser\""));
        assert_eq!(stored["rows"], asked["rows"], "{stored}");
        assert_eq!(calls.load(Ordering::Acquire), 1, "a stored text");
        // A text never stored is unranked, and says why, without a call.
        let (_, notes) =
            crate::test_answers::degraded(query("messages where similar:\"never asked\""));
        assert!(
            notes
                .iter()
                .any(|note| note
                    .ends_with("semantic search for new text is off (model calls are off)")),
            "{notes:?}"
        );
        let refused = crate::test_daemon::call(d, "embed", json!({ "yes": true }));
        assert!(
            matches!(refused.outcome, command_surface::Outcome::Error(_)),
            "{refused:?}"
        );
        assert_eq!(calls.load(Ordering::Acquire), 1, "calls off");
    });
}

fn restored_vector(hash: u8, at: i64, values: [f32; 2]) -> polylogue_store::vectors::StoredVector {
    polylogue_store::vectors::StoredVector {
        text_hash: vec![hash],
        model: "synthetic/model/2/document/unit".into(),
        chunk: 0,
        dims: 2,
        vector: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        embedded_at: at,
    }
}

#[test]
fn restore_without_vector_demand_never_starts_a_model_reload_per_commit() {
    let dir = tempfile::tempdir().unwrap();
    let (archive, _lock) = polylogue_store::Archive::open_or_create(dir.path()).unwrap();
    installed(&archive);
    let warm = Warm::new(
        &archive,
        polylogue_model::WallClock::new(jiff::Timestamp::now),
    );
    let mut store = polylogue_store::vectors::Vectors::writer(&archive, 256).unwrap();
    for at in 1..=600 {
        let mut batch = vec![restored_vector(
            1,
            at,
            if at % 2 == 0 { [1., 0.] } else { [0., 1.] },
        )];
        if at % 100 == 0 {
            batch.push(restored_vector((at / 100 + 1) as u8, at, [0., 1.]));
        }
        store.restore(&batch).unwrap();
    }
    assert!(warm.summary().vectors.is_none());
    assert_eq!(warm.summary().vector_loads, 0);
    let reads = polylogue_store::ReadPool::open(&archive).unwrap();
    let reader = reads.reader().unwrap();
    let loaded = warm
        .vectors_of(&reader, "synthetic/model/2/document/unit")
        .unwrap();
    assert_eq!(
        (loaded.vectors(), loaded.rows_read),
        (7, 7),
        "one initial demand reads only the final rows"
    );
    assert_eq!(warm.summary().vector_loads, 1);
    let revision = loaded.revision;
    drop(reader);
    store.restore(&[restored_vector(1, 600, [1., 0.])]).unwrap();
    let next = reads.reader().unwrap();
    let repeated = warm
        .vectors_of(&next, "synthetic/model/2/document/unit")
        .unwrap();
    assert_eq!(repeated.revision, revision);
    assert!(Arc::ptr_eq(&loaded, &repeated));
}

#[test]
fn a_warmed_model_refreshes_mixed_restore_and_preserves_old_reader_values() {
    let daemon = crate::test_daemon::start("paid vector restore");
    let key = "synthetic/model/2/document/unit";
    let mut store = polylogue_store::vectors::Vectors::writer(&daemon.daemon.archive, 256).unwrap();
    store
        .restore(&[
            restored_vector(1, 10, [1., 0.]),
            restored_vector(2, 10, [0., 1.]),
        ])
        .unwrap();
    let warm = Warm::new(&daemon.daemon.archive, daemon.daemon.clock.clone());
    let old_reader = daemon.daemon.reads.reader().unwrap();
    let old = warm.vectors_of(&old_reader, key).unwrap();
    // Keep the old transaction open even on hosts with a single reader slot.
    let fresh_reads = polylogue_store::ReadPool::open(&daemon.daemon.archive).unwrap();
    store
        .restore(&[
            restored_vector(1, 20, [0., 1.]),
            restored_vector(3, 20, [-1., 0.]),
            restored_vector(2, 30, [0., 1.]),
        ])
        .unwrap();
    let reader = fresh_reads.reader().unwrap();
    let current = warm.vectors_of(&reader, key).unwrap();
    drop(reader);
    assert_eq!(
        current.revision,
        old.revision + 3,
        "row triggers count all three writes, not one transaction"
    );
    assert_eq!(
        (current.vectors(), current.rows_read),
        (3, 4),
        "one appended-row probe plus one full model refresh"
    );
    assert_eq!(current.similarity(&[1., 0.]).unwrap().of(&[1]), Some(0.));
    assert_eq!(old.similarity(&[1., 0.]).unwrap().of(&[1]), Some(1.));
    assert!(Arc::ptr_eq(
        &warm.vectors_of(&old_reader, key).unwrap(),
        &old
    ));
    // A paid-time-only change still creates a truthful new snapshot.
    store.restore(&[restored_vector(1, 40, [0., 1.])]).unwrap();
    let newer_reader = fresh_reads.reader().unwrap();
    let newer = warm.vectors_of(&newer_reader, key).unwrap();
    drop(newer_reader);
    assert_eq!(newer.revision, current.revision + 1);
    assert_eq!(newer.rows_read, 3);
    assert_eq!(newer.similarity(&[1., 0.]).unwrap().of(&[1]), Some(0.));
    assert_eq!(
        warm.vectors_of(&old_reader, key)
            .unwrap()
            .similarity(&[1., 0.])
            .unwrap()
            .of(&[1]),
        Some(1.)
    );
}

/// Installs a new archive's empty build, as a daemon's first pass over
/// nothing does, so reads are served.
fn installed(archive: &polylogue_store::Archive) {
    polylogue_store::Writer::start(archive)
        .unwrap()
        .build(Vec::new())
        .unwrap();
}
