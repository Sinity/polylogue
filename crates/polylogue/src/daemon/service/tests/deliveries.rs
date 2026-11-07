//! A delivery is recorded once per request: a hook that resends its request
//! after a lost answer gets the same delivery back. Its record is queued for
//! the writer, so a context answer never waits for a writer turn.

use super::*;
use crate::briefing::{Ask, StartSource};

/// Returns once the writer has applied every request queued before this
/// call: it answers a synchronous write only after the queued ones.
fn after_queued_writes(d: &test_daemon::TestDaemon) {
    crate::test_answers::ok(handle(
        Request::Keep {
            request_id: polylogue_store::user_writes::new_request_id(),
            text: "sessions".into(),
            name: None,
        },
        &d.daemon,
    ));
}

fn takeover(d: &test_daemon::TestDaemon, request_id: &str) -> serde_json::Value {
    let request = Request::Context {
        budget_tokens: None,
        request_id: request_id.into(),
        ask: Ask::Takeover {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            for_session: None,
        },
        sender: Some(polylogue_model::DeliverySender::Hook.as_str().into()),
    };
    crate::test_answers::ok(handle(request, &d.daemon))
}

fn deliveries(d: &test_daemon::TestDaemon) -> u64 {
    let request = Request::Query {
        text: "deliveries".into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        fields: vec![],
    };
    crate::test_answers::ok(handle(request, &d.daemon))["count"]
        .as_u64()
        .unwrap()
}

#[test]
fn a_resent_context_request_is_one_delivery() {
    let d = test_daemon::start("the walrus migration");
    let request_id = polylogue_store::user_writes::new_request_id();
    let first = takeover(&d, &request_id);
    let delivery = first["delivery"].as_str().unwrap().to_owned();
    let again = takeover(&d, &request_id);
    assert_eq!(again["delivery"], delivery.as_str());
    after_queued_writes(&d);
    assert_eq!(deliveries(&d), 1);
    let other = takeover(&d, &polylogue_store::user_writes::new_request_id());
    assert_ne!(other["delivery"], delivery.as_str());
    after_queued_writes(&d);
    assert_eq!(deliveries(&d), 2);
}

#[test]
fn an_accepted_prepared_receipt_crosses_the_read_deadline_without_a_false_refusal() {
    use polylogue_store::read::limit::ReadLimit;
    use std::time::{Duration, Instant};
    let d = test_daemon::start("receipt handoff");
    let blocker = std::fs::File::open(d.daemon.archive.archive_db()).unwrap();
    // Readers may start; the writer waits only at the register commit boundary.
    blocker.lock_shared().unwrap();
    let id = polylogue_store::user_writes::new_request_id();
    let limit = ReadLimit::new(Duration::from_secs(2), Default::default());
    let request = Request::Context {
        budget_tokens: None,
        request_id: id.clone(),
        ask: Ask::Takeover {
            session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
            for_session: None,
        },
        sender: Some("mcp".into()),
    };
    let (response, handed_off) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| super::super::handle_with_limit(request, &d.daemon, &limit));
        let until = Instant::now() + Duration::from_secs(5);
        while limit.stage() != Some("record delivery")
            && !worker.is_finished()
            && Instant::now() < until
        {
            std::thread::yield_now();
        }
        let handed_off = limit.stage() == Some("record delivery");
        if handed_off {
            while limit.check().is_ok() {
                std::thread::yield_now();
            }
        }
        drop(blocker);
        (worker.join().unwrap(), handed_off)
    });
    assert!(
        handed_off,
        "context never handed off its receipt: {response:?}"
    );
    assert!(matches!(
        limit.check(),
        Err(polylogue_store::StoreError::ReadDeadlineExceeded)
    ));
    let Some(data) = response.value().cloned() else {
        panic!("accepted receipt was reported as a refusal: {response:?}");
    };
    assert_eq!(data["delivery"], id);
    after_queued_writes(&d);
    let reader = d.daemon.reads.reader().unwrap();
    let saved = reader.user_request(&id).unwrap().unwrap();
    assert!(matches!(
        saved.write,
        polylogue_model::user::UserWrite::Deliver(_)
    ));
    let events = reader
        .query(
            "SELECT count(*) AS n FROM delivery_event WHERE delivery=?1",
            &[polylogue_store::SqlValue::Text(id)],
        )
        .unwrap();
    assert_eq!(events[0]["n"], 0, "prepared is not a delivered event");
}

/// A daemon whose archive also holds `claude-code:s9`, which worked in
/// `/home/op/proj`, with the scope index current.
fn daemon_with_a_project() -> test_daemon::TestDaemon {
    let d = test_daemon::start("the walrus migration");
    std::fs::write(
        d.home.join(".claude/projects/-p/s9.jsonl"),
        format!(
            "{}\n",
            serde_json::json!({
                "type": "user", "uuid": "u9", "sessionId": "s9",
                "timestamp": "2026-03-01T10:00:00Z", "cwd": "/home/op/proj",
                "message": {"role": "user", "content": "plan the otter release"}
            })
        ),
    )
    .unwrap();
    let started = std::time::Instant::now();
    d.wait_for(|status| {
        let ready = status["store"]["messages"] == 2 && status["store"]["search_pending"] == 0;
        assert!(
            ready || started.elapsed() < std::time::Duration::from_secs(20),
            "the second session was not stored; status: {status}"
        );
        ready
    });
    d.daemon
        .scope
        .refresh(&d.daemon.reads.reader().unwrap())
        .unwrap();
    d
}

fn brief(d: &test_daemon::TestDaemon, request_id: &str, source: StartSource) -> serde_json::Value {
    let request = Request::Context {
        budget_tokens: None,
        request_id: request_id.into(),
        ask: Ask::SessionStart {
            cwd: "/home/op/proj".into(),
            session: Some(polylogue_model::SessionId::parse("claude-code:newcomer").unwrap()),
            source,
        },
        sender: Some(polylogue_model::DeliverySender::Hook.as_str().into()),
    };
    match handle(request, &d.daemon) {
        command_surface::Answer {
            outcome: command_surface::Outcome::Ok(data),
            ..
        } => data,
        other => panic!("{other:?}"),
    }
}

/// Holds the writer in a turn: a write with this request ID waits at its
/// dispatch until the returned guard is released.
fn hold_writer(
    d: &test_daemon::TestDaemon,
    id: &'static str,
) -> (
    polylogue_store::writer::DispatchBarrier,
    std::thread::JoinHandle<()>,
) {
    let writer = d.daemon.writes.as_ref().unwrap();
    let barrier =
        writer.pause_user_dispatch_for_test(id, polylogue_store::writer::DispatchPhase::Owned);
    let daemon = std::sync::Arc::clone(&d.daemon);
    let held = std::thread::spawn(move || {
        crate::test_answers::ok(handle(
            Request::Keep {
                request_id: id.into(),
                text: "sessions".into(),
                name: None,
            },
            &daemon,
        ));
    });
    barrier.wait().unwrap();
    (barrier, held)
}

/// law: brief-answers-before-delivery-commit
#[test]
fn a_brief_answers_while_the_writer_is_held_and_its_delivery_commits_after() {
    let d = daemon_with_a_project();
    let (barrier, held) = hold_writer(&d, "held-writer-turn");
    let id = polylogue_store::user_writes::new_request_id();
    let answer = brief(&d, &id, StartSource::Startup);
    assert_eq!(answer["delivery"], id.as_str());
    assert!(
        answer["prepared"]["text"]
            .as_str()
            .unwrap()
            .contains("plan the otter release"),
        "{answer}"
    );
    let recorded = |d: &test_daemon::TestDaemon| {
        d.daemon
            .reads
            .reader()
            .unwrap()
            .delivery(&id)
            .unwrap()
            .is_some()
    };
    assert!(
        !recorded(&d),
        "the delivery waited for the held writer turn"
    );
    barrier.release();
    held.join().unwrap();
    after_queued_writes(&d);
    assert!(recorded(&d));
}

/// law: delivery-event-once
#[test]
fn a_repeated_delivery_event_is_applied_once() {
    let d = daemon_with_a_project();
    let id = polylogue_store::user_writes::new_request_id();
    brief(&d, &id, StartSource::Startup);
    let event = Request::DeliveryEvent {
        request_id: polylogue_store::user_writes::new_request_id(),
        delivery: id.clone(),
        what: polylogue_model::DeliveryOutcome::Delivered,
        detail: Some("written to the harness hook output".into()),
    };
    for _ in 0..2 {
        let answer = crate::test_answers::ok(handle(event.clone(), &d.daemon));
        assert_eq!(answer["queued"], true);
    }
    after_queued_writes(&d);
    let (row, _) = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .delivery(&id)
        .unwrap()
        .unwrap();
    assert_eq!(row.events, ["delivered"]);
}

/// law: brief-delta-after-queued-delivery
#[test]
fn a_compaction_brief_right_after_a_startup_brief_reports_only_what_changed_since() {
    let d = daemon_with_a_project();
    // The startup brief's record is still queued when the compaction brief
    // is prepared.
    let (barrier, held) = hold_writer(&d, "held-before-compaction");
    let startup = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Startup,
    );
    let text = startup["prepared"]["text"].as_str().unwrap();
    assert!(text.contains("Recent sessions here"), "{text}");
    assert!(text.contains("claude-code:s9"), "{text}");
    let compact = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Compact,
    );
    let text = compact["prepared"]["text"].as_str().unwrap();
    assert!(text.contains("Changes since your last brief at"), "{text}");
    assert!(!text.contains("claude-code:s9"), "{text}");
    assert_eq!(compact["prepared"]["empty"], true, "{text}");
    barrier.release();
    held.join().unwrap();
}

/// law: brief-never-waits-for-the-writer-queue
#[test]
fn a_brief_answers_while_the_writer_queue_is_full_and_its_delivery_commits_once_later() {
    use polylogue_model::user::UserWrite;
    use polylogue_store::writer::Queued;
    use std::time::{Duration, Instant};
    let d = daemon_with_a_project();
    let (barrier, held) = hold_writer(&d, "held-with-a-full-queue");
    let writes = d.daemon.writes.as_ref().unwrap();
    let invocation = d
        .daemon
        .invocation(&polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        });
    let mut filling = Vec::new();
    loop {
        let keep = invocation.request(
            polylogue_store::user_writes::new_request_id(),
            UserWrite::KeepResult {
                name: None,
                query: "sessions".into(),
                rows_json: "{\"unit\":\"sessions\",\"columns\":[],\"rows\":[]}".into(),
                read_at: "{}".into(),
            },
        );
        match writes.try_enqueue_user(keep, None).unwrap() {
            Queued::Committing(committed) => filling.push(committed),
            Queued::Full(..) => break,
        }
        assert!(filling.len() < 10_000, "the writer's queue never filled");
    }
    let id = polylogue_store::user_writes::new_request_id();
    let started = Instant::now();
    let answer = brief(&d, &id, StartSource::Startup);
    assert!(
        started.elapsed() < Duration::from_millis(1_500),
        "a full queue delayed the brief by {:?}",
        started.elapsed()
    );
    assert_eq!(answer["delivery"], id.as_str());
    // Its outcome waits behind it, in order.
    let event = crate::test_answers::ok(handle(
        Request::DeliveryEvent {
            request_id: polylogue_store::user_writes::new_request_id(),
            delivery: id.clone(),
            what: polylogue_model::DeliveryOutcome::Delivered,
            detail: None,
        },
        &d.daemon,
    ));
    assert_eq!(event["queued"], true);
    barrier.release();
    held.join().unwrap();
    for committed in filling {
        committed.recv().unwrap().unwrap();
    }
    let recorded = || {
        d.daemon
            .reads
            .reader()
            .unwrap()
            .delivery(&id)
            .unwrap()
            .map(|(row, _)| row.events)
    };
    let until = Instant::now() + Duration::from_secs(20);
    let events = loop {
        match recorded() {
            Some(events) if !events.is_empty() => break events,
            _ => {
                assert!(
                    Instant::now() < until,
                    "the waiting delivery never committed"
                );
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    };
    assert_eq!(events, ["delivered"]);
    let reader = d.daemon.reads.reader().unwrap();
    let rows = reader
        .query(
            "SELECT count(*) AS n FROM source_delivery WHERE id=?1",
            &[polylogue_store::SqlValue::Text(id)],
        )
        .unwrap();
    assert_eq!(rows[0]["n"], 1);
}

#[test]
fn timeout_outcome_requires_positive_preparation_without_waiting_for_the_writer() {
    let d = daemon_with_a_project();
    let (barrier, held) = hold_writer(&d, "held-timeout-receipt");
    let id = polylogue_store::user_writes::new_request_id();
    let prepared = brief(&d, &id, StartSource::Startup);
    let event = |delivery: &str| {
        handle(
            Request::DeliveryEvent {
                request_id: polylogue_store::user_writes::new_request_id(),
                delivery: delivery.into(),
                what: polylogue_model::DeliveryOutcome::NotSent,
                detail: Some("timeout".into()),
            },
            &d.daemon,
        )
    };
    let unknown = crate::test_answers::ok(event("unconfirmed-prepare"));
    let known = crate::test_answers::ok(event(&id));
    barrier.release();
    held.join().unwrap();
    assert_eq!(prepared["delivery"], id);
    assert_eq!(unknown["queued"], false);
    assert_eq!(known["queued"], true);
    after_queued_writes(&d);
    let reader = d.daemon.reads.reader().unwrap();
    assert!(reader.delivery("unconfirmed-prepare").unwrap().is_none());
    let delivery = reader.delivery(&id).unwrap().unwrap();
    assert!(delivery.0.events.iter().any(|event| event == "not_sent"));
}
