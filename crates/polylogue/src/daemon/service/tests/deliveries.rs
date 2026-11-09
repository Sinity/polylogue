//! A delivery is recorded once per request: a hook that resends its request
//! after a lost answer gets the same delivery back. Its record is queued for
//! the writer, so a context answer never waits for a writer turn.

use super::*;
use crate::briefing::StartSource;

/// Returns once the writer has applied every request queued before this
/// call: it answers a synchronous write only after the queued ones.
fn after_queued_writes(d: &test_daemon::TestDaemon) {
    crate::test_answers::ok(
        crate::test_daemon::Call::new("keep", serde_json::json!({"query": "sessions"}))
            .answer(&d.daemon),
    );
}

fn takeover(d: &test_daemon::TestDaemon, request_id: &str) -> serde_json::Value {
    let request =
        crate::test_daemon::Call::new("takeover", serde_json::json!({"session": "claude-code:s1"}))
            .id(request_id)
            .surface(crate::commands::Surface::Hook);
    crate::test_answers::ok((request).answer(&d.daemon))
}

fn deliveries(d: &test_daemon::TestDaemon) -> u64 {
    let request =
        crate::test_daemon::Call::new("query", serde_json::json!({"query": "deliveries"}));
    crate::test_answers::ok((request).answer(&d.daemon))["count"]
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
    let request =
        crate::test_daemon::Call::new("takeover", serde_json::json!({"session": "claude-code:s1"}))
            .id(&id)
            .surface(crate::commands::Surface::Mcp);
    let (response, handed_off) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| request.limit(limit.clone()).answer(&d.daemon));
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
    let request = crate::test_daemon::Call::new(
        "brief",
        serde_json::json!({
            "cwd": "/home/op/proj",
            "session": "claude-code:newcomer",
            "source": source,
        }),
    )
    .id(request_id)
    .surface(crate::commands::Surface::Hook);
    match (request).answer(&d.daemon) {
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
        crate::test_answers::ok(
            crate::test_daemon::Call::new("keep", serde_json::json!({"query": "sessions"}))
                .id(id)
                .answer(&daemon),
        );
    });
    barrier.wait().unwrap();
    (barrier, held)
}

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

#[test]
fn a_repeated_delivery_event_is_applied_once() {
    let d = daemon_with_a_project();
    let id = polylogue_store::user_writes::new_request_id();
    brief(&d, &id, StartSource::Startup);
    let event = crate::test_daemon::Call::new(
        "delivery-event",
        serde_json::json!({"delivery": id.clone(), "what": polylogue_model::DeliveryOutcome::Delivered, "detail": Some("written to the harness hook output")}),
    );
    for _ in 0..2 {
        let answer = crate::test_answers::ok((event.clone()).answer(&d.daemon));
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
    // The hook hands over its `delivered` outcome right after writing it.
    delivered(&d, startup["delivery"].as_str().unwrap());
    let compact = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Compact,
    );
    let text = compact["prepared"]["text"].as_str().unwrap();
    assert!(text.contains("Changes since your last brief at"), "{text}");
    assert!(!text.contains("claude-code:s9"), "{text}");
    assert!(text.contains("Nothing changed."), "{text}");
    assert!(text.contains(&crate::manual::text()), "{text}");
    assert_eq!(compact["prepared"]["refs"], serde_json::json!([]));
    assert_eq!(compact["prepared"]["left_out"], serde_json::json!([]));
    // The protected compaction manual is still handed over without a delta.
    assert_eq!(compact["prepared"]["empty"], false, "{text}");
    barrier.release();
    held.join().unwrap();
}

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
    let event = crate::test_answers::ok(crate::test_daemon::Call::new("delivery-event", serde_json::json!({"delivery": id.clone(), "what": polylogue_model::DeliveryOutcome::Delivered})).answer(&d.daemon));
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
        crate::test_daemon::Call::new("delivery-event", serde_json::json!({"delivery": delivery, "what": polylogue_model::DeliveryOutcome::NotSent, "detail": Some("timeout")})).answer(&d.daemon)
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

/// A hook's report that it wrote the delivery to the harness.
fn delivered(d: &test_daemon::TestDaemon, delivery: &str) {
    crate::test_answers::ok(
        crate::test_daemon::Call::new(
            "delivery-event",
            serde_json::json!({"delivery": delivery, "what": polylogue_model::DeliveryOutcome::Delivered, "detail": "written to the harness hook output"}),
        )
        .answer(&d.daemon),
    );
}

/// An operator lesson about the brief's directory, written now.
fn lesson(d: &test_daemon::TestDaemon, text: &str) {
    use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
    let lesson = NewAssertion {
        kind: AssertionKind::Lesson,
        key: None,
        value: Some(serde_json::json!(text)),
        valid_until: None,
        links: vec![NewAssertionLink {
            role: LinkRole::About,
            reference: polylogue_model::Reference::Cwd("/home/op/proj".into()),
        }],
    };
    crate::test_answers::ok(
        crate::test_daemon::Call::new("assert", serde_json::json!({"assertions": [lesson]}))
            .authorship(polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            })
            .answer(&d.daemon),
    );
}

/// A delivery of `event` to the brief's session, as another sender records it.
fn other_delivery(d: &test_daemon::TestDaemon, event: polylogue_model::DeliveryEvent) -> String {
    use polylogue_model::user::{Delivery, UserWrite};
    let id = polylogue_store::user_writes::new_request_id();
    let invocation = d
        .daemon
        .invocation(&crate::operation::Context::worker_authorship(
            "deliveries-test",
        ));
    crate::daemon::service::acts::author(
        &d.daemon,
        &invocation,
        id.clone(),
        UserWrite::Deliver(Delivery {
            recipient_kind: "session".into(),
            recipient_id: Some("claude-code:newcomer".into()),
            sender: polylogue_model::DeliverySender::Mcp.as_str().into(),
            event: event.as_str().into(),
            cwd: None,
            text: "handed over".into(),
            refs: vec![],
            left_out: vec![],
        }),
    )
    .unwrap();
    id
}

#[test]
fn takeover_delivery_does_not_move_the_changes_since_point() {
    let d = daemon_with_a_project();
    let startup = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Startup,
    );
    delivered(&d, startup["delivery"].as_str().unwrap());
    after_queued_writes(&d);
    lesson(&d, "the otter release needs a signed tag");
    // A pasted takeover reaches the session later and is delivered.
    let takeover = other_delivery(&d, polylogue_model::DeliveryEvent::Takeover);
    delivered(&d, &takeover);
    after_queued_writes(&d);
    let resume = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Resume,
    );
    let text = resume["prepared"]["text"].as_str().unwrap();
    assert!(text.contains("Changes since your last brief at"), "{text}");
    assert!(
        text.contains("the otter release needs a signed tag"),
        "{text}"
    );
}

#[test]
fn undelivered_brief_does_not_move_the_changes_since_point() {
    let d = daemon_with_a_project();
    let startup = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Startup,
    );
    delivered(&d, startup["delivery"].as_str().unwrap());
    after_queued_writes(&d);
    lesson(&d, "the otter release needs a signed tag");
    // A second brief that never reached the harness.
    brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Startup,
    );
    after_queued_writes(&d);
    let resume = brief(
        &d,
        &polylogue_store::user_writes::new_request_id(),
        StartSource::Resume,
    );
    let text = resume["prepared"]["text"].as_str().unwrap();
    assert!(text.contains("Changes since your last brief at"), "{text}");
    assert!(
        text.contains("the otter release needs a signed tag"),
        "{text}"
    );
}
