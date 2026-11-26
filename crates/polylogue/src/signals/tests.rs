use super::*;
use polylogue_model::signal::PendingDelivery;

pub(super) fn operation(d: &Daemon) -> Result<Arc<polylogue_reading::operation::Context>, Held> {
    d.operation_context(&polylogue_reading::operation::Context::worker_authorship(
        "polylogue-signals",
    ))
    .map_err(Held::from)
}

#[test]
fn rule_duration_uses_shared_parser_without_overflow() {
    assert_eq!(duration("20m"), Some(1_200_000));
    assert_eq!(duration("-1m"), None);
    assert_eq!(duration("999999999999999999999d"), None);
}

#[test]
fn signal_schema_uses_common_closed_rule_and_delivery_contract() {
    let rule: shared_signals::Rule = serde_json::from_value(
        json!({"name":"test","query":"sessions","fires":{"on_change":{"what":"membership"}}}),
    )
    .unwrap();
    assert!(
        rule.validate(&shared_signals::Configuration {
            value_targets: Default::default(),
            action_commands: Default::default()
        })
        .is_ok()
    );
    assert!(
        serde_json::from_value::<shared_signals::Rule>(
            json!({"name":"forged","query":"sessions","operator":true})
        )
        .is_err()
    );
}

fn annotation(author: polylogue_model::Author) -> polylogue_model::annotation::Annotation {
    use polylogue_model::annotation::{Annotation, AnnotationFact, AnnotationKind};
    Annotation {
        fact: AnnotationFact {
            id: "revision".into(),
            root: "family".into(),
            kind: AnnotationKind::Signal,
            key: Some("waiting".into()),
            value: Some(json!({"name":"waiting","query":"sessions","interrupt":true})),
            author,
            via: polylogue_model::authorship::Channel::Import,
            machine: "synthetic-host".into(),
            created_at: Some(polylogue_model::Timestamp::from_unix_ms(1).unwrap()),
            valid_until: None,
            links: vec![],
        },
        commit_seq: 0,
    }
}

#[test]
fn signal_annotation_claims_never_supply_an_active_author_or_revision_epoch() {
    use polylogue_model::{Author, Origin};
    let config = crate::config::SignalDeployment::default().rule_configuration();
    let expected = rules::Definition {
        family: "family".into(),
        rule: serde_json::from_value(json!({"name":"waiting","query":"sessions","interrupt":true}))
            .unwrap(),
    };
    for author in [
        Author::Operator,
        Author::Script("unit".into()),
        Author::Unattributed,
        Author::Agent(None),
        Author::AgentOrigin(Origin::new("codex").unwrap()),
    ] {
        let mut candidate = annotation(author);
        candidate.commit_seq = 999;
        candidate.fact.id = "another-revision".into();
        assert_eq!(rules::definition(&candidate, &config).unwrap(), expected);
    }
}

#[test]
fn signal_definition_uses_typed_fact_and_deployment_validation() {
    let config = crate::config::SignalDeployment::default().rule_configuration();
    let mut candidate = annotation(polylogue_model::Author::Operator);
    assert_eq!(
        rules::definition(&candidate, &config).unwrap(),
        rules::Definition {
            family: "family".into(),
            rule: serde_json::from_value(
                json!({"name":"waiting","query":"sessions","interrupt":true})
            )
            .unwrap(),
        }
    );
    candidate.fact.key = Some("different-name".into());
    assert!(rules::definition(&candidate, &config).is_err());
    candidate.fact.key = Some("waiting".into());
    candidate.fact.value = Some(
        json!({"name":"waiting","query":"sessions","deliver":"action","command":"unconfigured"}),
    );
    assert!(rules::definition(&candidate, &config).is_err());
    candidate.fact.kind = polylogue_model::annotation::AnnotationKind::Note;
    assert!(rules::definition(&candidate, &config).is_err());
}

#[test]
fn a_tick_with_a_stored_checkpoint_writes_no_clock_heartbeat() {
    use std::sync::atomic::{AtomicI64, Ordering};
    let at = Arc::new(AtomicI64::new(1_792_332_000_000));
    let sampled = Arc::clone(&at);
    let clock = polylogue_model::WallClock::new(move || {
        jiff::Timestamp::from_millisecond(sampled.load(Ordering::SeqCst)).unwrap()
    });
    let mut fixture = crate::test_daemon::start_with_clock("synthetic", clock);
    fixture.pause_signals();
    tick(&fixture.daemon, operation(&fixture.daemon).unwrap()).unwrap();
    let before = load(&fixture.daemon).unwrap().unwrap().identity().unwrap();
    at.fetch_add(30_000, Ordering::SeqCst);
    tick(&fixture.daemon, operation(&fixture.daemon).unwrap()).unwrap();
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        before
    );
}

pub(super) fn pending_delivery(name: &str) -> PendingDelivery {
    polylogue_store::fixture::signals::pending_delivery("synthetic", name, 0, true)
}

#[test]
fn restart_recovers_confirmed_deliveries_and_unknown_attempts() {
    use shared_signals::notifications::NotificationState;
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(
            jiff::Timestamp::from_millisecond(1_792_332_000_000).unwrap(),
        ),
    );
    fixture.pause_signals();
    let ready = pending_delivery("confirmed-before-worker");
    let unknown = pending_delivery("interrupted-after-unknown");
    let ready_id = ready.metadata.delivery.id().to_owned();
    let unknown_id = unknown.metadata.delivery.id().to_owned();
    let mut state = NotificationState::default();
    let began = state.begin(unknown.metadata.notice.clone());
    let context = operation(&fixture.daemon).unwrap();
    let reader = fixture.daemon.reads.reader().unwrap();
    let attention = local_attention(&reader, &context).unwrap();
    drop(reader);
    let mut checkpoint = load(&fixture.daemon)
        .unwrap()
        .unwrap_or_else(|| initial(&fixture.daemon, &context, Some(&attention)).unwrap());
    let prior = load(&fixture.daemon)
        .unwrap()
        .map(|c| c.identity().unwrap());
    checkpoint.notifications = Some(state);
    assert!(
        persist(
            &fixture.daemon,
            &context,
            prior,
            checkpoint,
            vec![ready, unknown],
            began
                .into_iter()
                .map(|receipt| Event::Desktop { receipt })
                .collect()
        )
        .unwrap()
    );
    fixture.restart();
    fixture.wait_for(|_| {
        load(&fixture.daemon)
            .unwrap()
            .and_then(|c| c.notifications)
            .and_then(|state| state.attempt(&ready_id).cloned())
            .is_some_and(|attempt| {
                attempt.desktop
                    == Some(shared_signals::DeliveryOutcome::NotSent(
                        "no_notification_server".into(),
                    ))
            })
    });
    fixture.pause_signals();
    let state = load(&fixture.daemon)
        .unwrap()
        .unwrap()
        .notifications
        .unwrap();
    let preserved = state.attempt(&unknown_id).unwrap();
    assert_eq!(
        preserved.desktop,
        Some(shared_signals::DeliveryOutcome::Unknown)
    );
    assert!(!preserved.retried);
    let mut receipts = Vec::new();
    fixture
        .daemon
        .reads
        .reader()
        .unwrap()
        .visit_signals(|stored| {
            if stored.id == unknown_id {
                receipts = stored.events;
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(receipts.len(), 1);
}

/// The signals worker on its own thread, stopped and joined on drop.
struct Worker {
    stop: Option<polylogue_store::cancellation::Cancellation>,
    thread: Option<std::thread::JoinHandle<std::io::Result<()>>>,
}
impl Drop for Worker {
    fn drop(&mut self) {
        if let Some(stop) = self.stop.take() {
            stop.cancel();
        }
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap().unwrap();
        }
    }
}
/// Reads and new content leave the stored checkpoint unchanged; an authored
/// commit makes its rule visible in status.
#[test]
fn reads_and_content_preserve_the_checkpoint_before_an_authored_rule() {
    let mut fixture = crate::test_daemon::start("synthetic");
    fixture.pause_signals();
    tick(&fixture.daemon, operation(&fixture.daemon).unwrap()).unwrap();
    let baseline = load(&fixture.daemon).unwrap().unwrap().identity().unwrap();
    *fixture.daemon.signals.lock().unwrap() = Value::Null;
    let stop = fixture.daemon.archive.stop().child();
    let stopped = stop.clone();
    let daemon = Arc::clone(&fixture.daemon);
    let worker = Worker {
        stop: Some(stop),
        thread: Some(std::thread::spawn(move || run(daemon, stopped))),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    fixture.wait_for(|status| {
        assert!(
            std::time::Instant::now() < deadline,
            "startup did not reach status: {status}"
        );
        status["signals"]["rules"].is_array()
    });
    let writes = fixture.daemon.writes.as_ref().unwrap();
    let committed = writes.completion_commits();
    let read = crate::test_daemon::Call::new(
        "rows",
        json!({"selection":"messages", "steps":[{"verb":"read"}]}),
    )
    .answer(&fixture.daemon);
    assert!(read.value().is_some(), "{read:?}");
    let records = committed.recv().unwrap();
    assert!(records.query_records && !records.authored, "{records:?}");
    let source = fixture.home.join(".claude/projects/-p/s1.jsonl");
    let mut bytes = std::fs::read(&source).unwrap();
    bytes.extend_from_slice(polylogue_testkit::claude_user("u2", "s1", "later").as_bytes());
    std::fs::write(&source, bytes).unwrap();
    loop {
        if committed.recv().unwrap().content {
            break;
        }
    }
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        baseline
    );
    let write = UserWrite::Assert {
        annotations: vec![polylogue_model::annotation::NewAnnotation {
            kind: polylogue_model::annotation::AnnotationKind::Signal,
            key: Some("authored-wake".into()),
            value: Some(json!({"name":"authored-wake","query":"sessions"})),
            valid_until: None,
            links: vec![],
        }],
    };
    let context = operation(&fixture.daemon).unwrap();
    writes
        .archive()
        .user(context.request("authored-wake".into(), write))
        .unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    fixture.wait_for(|status| {
        assert!(
            std::time::Instant::now() < deadline,
            "rule did not reach status: {status}"
        );
        status["signals"]["rules"]
            .as_array()
            .is_some_and(|rules| rules.iter().any(|rule| rule["name"] == "authored-wake"))
    });
    drop(worker);
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        baseline
    );
}

#[test]
fn completion_commit_wakes_rules_while_operation_clock_is_stopped() {
    const FIXED: i64 = 1_792_332_000_000;
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(jiff::Timestamp::from_millisecond(FIXED).unwrap()),
    );
    fixture.pause_signals();
    tick(&fixture.daemon, operation(&fixture.daemon).unwrap()).unwrap();
    let baseline = load(&fixture.daemon).unwrap().unwrap().identity().unwrap();
    *fixture.daemon.signals.lock().unwrap() = Value::Null;
    let stop = fixture.daemon.archive.stop().child();
    let stopped = stop.clone();
    let daemon = Arc::clone(&fixture.daemon);
    let worker = Worker {
        stop: Some(stop),
        thread: Some(std::thread::spawn(move || run(daemon, stopped))),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    fixture.wait_for(|status| {
        assert!(
            std::time::Instant::now() < deadline,
            "startup did not reach status: {status}"
        );
        status["signals"]["rules"].is_array()
    });
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        baseline
    );
    let writes = fixture.daemon.writes.as_ref().unwrap();
    let committed = writes.completion_commits();
    let write = UserWrite::Assert {
        annotations: vec![polylogue_model::annotation::NewAnnotation {
            kind: polylogue_model::annotation::AnnotationKind::Signal,
            key: Some("completion-wake".into()),
            value: Some(json!({"name":"completion-wake","query":"sessions"})),
            valid_until: None,
            links: vec![],
        }],
    };
    let context = operation(&fixture.daemon).unwrap();
    assert_eq!(context.clock.now.as_millisecond(), FIXED);
    let outcome = writes
        .archive()
        .user(context.request("cue:rule".into(), write))
        .unwrap();
    assert!(!outcome.pending);
    assert!(!outcome.repeat);
    let reader = fixture.daemon.reads.reader().unwrap();
    let stored = reader
        .annotation(&polylogue_model::RevisionReference::exact(
            outcome.id.clone(),
            outcome.id.clone(),
        ))
        .unwrap()
        .expect("acknowledged rule row is visible");
    assert_eq!(
        stored.fact.value.as_ref().unwrap()["name"],
        "completion-wake"
    );
    drop(reader);
    committed
        .try_recv()
        .expect("writer acknowledgment followed its committed cue");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    fixture.wait_for(|status| {
        assert!(
            std::time::Instant::now() < deadline,
            "completion cue did not reach status: {status}"
        );
        status["signals"]["rules"].as_array().is_some_and(|rules| {
            rules.iter().any(|rule| {
                rule["name"] == "completion-wake"
                    && rule["author"]
                        .as_str()
                        .is_some_and(|author| author != "unavailable")
            })
        })
    });
    drop(worker);
    let status = crate::protocol::into_value(
        crate::test_daemon::Call::new("status", json!({})).answer(&fixture.daemon),
    )
    .unwrap();
    assert!(!status["signals"]["held"].as_array().unwrap().is_empty());
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        baseline,
        "an unconfirmed annotation adoption changed the stored checkpoint"
    );
    assert_eq!(fixture.daemon.config.requests.now().unix_ms(), FIXED);
}

#[test]
fn unavailable_dependencies_keep_the_annotation_pending() {
    use shared_signals::{ActiveRule, Evaluation, Occurrence, Origin, Row, RuleVersion};
    use std::sync::atomic::{AtomicI64, Ordering};
    let at = Arc::new(AtomicI64::new(1_792_332_000_000));
    let sampled = Arc::clone(&at);
    let clock = polylogue_model::WallClock::new(move || {
        jiff::Timestamp::from_millisecond(sampled.load(Ordering::SeqCst)).unwrap()
    });
    let mut fixture = crate::test_daemon::start_with_clock("synthetic", clock);
    fixture.pause_signals();
    let context = operation(&fixture.daemon).unwrap();
    let reader = fixture.daemon.reads.reader().unwrap();
    let attention = local_attention(&reader, &context).unwrap();
    drop(reader);
    let prior = load(&fixture.daemon)
        .unwrap()
        .map(|checkpoint| checkpoint.identity().unwrap());
    let mut checkpoint = initial(&fixture.daemon, &context, Some(&attention)).unwrap();
    let calendar = DayCalendar::new(context.clock.day.clone());
    let now = context.clock.now.as_millisecond();
    let active = ActiveRule {
        version: RuleVersion {
            product: "polylogue".into(),
            id: "authored:kept".into(),
            revision: 7,
        },
        rule: serde_json::from_value(
            json!({"name":"kept","query":"sessions","for":"1h","interrupt":true}),
        )
        .unwrap(),
        author: shared_signals::Author::Operator,
        via: "historical-claim".into(),
        live: true,
        shipped: Default::default(),
    };
    checkpoint.evaluator = Evaluator::new(now, vec![active.clone()], &calendar).unwrap();
    checkpoint
        .evaluator
        .observe(Evaluation {
            rule: active.version.clone(),
            at: now,
            complete: true,
            value: None,
            rows: vec![Row {
                occurrence: Occurrence {
                    id: "waiting".into(),
                    period: "one".into(),
                },
                reference: "session:synthetic".into(),
                observed: now,
                start: now,
                end: None,
                origin: Origin::Source,
                evidence: json!({"observed":now}),
            }],
        })
        .unwrap();
    assert!(
        !checkpoint
            .evaluator
            .reports(now, &calendar)
            .unwrap()
            .iter()
            .any(|report| matches!(report, shared_signals::Report::Fired(_)))
    );
    assert_eq!(checkpoint.evaluator.pending_count(), 1);
    checkpoint.rules = vec![active];
    assert!(
        persist(
            &fixture.daemon,
            &context,
            prior,
            checkpoint.clone(),
            vec![],
            vec![]
        )
        .unwrap()
    );
    let before = checkpoint.identity().unwrap();
    at.fetch_add(2 * shared_signals::HOUR, Ordering::SeqCst);
    // An empty current-looking annotation read does not show a withdrawal or closure.
    tick(&fixture.daemon, operation(&fixture.daemon).unwrap()).unwrap();
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        before
    );
    let context = operation(&fixture.daemon).unwrap();
    fixture
        .daemon
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .user(context.request(
            "asserted:signal".into(),
            UserWrite::Assert {
                annotations: vec![polylogue_model::annotation::NewAnnotation {
                    kind: polylogue_model::annotation::AnnotationKind::Signal,
                    key: Some("waiting".into()),
                    value: Some(json!({"name":"waiting","query":"sessions","interrupt":true})),
                    valid_until: None,
                    links: vec![],
                }],
            },
        ))
        .unwrap();
    tick(&fixture.daemon, operation(&fixture.daemon).unwrap()).unwrap();
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        before
    );
    let mut delivered = 0;
    fixture
        .daemon
        .reads
        .reader()
        .unwrap()
        .visit_signals(|_| {
            delivered += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(delivered, 0);
    let status = fixture.daemon.signals.lock().unwrap();
    assert!(status["rules"].as_array().unwrap().iter().any(|rule| {
        rule["name"] == "waiting"
            && rule["author"]
                .as_str()
                .is_some_and(|author| author != "unavailable")
    }));
    assert!(!status["held"].as_array().unwrap().is_empty());
}

/// A rule's last firing is the latest time of any of its stored
/// deliveries, per rule name.
#[test]
fn signal_rules_report_when_they_last_fired() {
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(
            jiff::Timestamp::from_millisecond(1_792_332_000_000).unwrap(),
        ),
    );
    fixture.pause_signals();
    let context = operation(&fixture.daemon).unwrap();
    let reader = fixture.daemon.reads.reader().unwrap();
    let attention = local_attention(&reader, &context).unwrap();
    assert!(reader.last_firings().unwrap().is_empty());
    drop(reader);
    let checkpoint = load(&fixture.daemon)
        .unwrap()
        .unwrap_or_else(|| initial(&fixture.daemon, &context, Some(&attention)).unwrap());
    let prior = load(&fixture.daemon)
        .unwrap()
        .map(|c| c.identity().unwrap());
    let delivery = |rule: &str, occurrence: &str, at: i64| {
        polylogue_store::fixture::signals::pending_delivery(rule, occurrence, at, true)
    };
    assert!(
        persist(
            &fixture.daemon,
            &context,
            prior,
            checkpoint,
            vec![
                delivery("waiting", "first", 5_000),
                delivery("waiting", "second", 9_000),
                delivery("waiting", "third", 7_000),
                delivery("stalled", "only", 2_000),
            ],
            vec![],
        )
        .unwrap()
    );
    let firings = fixture
        .daemon
        .reads
        .reader()
        .unwrap()
        .last_firings()
        .unwrap();
    assert_eq!(
        firings.into_iter().collect::<Vec<_>>(),
        [("stalled".to_owned(), 2_000), ("waiting".to_owned(), 9_000)]
    );
}
