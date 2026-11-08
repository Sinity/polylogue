use super::*;
use polylogue_model::signal::Outbox;

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

fn assertion(author: polylogue_model::Author) -> polylogue_model::assertion::Assertion {
    use polylogue_model::assertion::{Assertion, AssertionFact, AssertionKind};
    Assertion {
        fact: AssertionFact {
            id: "revision".into(),
            root: "family".into(),
            kind: AssertionKind::Signal,
            key: Some("waiting".into()),
            value: Some(json!({"name":"waiting","query":"sessions","interrupt":true})),
            author,
            via: polylogue_model::authorship::Channel::Import,
            machine: "synthetic-host".into(),
            created_at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
            valid_until: None,
            links: vec![],
        },
        commit_seq: 0,
    }
}

#[test]
fn signal_assertion_claims_never_supply_an_active_author_or_revision_epoch() {
    use polylogue_model::{Author, Origin};
    let config = crate::config::SignalDeployment::default().rule_configuration();
    let expected = rules::definition(&assertion(Author::Unattributed), &config).unwrap();
    for author in [
        Author::Operator,
        Author::Script("unit".into()),
        Author::Unattributed,
        Author::Agent(None),
        Author::AgentOrigin(Origin::new("codex").unwrap()),
    ] {
        let mut candidate = assertion(author);
        candidate.commit_seq = 999;
        candidate.fact.id = "another-revision".into();
        assert_eq!(rules::definition(&candidate, &config).unwrap(), expected);
    }
}

#[test]
fn signal_definition_uses_typed_fact_and_deployment_validation() {
    let config = crate::config::SignalDeployment::default().rule_configuration();
    let mut candidate = assertion(polylogue_model::Author::Operator);
    assert!(rules::definition(&candidate, &config).is_ok());
    candidate.fact.key = Some("different-name".into());
    assert!(rules::definition(&candidate, &config).is_err());
    candidate.fact.key = Some("waiting".into());
    candidate.fact.value = Some(
        json!({"name":"waiting","query":"sessions","deliver":"action","command":"unconfigured"}),
    );
    assert!(rules::definition(&candidate, &config).is_err());
    candidate.fact.kind = polylogue_model::assertion::AssertionKind::Note;
    assert!(rules::definition(&candidate, &config).is_err());
}

#[test]
fn incomplete_timer_does_not_write_a_clock_heartbeat() {
    use std::sync::atomic::{AtomicI64, Ordering};
    let at = Arc::new(AtomicI64::new(1_792_332_000_000));
    let sampled = Arc::clone(&at);
    let clock = polylogue_model::WallClock::new(move || {
        jiff::Timestamp::from_millisecond(sampled.load(Ordering::SeqCst)).unwrap()
    });
    let mut fixture = crate::test_daemon::start_with_clock("synthetic", clock);
    fixture.pause_signals();
    tick(&fixture.daemon).unwrap();
    let before = load(&fixture.daemon).unwrap().unwrap().identity().unwrap();
    at.fetch_add(30_000, Ordering::SeqCst);
    tick(&fixture.daemon).unwrap();
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        before
    );
}

pub(super) fn outbox(name: &str) -> Outbox {
    polylogue_store::fixture::signals::outbox("synthetic", name, 0, true)
}

#[test]
fn restart_recovers_confirmed_outbox_and_preserves_unknown_attempt() {
    use shared_signals::notifications::NotificationState;
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(
            jiff::Timestamp::from_millisecond(1_792_332_000_000).unwrap(),
        ),
    );
    fixture.pause_signals();
    let ready = outbox("confirmed-before-worker");
    let unknown = outbox("interrupted-after-unknown");
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
        .1
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

#[test]
fn completion_commit_wakes_rules_while_operation_clock_is_stopped() {
    use super::observations::Phase;
    struct Worker {
        stop: Option<std::sync::mpsc::Sender<()>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }
    impl Drop for Worker {
        fn drop(&mut self) {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
            if let Some(thread) = self.thread.take() {
                thread.join().unwrap();
            }
        }
    }
    struct Observe<'a> {
        events: std::sync::mpsc::Receiver<Phase>,
        daemon: &'a Daemon,
        deadline: std::time::Instant,
        seen: Vec<Phase>,
    }
    impl Observe<'_> {
        fn next(&mut self, stage: &str) -> Phase {
            let result = self.events.recv_timeout(
                self.deadline
                    .saturating_duration_since(std::time::Instant::now()),
            );
            let event = result.unwrap_or_else(|error| {
                let phases = self.seen.iter().map(ToString::to_string).collect::<Vec<_>>().join("\n");
                panic!("completion cue did not reach the signal controller before its 30s refresh; stage={stage}; receive={error}; status={}; phases:\n{phases}", self.daemon.signals.lock().unwrap());
            });
            self.seen.push(event.clone());
            event
        }
        fn until(&mut self, stage: &str, mut matches: impl FnMut(&Phase) -> bool) -> Phase {
            loop {
                let event = self.next(stage);
                if matches(&event) {
                    return event;
                }
            }
        }
    }
    const FIXED: i64 = 1_792_332_000_000;
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(jiff::Timestamp::from_millisecond(FIXED).unwrap()),
    );
    fixture.pause_signals();
    tick(&fixture.daemon).unwrap();
    let baseline = load(&fixture.daemon).unwrap().unwrap().identity().unwrap();
    *fixture.daemon.signals.lock().unwrap() = Value::Null;
    let (events, observed) = std::sync::mpsc::channel();
    *fixture.daemon.signal_trace.lock().unwrap() = Some(events);
    let (stop, stopped) = std::sync::mpsc::channel();
    let daemon = Arc::clone(&fixture.daemon);
    let _worker = Worker {
        stop: Some(stop),
        thread: Some(std::thread::spawn(move || run(daemon, stopped))),
    };
    let mut observe = Observe {
        events: observed,
        daemon: &fixture.daemon,
        deadline: std::time::Instant::now() + std::time::Duration::from_secs(10),
        seen: vec![],
    };
    observe.until("subscription", |phase| matches!(phase, Phase::Subscribed));
    observe.until(
        "startup rule status",
        |phase| matches!(phase, Phase::Status(value) if value["rules"].is_array()),
    );
    // Confirm startup was read-only instead of letting a self-generated cue
    // stand in for the authored commit this test needs to observe.
    let startup = observe.until("startup completion", |phase| {
        matches!(phase, Phase::TickCompleted { .. })
    });
    assert!(
        matches!(startup, Phase::TickCompleted { persisted: false }),
        "startup changed the durable baseline: {:?}",
        observe.seen
    );
    observe.until("startup controller wait", |phase| {
        matches!(phase, Phase::ControllerWaiting)
    });
    assert!(fixture.daemon.signals.lock().unwrap()["rules"].is_array());

    let writes = fixture.daemon.writes.as_ref().unwrap();
    let committed = writes.completion_commits();
    let write = UserWrite::Assert {
        assertions: vec![polylogue_model::assertion::NewAssertion {
            kind: polylogue_model::assertion::AssertionKind::Signal,
            key: Some("completion-wake".into()),
            value: Some(json!({"name":"completion-wake","query":"sessions"})),
            valid_until: None,
            links: vec![],
        }],
    };
    let context = operation(&fixture.daemon).unwrap();
    assert_eq!(context.clock.now.as_millisecond(), FIXED);
    observe.deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let outcome = writes
        .user(context.request("cue:rule".into(), write))
        .unwrap();
    assert!(!outcome.pending);
    assert!(!outcome.repeat);
    let reader = fixture.daemon.reads.reader().unwrap();
    let stored = reader
        .assertion(&polylogue_model::RevisionReference::exact(
            outcome.id.clone(),
            outcome.id.clone(),
        ))
        .unwrap()
        .expect("acknowledged rule row is visible");
    assert_eq!(
        stored.fact.value.as_ref().unwrap()["name"],
        "completion-wake"
    );
    let assertion_id = stored.fact.id;
    drop(reader);
    observations::emit(
        &fixture.daemon,
        Phase::CommittedSource {
            request: "cue:rule".into(),
        },
    );
    committed
        .try_recv()
        .expect("writer ACK followed its actual committed cue");
    observations::emit(&fixture.daemon, Phase::IndependentCommitCue);

    let mut cue = false;
    let mut read = false;
    let mut status = false;
    let mut complete = false;
    let mut source = false;
    let mut independent = false;
    while !(cue && read && status && complete && source && independent) {
        match observe.next("post-commit controller read and held checkpoint") {
            Phase::ControllerWake { refresh: true } => panic!(
                "refresh substituted for the completion cue: {:?}",
                observe.seen
            ),
            Phase::ControllerWake { refresh: false } => cue = true,
            Phase::RulesRead { ids } if ids.iter().any(|id| id == &assertion_id) => read = true,
            Phase::Status(value)
                if value["rules"].as_array().is_some_and(|rules| {
                    rules.iter().any(|rule| rule["name"] == "completion-wake")
                }) =>
            {
                status = true
            }
            Phase::PersistStarted { .. } | Phase::PersistAcknowledged { .. } => {
                panic!(
                    "missing proof changed the durable checkpoint: {:?}",
                    observe.seen
                )
            }
            Phase::TickCompleted { persisted: false } if status => complete = true,
            Phase::TickCompleted { persisted: true } => {
                panic!(
                    "missing proof persisted a rule adoption: {:?}",
                    observe.seen
                )
            }
            Phase::CommittedSource { request } if request == "cue:rule" => source = true,
            Phase::IndependentCommitCue => independent = true,
            Phase::TickStarted { at } => assert_eq!(at, FIXED, "operation clock advanced"),
            _ => {}
        }
    }
    {
        let status = fixture.daemon.signals.lock().unwrap();
        assert!(
            status["rules"]
                .as_array()
                .is_some_and(|rules| rules.iter().any(|rule| {
                    rule["name"] == "completion-wake" && rule["author"] == "unavailable"
                })),
            "status changed after the observed rule: {:?}",
            observe.seen
        );
        assert!(!status["held"].as_array().unwrap().is_empty());
    }
    assert_eq!(
        load(&fixture.daemon).unwrap().unwrap().identity().unwrap(),
        baseline,
        "unproven assertion adoption changed the durable checkpoint"
    );
    assert_eq!(fixture.daemon.config.requests.now().unix_ms(), FIXED);
}

#[test]
fn unavailable_assertion_closure_preserves_pending_state_without_author_exemption() {
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
            id: "authored:retained".into(),
            revision: 7,
        },
        rule: serde_json::from_value(
            json!({"name":"retained","query":"sessions","for":"1h","interrupt":true}),
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
        .1
    );
    let before = checkpoint.identity().unwrap();
    at.fetch_add(2 * shared_signals::HOUR, Ordering::SeqCst);
    // An empty current-looking assertion read is not withdrawal/closure proof.
    tick(&fixture.daemon).unwrap();
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
        .user(context.request(
            "asserted:signal".into(),
            UserWrite::Assert {
                assertions: vec![polylogue_model::assertion::NewAssertion {
                    kind: polylogue_model::assertion::AssertionKind::Signal,
                    key: Some("waiting".into()),
                    value: Some(json!({"name":"waiting","query":"sessions","interrupt":true})),
                    valid_until: None,
                    links: vec![],
                }],
            },
        ))
        .unwrap();
    tick(&fixture.daemon).unwrap();
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
    assert!(
        status["rules"]
            .as_array()
            .unwrap()
            .iter()
            .any(|rule| rule["name"] == "waiting" && rule["author"] == "unavailable")
    );
    assert!(!status["held"].as_array().unwrap().is_empty());
}
