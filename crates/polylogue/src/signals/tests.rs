use super::*;

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

fn registry() -> Checkpoint {
    let day = query_language::time::DaySetting::new(
        jiff::tz::TimeZone::UTC,
        jiff::civil::Time::new(0, 0, 0, 0).unwrap(),
    )
    .unwrap();
    Checkpoint {
        owner: "fixture".into(),
        evaluator: Evaluator::new(0, vec![], Policy::default(), &DayCalendar::new(day)).unwrap(),
        rules: vec![],
        rule_sources: BTreeMap::new(),
        rule_revisions: BTreeMap::new(),
        host: None,
        notifications: None,
        outlet_attempts: BTreeMap::new(),
    }
}
fn note(author: polylogue_model::Author) -> polylogue_model::Note {
    polylogue_model::Note {
        id: "revision".into(),
        first_note: "family".into(),
        kind: polylogue_model::NoteKind::Signal,
        key: None,
        value: Some(json!({"name":"waiting","query":"sessions","interrupt":true}).to_string()),
        author,
        created_at: polylogue_model::Timestamp::from_unix_ms(0).unwrap(),
        valid_until: None,
        targets: vec![],
        relations: vec![],
    }
}
#[test]
fn only_exact_operator_authorship_has_shared_budget_exemption() {
    use polylogue_model::{Author, Origin};
    let config = crate::config::SignalDeployment::default().rule_configuration();
    for author in [
        Author::Operator,
        Author::Script("unit".into()),
        Author::Unattributed,
        Author::Agent(None),
        Author::AgentOrigin(Origin::new("codex").unwrap()),
    ] {
        let expected = author == Author::Operator;
        let active = rules::active(&note(author), &registry(), &config).unwrap();
        assert_eq!(active.author.budget_exempt(), expected);
    }
}
#[test]
fn rule_withdrawal_and_restore_cannot_reuse_an_adopted_revision() {
    let config = crate::config::SignalDeployment::default().rule_configuration();
    let mut state = registry();
    let note = note(polylogue_model::Author::Operator);
    let first = rules::active(&note, &state, &config).unwrap();
    state
        .rule_sources
        .insert(first.version.id.clone(), note.id.clone());
    state
        .rule_revisions
        .insert(first.version.id.clone(), first.version.revision);
    state.rules.push(first.clone());
    let restored: Checkpoint =
        serde_json::from_slice(&serde_json::to_vec(&state).unwrap()).unwrap();
    assert_eq!(
        rules::active(&note, &restored, &config).unwrap().version,
        first.version
    );
    state.rules.clear();
    let again = rules::active(&note, &state, &config).unwrap();
    assert_eq!(again.version.id, first.version.id);
    assert_eq!(again.version.revision, first.version.revision + 1);
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

fn outbox(name: &str) -> Outbox {
    use shared_signals::{Channel, Occurrence, RuleVersion, Trigger};
    let trigger = Trigger {
        rule: RuleVersion {
            product: "polylogue".into(),
            id: "synthetic".into(),
            revision: 1,
        },
        occurrence: Occurrence {
            id: name.into(),
            episode: "one".into(),
        },
    };
    let id = trigger.delivery_id();
    let delivery = shared_signals::Delivery {
        id: id.clone(),
        trigger,
        matched: 0,
        at: 0,
        basis: vec![],
        channel: Channel::Interrupt,
        downgrade: None,
        change: None,
        value: None,
        scope: None,
    };
    let notice = shared_signals::notifications::Notice {
        delivery_id: id,
        product: "polylogue".into(),
        icon: String::new(),
        rule: name.into(),
        occurrence: name.into(),
        summary: name.into(),
        body: String::new(),
        reference: "repo:synthetic".into(),
        command_hint: String::new(),
        interrupt: true,
        current: false,
        data_loss: false,
        budget_passive_count: None,
    };
    Outbox {
        record: Delivery {
            recipient_kind: "operator".into(),
            recipient_id: None,
            sender: "signal".into(),
            event: "signal".into(),
            cwd: None,
            text: name.into(),
            refs: vec![],
            left_out: vec![],
        },
        metadata: Metadata {
            delivery,
            notice,
            forward: None,
            outlet: None,
            execution: ExecutionEligibility::Live,
        },
    }
}

#[test]
fn restart_recovers_confirmed_outbox_and_preserves_unknown_attempt() {
    use shared_signals::notifications::{NotificationState, Route};
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(
            jiff::Timestamp::from_millisecond(1_792_332_000_000).unwrap(),
        ),
    );
    fixture.pause_signals();
    let ready = outbox("confirmed-before-worker");
    let unknown = outbox("interrupted-after-unknown");
    let ready_id = ready.metadata.delivery.id.clone();
    let unknown_id = unknown.metadata.delivery.id.clone();
    let mut state = NotificationState::default();
    let began = state.begin(unknown.metadata.notice.clone(), Route::Dbus);
    let context = operation(&fixture.daemon).unwrap();
    let reader = fixture.daemon.reads.reader().unwrap();
    let attention = local_attention(&reader, &context).unwrap();
    let mut checkpoint = load(&fixture.daemon)
        .unwrap()
        .unwrap_or_else(|| initial(&context, Some(&attention)).unwrap());
    let prior = load(&fixture.daemon)
        .unwrap()
        .map(|c| c.identity().unwrap());
    checkpoint.notifications = Some(state);
    drop(reader);
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
                attempt.dbus
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
        preserved.dbus,
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
    fn observe(mut condition: impl FnMut() -> bool) {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !condition() {
            assert!(
                std::time::Instant::now() < end,
                "completion cue did not reach the signal controller before its 30s refresh"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    }
    let mut fixture = crate::test_daemon::start_with_clock(
        "synthetic",
        polylogue_model::WallClock::fixed(
            jiff::Timestamp::from_millisecond(1_792_332_000_000).unwrap(),
        ),
    );
    fixture.pause_signals();
    tick(&fixture.daemon).unwrap();
    *fixture.daemon.signals.lock().unwrap() = Value::Null;
    let (stop, stopped) = std::sync::mpsc::channel();
    let daemon = Arc::clone(&fixture.daemon);
    let _worker = Worker {
        stop: Some(stop),
        thread: Some(std::thread::spawn(move || run(daemon, &stopped))),
    };
    // The saved baseline makes this startup evaluation read-only. There is no
    // self-generated checkpoint cue left to mask a missing source subscription.
    observe(|| fixture.daemon.signals.lock().unwrap()["rules"].is_array());
    let write = UserWrite::AddNote {
        note: polylogue_model::NewNote {
            kind: polylogue_model::NoteKind::Signal,
            key: None,
            value: Some(json!({"name":"completion-wake","query":"sessions"}).to_string()),
            targets: vec![],
            relations: vec![],
            revises: None,
            valid_until: None,
        },
    };
    let context = operation(&fixture.daemon).unwrap();
    fixture
        .daemon
        .writes
        .as_ref()
        .unwrap()
        .user(context.request("cue:rule".into(), write))
        .unwrap();
    observe(|| {
        fixture.daemon.signals.lock().unwrap()["rules"]
            .as_array()
            .is_some_and(|rules| rules.iter().any(|rule| rule["name"] == "completion-wake"))
    });
}
