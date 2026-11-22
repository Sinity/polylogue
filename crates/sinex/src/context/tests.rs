use super::*;

fn available() -> PublicContext {
    PublicContext::Available {
        settings_identity: "settings-1".into(),
        settings_revision: 7,
        day: query_language::time::DaySettingWire {
            zone: "Europe/Warsaw".into(),
            start: "04:00".parse().unwrap(),
        },
        quiet_hours: Some(shared_signals::QuietHours {
            from: 1320,
            to: 420,
        }),
        interrupt_budget: sinex_protocol::PublicInterruptBudget {
            per_hour: 3,
            used: 1,
            remaining: 2,
            window_ms: 3600000,
        },
        observed_at: sinex_model::Timestamp::from_unix_secs(100).unwrap(),
    }
}

#[test]
fn unavailable_sinex_never_makes_a_known_context_current() {
    let mut cache = Cache::default();
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Missing
    );
    cache.record(Ok(available()));
    let first = cache
        .resolve(|name| jiff::tz::TimeZone::get(name).map_err(|e| e.to_string()))
        .unwrap();
    assert_eq!(first.settings_revision, 7);
    assert_eq!(first.interrupt_budget.remaining, 2);
    cache.record(Err("Sinex offline".into()));
    let failure = cache.resolve(|_| unreachable!()).unwrap_err();
    assert_eq!(failure.kind, ContextFailureKind::Unavailable);
    assert_eq!(failure.last_known_identity.as_deref(), Some("settings-1"));
    cache.record(Ok(PublicContext::Pending {
        reason: "new configuration has no checkpoint".into(),
    }));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Pending
    );
    assert_eq!(first.settings_identity, "settings-1");
}

#[test]
fn current_context_requires_its_resolved_zone_and_observation_age() {
    let mut cache = Cache::default();
    cache.record(Ok(available()));
    let wrong = cache
        .resolve(|_| Ok(jiff::tz::TimeZone::get("Etc/UTC").unwrap()))
        .unwrap_err();
    assert_eq!(wrong.kind, ContextFailureKind::Invalid);
    cache.received = Some(Instant::now() - Duration::from_secs(10));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Stale
    );
    cache.record(Ok(PublicContext::Refused {
        reason: "public context grant denied".into(),
    }));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Refused
    );
}

#[test]
fn malformed_policy_never_reaches_query_or_signal_callers() {
    let mut reply = available();
    if let PublicContext::Available {
        interrupt_budget, ..
    } = &mut reply
    {
        interrupt_budget.remaining = 3;
    }
    let mut cache = Cache::default();
    cache.record(Ok(reply));
    assert_eq!(
        cache.resolve(|_| unreachable!()).unwrap_err().kind,
        ContextFailureKind::Invalid
    );
}

#[test]
fn context_refreshes_an_expired_observation_on_the_same_connection_and_keeps_refusals() {
    use futures_util::StreamExt;
    use std::sync::{Arc, Mutex, mpsc};

    let subjects = sinex_protocol::Subjects::new("context");
    let server = devkit::nats::Server::start_with(&devkit::nats::Options {
        config: subjects.authorization(
            ("daemon", "daemon-password"),
            &[("polylogue", "producer-password")],
            &[("polylogue", "fixture-scope".into())],
        ),
    })
    .unwrap();
    let url = server.url().to_owned();
    let slow_context = polylogue_store::cancellation::Cancellation::default();
    let cancel_context = slow_context.clone();
    let (ready, started) = mpsc::channel();
    let fake = std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            tokio::time::timeout(Duration::from_secs(10), async move {
                let nats = async_nats::ConnectOptions::with_user_and_password(
                    "daemon".into(),
                    "daemon-password".into(),
                )
                .connect(url)
                .await
                .unwrap();
                let mut hello = nats.subscribe("context.connect.polylogue").await.unwrap();
                let mut contexts = nats
                    .subscribe("context.public_context.polylogue")
                    .await
                    .unwrap();
                nats.flush().await.unwrap();
                ready.send(()).unwrap();
                let hello = hello.next().await.unwrap();
                sinex_transport::replies::publish_json(
                    &nats,
                    hello.reply.unwrap().to_string(),
                    sinex_protocol::ProducerWelcome::Welcome {
                        family: Some("polylogue".into()),
                        canonical_publisher: true,
                    },
                    Duration::from_secs(2),
                )
                .await
                .unwrap();
                let mut refreshed = available();
                if let PublicContext::Available { observed_at, .. } = &mut refreshed {
                    *observed_at = sinex_model::Timestamp::from_unix_secs(101).unwrap();
                }
                for reply in [
                    Some(available()),
                    Some(refreshed),
                    None,
                    Some(PublicContext::Refused {
                        reason: "public context grant denied".into(),
                    }),
                ] {
                    let request = contexts.next().await.unwrap();
                    let _: sinex_protocol::PublicContextRequest =
                        serde_json::from_slice(&request.payload).unwrap();
                    match reply {
                        Some(reply) => sinex_transport::replies::publish_json(
                            &nats,
                            request.reply.unwrap().to_string(),
                            reply,
                            Duration::from_secs(2),
                        )
                        .await
                        .unwrap(),
                        None => cancel_context.cancel(),
                    }
                }
            })
            .await
            .unwrap();
        });
    });
    started.recv_timeout(Duration::from_secs(10)).unwrap();
    let root = tempfile::tempdir().unwrap();
    let config = crate::Config {
        mode: crate::Mode::Primary,
        nats: server.url().into(),
        producer: "polylogue".into(),
        subject_root: "context".into(),
        store: root.path().into(),
        password_file: None,
        upload_root: None,
    };
    let sinex = Sinex {
        config,
        authentication: crate::Authentication::Password("producer-password".into()),
        health: Arc::new(Mutex::new(crate::status::Health::default())),
        spool: Arc::new(Mutex::new(
            crate::spool::Spool::open(root.path(), "polylogue").unwrap(),
        )),
        runtime: Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .enable_all()
                .build()
                .unwrap(),
        ),
        publication: Default::default(),
        sender: None,
        stopping: None,
        client: tokio::sync::Mutex::new(None),
        host: "fixture-host".into(),
    };
    let resolve = |name: &str| jiff::tz::TimeZone::get(name).map_err(|error| error.to_string());
    let background = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    assert_eq!(
        background
            .block_on(sinex.context_async(resolve))
            .unwrap_err()
            .kind,
        ContextFailureKind::Missing
    );
    sinex.read_client(async |_| Ok(())).unwrap();
    let first = sinex.context(resolve).unwrap();
    let expire = || {
        let mut health = sinex.health.locked();
        let old = Instant::now() - Duration::from_secs(10);
        health.context.received = Some(old);
        health.context.attempted = Some(old);
    };
    expire();
    let next = background.block_on(sinex.context_async(resolve)).unwrap();
    assert_eq!(first.observed_at.unix_ns(), 100_000_000_000);
    assert_eq!(next.observed_at.unix_ns(), 101_000_000_000);
    assert_eq!(next.settings_identity, first.settings_identity);
    assert_eq!(next.settings_revision, first.settings_revision);
    assert!(
        next.received_seconds_ago <= polylogue_model::freshness::SINEX_OBSERVATION_MAX_AGE_SECONDS
    );
    // A fresh call reuses the observation. Expired calls still obey the
    // caller's cancellation and current authorization.
    assert_eq!(
        sinex.context(resolve).unwrap().observed_at,
        next.observed_at
    );
    let limit = polylogue_store::read::limit::ReadLimit::new(
        Duration::from_millis(10),
        polylogue_store::cancellation::Cancellation::default(),
    );
    let timed_out = limit
        .scope(|| sinex.read_client(async |_| std::future::pending::<crate::Result<()>>().await));
    assert!(matches!(
        timed_out,
        Err(crate::Error::Store(
            polylogue_store::StoreError::ReadDeadlineExceeded
        ))
    ));
    let cancellation = polylogue_store::cancellation::Cancellation::default();
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(cancellation.clone());
    let cancelled = limit.scope(|| {
        sinex.read_client(async |_| {
            cancellation.cancel();
            std::future::pending::<crate::Result<()>>().await
        })
    });
    assert!(matches!(
        cancelled,
        Err(crate::Error::Store(
            polylogue_store::StoreError::ReadCancelled
        ))
    ));
    let held = sinex.client.try_lock().unwrap();
    let waiting = polylogue_store::read::limit::ReadLimit::new(
        Duration::from_millis(10),
        polylogue_store::cancellation::Cancellation::default(),
    );
    assert!(matches!(
        waiting.scope(|| sinex.read_client(async |_| Ok(()))),
        Err(crate::Error::Store(
            polylogue_store::StoreError::ReadDeadlineExceeded
        ))
    ));
    drop(held);
    expire();
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(slow_context);
    assert_eq!(
        limit.scope(|| sinex.context(resolve)).unwrap_err().kind,
        ContextFailureKind::Unavailable
    );
    assert!(matches!(
        limit.check(),
        Err(polylogue_store::StoreError::ReadCancelled)
    ));
    expire();
    let error = background
        .block_on(sinex.context_async(resolve))
        .unwrap_err();
    assert_eq!(error.kind, ContextFailureKind::Refused);
    assert_eq!(error.reason, "public context grant denied");
    assert_eq!(error.last_known_identity.as_deref(), Some("settings-1"));
    fake.join().unwrap();
}
