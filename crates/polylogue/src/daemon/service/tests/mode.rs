mod confirmed_inputs;
mod withdrawals;

use super::*;
use polylogue_testkit::{ClaudeHome, claude_user as user};

fn path(home: &ClaudeHome) -> std::path::PathBuf {
    home.root.join("sinex.toml")
}

fn startup(home: &ClaudeHome) -> crate::config::Startup {
    use crate::config::{EnvSnapshot, Identity, Overrides, ProcessInputs, Startup};
    Startup::load(
        &ProcessInputs {
            environment: EnvSnapshot::from_values([
                ("HOME".into(), home.home.clone().into_os_string()),
                (
                    "XDG_CONFIG_HOME".into(),
                    home.dir.path().join("config").into_os_string(),
                ),
            ]),
            output: crate::config::OutputConfig::new(false, false, true, None),
            clock: polylogue_model::WallClock::fixed("2026-10-07T10:00:00Z".parse().unwrap()),
            identity: Some(Identity {
                machine: "synthetic-machine".into(),
                host: "synthetic-host".into(),
            }),
            memory_bytes: Some(4 << 30),
            workers: Some(1),
            cwd: Some(home.dir.path().into()),
        },
        Overrides {
            archive: Some(home.root.clone()),
            listen: Some("127.0.0.1:0".parse().unwrap()),
            ..Default::default()
        },
    )
    .unwrap()
}

fn loaded(home: &ClaudeHome, text: &str) -> crate::config::Startup {
    let path = path(home);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
    startup(home)
}

fn primary(home: &ClaudeHome) -> crate::config::Startup {
    loaded(
        home,
        &format!(
            "mode = 'primary'\nnats = 'nats://127.0.0.1:1'\nstore = {:?}\n",
            home.dir.path().join("sinex-store"),
        ),
    )
}

fn launch(startup: crate::config::Startup) -> std::io::Result<Runtime> {
    let attempt = startup
        .mode
        .needs_engine()
        .then(polylogue_testkit::stopped_attempt);
    start(
        startup.config,
        startup.engine,
        startup.mode,
        polylogue_engine::Lifetime::UntilStopped,
        attempt,
    )
}

fn read_request() -> crate::test_daemon::Call {
    crate::test_daemon::Call::new(
        "rows",
        serde_json::json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "format":"json"}]}),
    )
}

fn seeded() -> ClaudeHome {
    let w = ClaudeHome::new();
    w.append("s1.jsonl", &user("u1", "s1", "Stored question"));
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    drop(engine);
    drop(_lock);
    w
}

fn blocked(runtime: &Runtime, path: &std::path::Path) {
    assert!(runtime.daemon.writes.is_none());
    assert!(runtime.engine.is_none());
    assert!(runtime.signals.is_none());
    assert!(runtime.signals_stop.is_none());
    // Sinex is unavailable while the configuration is broken, so status is degraded.
    let (status, _) = crate::test_answers::degraded(
        crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon),
    );
    assert_eq!(status["mode"]["state"], "blocked");
    assert_eq!(status["mode"]["path"], path.to_str().unwrap());
    assert!(!status["mode"]["error"].as_str().unwrap().is_empty());
    let requests = vec![
        crate::test_daemon::Call::new(
            "chat send",
            serde_json::json!({"provider": "chatgpt", "text": "Synthetic draft", "model": "X"}),
        )
        .id("send"),
        crate::test_daemon::Call::new("backfill start", serde_json::json!({"provider": "chatgpt"})),
        crate::test_daemon::Call::new("import", serde_json::json!({})).id("import"),
        crate::test_daemon::Call::new(
            "set",
            serde_json::json!({"key": Some("browser.actions"), "value": Some("on")}),
        ),
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"question", "steps":[{"verb":"brief"}]}),
        )
        .id("delivery")
        .surface(crate::commands::Surface::Hook),
        crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection":"sessions", "steps":[{"verb":"save"}]}),
        )
        .id("keep"),
        crate::test_daemon::Call::new("embed", serde_json::json!({"yes": true})),
    ];
    for request in requests {
        assert!(
            matches!((request).answer(&runtime.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. }
            if error.code.as_deref() == Some("sinex_unavailable")
                && error.hint.as_deref().is_some_and(|hint| hint.contains(path.to_str().unwrap())))
        );
    }
}

#[test]
fn malformed_config_keeps_primary_status_and_refuses_reads_and_writes() {
    for text in [
        "mode = [",
        "mode = 'unknown'\nnats = 'invalid://'\nstore = '/unused'\n",
    ] {
        let w = ClaudeHome::new();
        let store = w.dir.path().join("sinex-store");
        drop(Archive::open_primary(&w.root, &store).unwrap());
        w.append(
            "pending.jsonl",
            &user("later", "pending", "Must wait for configuration"),
        );
        let startup = loaded(&w, text);
        assert!(startup.config.standalone_day().is_err());
        let runtime = launch(startup).unwrap();
        blocked(&runtime, &path(&w));
        assert!(
            matches!((read_request()).answer(&runtime.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("sinex_unavailable"))
        );
        assert!(
            crate::test_answers::ok(
                crate::test_daemon::Call::new(
                    "set",
                    serde_json::json!({"key": Some("browser.actions")})
                )
                .answer(&runtime.daemon)
            )["authored"]
                .is_null()
        );
        assert!(w.file("pending.jsonl").exists());
        assert!(!w.root.join("archive.db").exists());
    }
}

#[test]
fn malformed_primary_configuration_does_not_reclassify_a_standalone_archive() {
    let w = seeded();
    let runtime = launch(loaded(&w, "mode = [")).unwrap();
    blocked(&runtime, &path(&w));
    assert!(!runtime.daemon.archive.is_primary());
    drop(runtime);
    assert_eq!(w.count("SELECT count(*) AS n FROM message"), 1);
    assert!(
        Archive::open_or_create(&w.root).is_ok(),
        "failed startup kept its lock"
    );
}

#[test]
fn missing_config_is_standalone() {
    let w = seeded();
    let startup = startup(&w);
    assert!(matches!(startup.mode, ArchiveMode::Standalone));
    let runtime = launch(startup).unwrap();
    assert!(runtime.daemon.writes.is_some());
    assert_eq!(
        crate::test_answers::ok(
            crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon)
        )["mode"]["state"],
        "standalone"
    );
}

#[test]
fn older_import_preserves_scope_refusal_and_succeeds_after_initial_confirmation() {
    use polylogue_store::fault::{self, Action};
    use polylogue_store::primary::{Confirmed, ConfirmedChange, ReplayControl};

    let home = ClaudeHome::new();
    let older = home.dir.path().join("older");
    let _fixture = polylogue_store::older_archive::fixture::standard(&older);
    let original_files = polylogue_store::older_archive::fixture::file_hashes(&older);
    let runtime = launch(primary(&home)).unwrap();
    let writer = runtime.daemon.writes.as_ref().unwrap();
    let control = |control| Confirmed {
        position: 0,
        event: String::new(),
        output_index: 0,
        sinex_context: None,
        change: ConfirmedChange::Control(Box::new(control)),
    };
    fault::clear();
    fault::arm("primary.scope_reset_commit", "", Action::Fail, 0, 1);
    let failed_reset = writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "initial-import-scope".into(),
        })])
        .unwrap_err();
    fault::clear();
    assert!(
        failed_reset
            .to_string()
            .contains("primary.scope_reset_commit"),
        "the reset must fail at its commit boundary: {failed_reset}"
    );
    let fenced = runtime
        .daemon
        .reads
        .reader()
        .err()
        .expect("failed reset fences reads");
    assert!(matches!(&fenced, StoreError::ScopeChanged), "{fenced}");
    let import = || {
        crate::test_daemon::Call::new("import", serde_json::json!({"from_archive": older}))
            .answer(&runtime.daemon)
    };
    let refused = import();
    assert!(
        matches!(&refused.outcome, command_surface::Outcome::Error(error)
            if error.code.as_deref() == Some("scope_changed") && error.retryable),
        "an initial scope refusal must remain a typed transient failure: {refused:?}"
    );
    assert!(matches!(
        runtime.daemon.reads.reader(),
        Err(StoreError::ScopeChanged)
    ));
    writer
        .apply_confirmed(vec![control(ReplayControl::Reset {
            scope: "initial-import-scope".into(),
        })])
        .unwrap();
    writer
        .apply_confirmed(vec![control(ReplayControl::Completion {
            position: 0,
            delivery: None,
        })])
        .unwrap();
    assert_eq!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .confirmed_checkpoint()
            .unwrap(),
        Some(0)
    );
    let accepted = import();
    let report = accepted
        .value()
        .unwrap_or_else(|| panic!("import accepted after scope completion: {accepted:?}"));
    assert_eq!(report["older_archive"]["pending"], true);
    assert!(
        report["older_archive"]["classes"]["input"]["pending"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(
        polylogue_store::older_archive::fixture::file_hashes(&older),
        original_files
    );
}

#[test]
fn sinex_outage_keeps_raw_bytes_without_standalone_day() {
    for existing in [false, true] {
        let w = ClaudeHome::new();
        let store = w.dir.path().join("sinex-store");
        if existing {
            drop(Archive::open_primary(&w.root, &store).unwrap());
        }
        w.append(
            "later.jsonl",
            &user("later", "later", "Must wait for Sinex"),
        );
        let startup = primary(&w);
        assert!(startup.config.standalone_day().is_err());
        let runtime = launch(startup).unwrap();
        assert!(runtime.daemon.archive.is_primary());
        assert!(runtime.daemon.writes.is_some());
        // Sinex is not running, so status is degraded.
        let (status, _) = crate::test_answers::degraded(
            crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&runtime.daemon),
        );
        assert_eq!(status["mode"]["state"], "primary");
        assert!(
            matches!((read_request()).answer(&runtime.daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. }
            if matches!(error.code.as_deref(), Some("primary_context_missing" | "primary_context_unavailable")))
        );
        let ModeState::Primary { connection } = &runtime.daemon.mode else {
            panic!("primary mode required")
        };
        while connection.status().unwrap().prepared_materials == 0 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        assert!(connection.status().unwrap().prepared_bytes > 0);
        assert!(w.file("later.jsonl").exists());
        assert!(!w.root.join("archive.db").exists());
        let reader = runtime.daemon.reads.reader().unwrap();
        assert_eq!(reader.confirmed_checkpoint().unwrap(), None);
        assert_eq!(
            reader
                .query("SELECT count(*) AS n FROM message", &[])
                .unwrap()[0]["n"],
            0
        );
    }
}

fn primary_table_call(
    runtime: &Runtime,
    operation: &str,
    arguments: serde_json::Value,
    id: &str,
) -> command_surface::Answer<serde_json::Value> {
    let table = crate::commands::table(std::sync::Arc::clone(&runtime.daemon));
    let mut request = crate::client::request(operation, arguments, id.into());
    request.deadline = command_surface::Deadline::None;
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(table.call_as(
            request,
            command_surface::Via::Cli,
            command_surface::Caller::default().with_access(command_surface::Access::Write),
            Default::default(),
        ))
}

fn pending_requests(runtime: &Runtime) -> Vec<polylogue_model::authored::PreparedRequest> {
    let ModeState::Primary { connection } = &runtime.daemon.mode else {
        panic!("primary required")
    };
    let mut requests = Vec::new();
    polylogue_store::primary::Sinex::visit_pending_requests(connection.as_ref(), &mut |pending| {
        if let Some(prepared) = pending.prepared() {
            requests.push(prepared.clone());
        }
        Ok(())
    })
    .unwrap();
    requests
}

fn confirm_authored(
    home: &ClaudeHome,
    connection: std::sync::Arc<polylogue_sinex::Sinex>,
    records: Vec<(i64, &str, polylogue_model::authored::PreparedRequest)>,
) {
    use polylogue_store::primary::{Confirmed, ConfirmedChange, Product, ReplayControl};
    let (archive, _lock) =
        Archive::open_primary(&home.root, &home.dir.path().join("sinex-store")).unwrap();
    let writer = polylogue_store::writer::Writer::start_on_sinex(&archive, connection).unwrap();
    for (position, event, record) in records {
        writer
            .apply_confirmed(vec![
                Confirmed {
                    sinex_context: None,
                    position,
                    output_index: 0,
                    event: event.into(),
                    change: ConfirmedChange::Live(Box::new(Product::Request(record))),
                },
                Confirmed {
                    sinex_context: None,
                    position,
                    output_index: 0,
                    event: String::new(),
                    change: ConfirmedChange::Control(Box::new(ReplayControl::Completion {
                        position,
                        delivery: None,
                    })),
                },
            ])
            .unwrap();
    }
}

#[test]
fn offline_setting_and_chat_writes_keep_own_results() {
    let home = ClaudeHome::new();
    let runtime = launch(primary(&home)).unwrap();
    let setting = primary_table_call(
        &runtime,
        "set",
        serde_json::json!({"key":"browser.actions","value":"on"}),
        "enable-offline-chat",
    );
    assert_eq!(setting.exit_code(), 5, "{setting:?}");
    let (value, notes) = crate::test_answers::degraded(setting);
    assert!(
        notes
            .iter()
            .any(|note| note == crate::protocol::AWAITING_CONFIRMATION)
    );
    let crate::protocol::setting::Answer::Outcome(write_result) =
        serde_json::from_value(value).unwrap()
    else {
        panic!("setting write must keep its write result")
    };
    assert!(write_result.pending);
    assert!(!write_result.id.is_empty());
    let pending_before_retry = pending_requests(&runtime);
    let repeated = primary_table_call(
        &runtime,
        "set",
        serde_json::json!({"key":"browser.actions","value":"on"}),
        "enable-offline-chat",
    );
    let (value, _) = crate::test_answers::degraded(repeated);
    let crate::protocol::setting::Answer::Outcome(repeat) = serde_json::from_value(value).unwrap()
    else {
        panic!("repeat must keep its write result")
    };
    assert_eq!(repeat.id, write_result.id);
    assert!(repeat.pending && repeat.repeat);
    assert_eq!(pending_requests(&runtime), pending_before_retry);

    let bytes = b"Synthetic attachment awaiting Sinex confirmation";
    let args = serde_json::json!({"provider":"chatgpt","model":"example","text":"A synthetic draft","files":[{"name":"attachment.txt","media_type":"text/plain","bytes":bytes.to_vec()}]});
    let unconfirmed =
        primary_table_call(&runtime, "chat send", args.clone(), "pending-file-action");
    assert!(
        matches!(unconfirmed.outcome, command_surface::Outcome::Error(ref error)
        if error.code.as_deref() == Some("browser_actions_off")),
        "{unconfirmed:?}"
    );
    assert_eq!(pending_requests(&runtime), pending_before_retry);

    // Pending policy cannot authorize a browser action. Install the same
    // immutable setting through the normal confirmed writer before testing
    // either browser preparation boundary during a later transport outage.
    let ModeState::Primary { connection } = &runtime.daemon.mode else {
        panic!("primary mode required")
    };
    let connection = connection.clone();
    let prepared = pending_before_retry
        .into_iter()
        .find(|prepared| {
            prepared.outcome.id == write_result.id
                && matches!(&prepared.request.write,
            polylogue_model::user::UserWrite::SetSetting { key, .. } if key == "browser.actions")
        })
        .unwrap();
    assert!(!prepared.outcome.pending);
    drop(runtime);
    confirm_authored(
        &home,
        connection,
        vec![(1, "synthetic-confirmed-browser-policy", prepared)],
    );
    let runtime = launch(primary(&home)).unwrap();
    assert_eq!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .setting_value("browser.actions", "synthetic-machine")
            .unwrap(),
        Some(polylogue_model::settings::Value::Boolean(true))
    );
    let capture = primary_table_call(&runtime, "chat send", args.clone(), "pending-file-action");
    assert_eq!(capture.http_status(), Some(200), "{capture:?}");
    let (value, _) = crate::test_answers::degraded(capture);
    assert!(value.get("queued").is_none() && value.get("action").is_none());
    let crate::protocol::browser::ChatAnswer::Capture(capture) =
        serde_json::from_value(value).unwrap()
    else {
        panic!("a held file is not a queued action")
    };
    assert!(!capture.identity.is_empty());
    assert!(!pending_requests(&runtime).iter().any(|prepared| matches!(
        &prepared.request.write,
        polylogue_model::user::UserWrite::ChatEnqueue(_)
    )));
    let retry = primary_table_call(&runtime, "chat send", args, "pending-file-action");
    let (value, _) = crate::test_answers::degraded(retry);
    let crate::protocol::browser::ChatAnswer::Capture(retry) =
        serde_json::from_value(value).unwrap()
    else {
        panic!("retry must keep the held capture")
    };
    assert_eq!(retry.identity, capture.identity);

    let action = primary_table_call(
        &runtime,
        "chat send",
        serde_json::json!({"provider":"chatgpt","model":"example","text":"A synthetic draft without files"}),
        "pending-authored-action",
    );
    assert_eq!(action.http_status(), Some(200), "{action:?}");
    let (value, _) = crate::test_answers::degraded(action);
    assert!(value.get("queued").is_none() && value.get("action").is_none());
    let crate::protocol::browser::ChatAnswer::Outcome(write_result) =
        serde_json::from_value(value).unwrap()
    else {
        panic!("a held authored write must keep its write result")
    };
    assert!(write_result.pending);
    let prepared = pending_requests(&runtime);
    let action=prepared.iter().find(|prepared|matches!(&prepared.request.write,polylogue_model::user::UserWrite::ChatEnqueue(action) if action.id==write_result.id)).expect("the exact authored action is durable in the outbox");
    assert_eq!(action.outcome.id, write_result.id);
    assert!(
        runtime
            .daemon
            .reads
            .reader()
            .unwrap()
            .chat_actions()
            .unwrap()
            .is_empty(),
        "unconfirmed actions must not be presented as queued current rows"
    );
}

#[test]
fn an_offline_tag_retry_keeps_its_pending_intent_when_another_tag_is_confirmed() {
    use polylogue_model::{
        Reference,
        assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink},
        user::UserWrite,
    };
    let home = ClaudeHome::new();
    let runtime = launch(primary(&home)).unwrap();
    let arguments = serde_json::json!({"references":["session:codex:tag-target"], "steps":[{"verb":"tag","name":"status:ready"}]});
    let refused = primary_table_call(&runtime, "rows", arguments.clone(), "offline-tag");
    assert!(
        matches!(refused.outcome, command_surface::Outcome::Error(error)
        if matches!(error.code.as_deref(), Some("primary_context_missing" | "primary_context_unavailable")))
    );
    assert!(pending_requests(&runtime).is_empty());

    // The ordinary line must wait for Sinex context before resolving rows.
    // Exercise the actual verb with an already-resolved selection to isolate
    // pending-intent ownership from that independent context barrier.
    let tag = |runtime: &Runtime| {
        use crate::daemon::verbs::{Given, Selection, StepCall, Verb, annotate::Tag};
        let caller = crate::commands::Caller {
            surface: crate::commands::Surface::Cli,
            cwd: None,
        };
        let invocation = runtime
            .daemon
            .invocation(&polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Operator,
                via: polylogue_model::authorship::Channel::Cli,
            });
        let acquired = std::sync::atomic::AtomicBool::new(false);
        let call = crate::daemon::handler::Call {
            daemon: &runtime.daemon,
            row: crate::commands::command("rows").unwrap(),
            invocation: &invocation,
            caller: &caller,
            request_id: "offline-tag",
            cursor: None,
            budget: Default::default(),
            reader_acquired: &acquired,
        };
        let reference = Reference::parse("session:codex:tag-target").unwrap();
        let selection = Selection {
            given: Given::Reference(reference.clone()),
            resolved: Some(vec![reference]),
            question: None,
            retained: None,
        };
        let done = Tag::run(
            crate::protocol::steps::TagStep {
                name: "status:ready".into(),
            },
            &selection,
            &StepCall::new(&call, "offline-tag-1"),
        )
        .unwrap();
        let crate::protocol::line::StepAnswer::WroteEach(report) = done.reply else {
            panic!("expected a tag result")
        };
        report
    };
    let original = tag(&runtime);
    assert!(original.pending);
    assert!(!original.repeat);
    assert_eq!(original.written, 1);
    let pending_before = pending_requests(&runtime);
    assert!(
        pending_before
            .iter()
            .any(|record| record.request.id == "offline-tag-1")
    );
    let ModeState::Primary { connection } = &runtime.daemon.mode else {
        panic!("primary required")
    };
    let connection = connection.clone();
    drop(runtime);

    // A separate real writer makes a different accepted request for the same
    // value. Its confirmed effect must not erase the original offline intent.
    let seed = polylogue_store::fixture::writer::WriterArchive::new();
    seed.writer
        .user(polylogue_store::fixture::authored::operator_request(
            "separate-confirmed-tag".into(),
            UserWrite::Assert {
                assertions: vec![NewAssertion {
                    kind: AssertionKind::Tag,
                    key: Some("status".into()),
                    value: Some(serde_json::json!({"value":"ready"})),
                    valid_until: None,
                    links: vec![NewAssertionLink {
                        role: LinkRole::About,
                        reference: Reference::parse("session:codex:tag-target").unwrap(),
                    }],
                }],
            },
        ))
        .unwrap();
    let pool = polylogue_store::ReadPool::open(&seed.archive).unwrap();
    let mut records = Vec::new();
    pool.reader()
        .unwrap()
        .visit_authored(&mut |record| {
            records.push(record.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(records.len(), 1);
    let prepared = records.remove(0);
    confirm_authored(
        &home,
        connection,
        vec![(1, "synthetic-confirmed-tag", prepared)],
    );
    let runtime = launch(primary(&home)).unwrap();
    {
        let reader = runtime.daemon.reads.reader().unwrap();
        assert!(reader.user_request("offline-tag-1").unwrap().is_none());
        assert_eq!(
            reader
                .tagged(
                    &[Reference::parse("session:codex:tag-target").unwrap()],
                    "status",
                    Some(&serde_json::json!("ready")),
                    &polylogue_model::Author::Operator
                )
                .unwrap()
                .len(),
            1
        );
    }
    assert_eq!(pending_requests(&runtime), pending_before);
    let repeated = tag(&runtime);
    assert!(repeated.pending && repeated.repeat);
    assert_eq!(repeated.id, original.id);
    assert_eq!(repeated.written, original.written);
    assert!(repeated.already.is_empty());
    assert_eq!(pending_requests(&runtime), pending_before);
}
