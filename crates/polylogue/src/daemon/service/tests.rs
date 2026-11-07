use super::*;
use crate::test_daemon;

fn read(
    d: &Daemon,
    view: &str,
    format: &str,
    budget: Option<usize>,
    cursor: Option<&command_surface::RunnableRequest>,
) -> command_surface::Answer<serde_json::Value> {
    let Some(view) = polylogue_model::read::View::parse(view) else {
        return error("bad_request", "unknown reading view");
    };
    if polylogue_render::Format::parse(format).is_none() {
        return error("bad_request", "unknown reading format");
    }
    let request = if let Some(resume) = cursor {
        let mut resume = resume.clone();
        let mut arguments: crate::protocol::read::Arguments =
            serde_json::from_value(resume.arguments.clone()).unwrap();
        arguments.page_tokens =
            budget.map(|value| std::num::NonZeroU64::new(value as u64).unwrap());
        resume.arguments = serde_json::to_value(arguments).unwrap();
        crate::protocol::operations::admit(
            crate::commands::command("read").unwrap(),
            &resume,
            &crate::commands::Caller {
                surface: crate::commands::Surface::Http,
                cwd: None,
                json: true,
            },
        )
        .unwrap()
    } else {
        Request::Read {
            target: "session:claude-code:s1".into(),
            options: polylogue_model::read::ReadOptions {
                view,
                ..Default::default()
            },
            delivery: polylogue_model::read::ReadDelivery::Page,
            format: format.into(),
            budget,
            with_blobs: false,
            cursor: None,
        }
    };
    handle(request, d)
}

fn typed(response: command_surface::Answer<serde_json::Value>) -> polylogue_model::read::ReadPage {
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    serde_json::from_value(data).unwrap()
}
fn encoded(page: &polylogue_model::read::ReadPage, format: polylogue_render::Format) -> String {
    let mut bytes = Vec::new();
    polylogue_render::encode_document(
        &page.document,
        format,
        polylogue_render::EncodeOptions::default(),
        &mut bytes,
    )
    .unwrap();
    String::from_utf8(bytes).unwrap()
}

#[test]
fn a_read_honours_its_view_format_and_budget() {
    use polylogue_model::read::{ItemKind, View};
    let w = test_daemon::start("a prompt long enough to need two pages of one token each");
    let page = typed(read(&w.daemon, "transcript", "markdown", None, None));
    assert!(encoded(&page, polylogue_render::Format::Markdown).contains("a prompt long enough"));
    let page = typed(read(&w.daemon, "transcript", "json", None, None));
    assert!(page.document.items.iter().any(|item| matches!(&item.kind, ItemKind::Text { text } if text.contains("a prompt long enough"))));
    let changes = typed(read(&w.daemon, "changes", "json", None, None));
    assert_eq!(changes.document.header.view, View::Changes);
    assert!(
        !changes
            .document
            .items
            .iter()
            .any(|item| matches!(&item.kind, ItemKind::Text { .. }))
    );
    let first = typed(read(&w.daemon, "transcript", "json", Some(1), None));
    let call = first.document.footer.next.as_ref().expect("a second page");
    let envelope: command_surface::RunnableRequest =
        serde_json::from_value(call.http.arguments.clone()).unwrap();
    let cursor = envelope;
    let second = typed(read(
        &w.daemon,
        "transcript",
        "json",
        Some(1),
        Some(&cursor),
    ));
    assert_ne!(first.document.items, second.document.items);
    let mut malformed = cursor.clone();
    malformed.cursor = Some("m0.b0.c0".into());
    for (view, format, cursor, expected) in [
        ("nope", "json", None, "bad_request"),
        ("transcript", "pdf", None, "bad_request"),
        ("transcript", "json", Some(&malformed), "invalid_query"),
    ] {
        assert!(
            matches!(
                read(&w.daemon, view, format, None, cursor),
                command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some(expected)
            ),
            "{view} {format} {cursor:?}"
        );
    }
}

fn config(home: &std::path::Path) -> Config {
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = home.to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    config
}

#[test]
fn runtime_shutdown_wakes_signals_releases_lock_and_refuses_old_handles() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let mut runtime =
        test_daemon::start_runtime(&root, config(dir.path()), ArchiveMode::Standalone).unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    assert!(daemon.archive.lock_held());
    let (done, stopped) = sync_channel(1);
    let thread = std::thread::spawn(move || {
        let result = runtime.shutdown();
        done.send(result).unwrap();
    });
    stopped
        .recv()
        .expect("shutdown waited for signal timer")
        .unwrap();
    thread.join().unwrap();
    assert!(
        matches!(handle(Request::Status, &daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
    );
    assert!(
        Archive::open_or_create(&root).is_ok(),
        "runtime still holds its writer lock"
    );
}

#[test]
fn startup_failure_after_workers_start_releases_the_runtime_and_lock() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let weak;
    {
        let runtime =
            test_daemon::start_runtime(&root, config(dir.path()), ArchiveMode::Standalone).unwrap();
        weak = Arc::downgrade(&runtime.daemon);
        // A directory cannot be replaced by the Unix socket binder.
        std::fs::create_dir(runtime.daemon.archive.socket()).unwrap();
        assert!(
            crate::daemon::server::Server::start(
                Arc::clone(&runtime.daemon),
                Arc::clone(&runtime.operations)
            )
            .is_err()
        );
    }
    assert!(weak.upgrade().is_none());
    assert!(Archive::open_or_create(&root).is_ok());
}

#[test]
fn a_panicking_test_drops_its_in_process_daemon_workers() {
    let (sent, received) = sync_channel(1);
    assert!(
        std::panic::catch_unwind(|| {
            let daemon = test_daemon::start("panic fixture");
            sent.send(Arc::downgrade(&daemon.daemon)).unwrap();
            panic!("synthetic test panic");
        })
        .is_err()
    );
    assert!(received.recv().unwrap().upgrade().is_none());
}

#[test]
fn repeated_test_daemon_drops_release_every_shared_daemon_reference() {
    for _ in 0..3 {
        let daemon = test_daemon::start("owned fixture");
        let weak = Arc::downgrade(&daemon.daemon);
        drop(daemon);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn an_old_daemon_handle_does_not_prevent_empty_archive_restart() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    let mut runtime =
        test_daemon::start_runtime(&root, config(dir.path()), ArchiveMode::Standalone).unwrap();
    let old = Arc::clone(&runtime.daemon);
    crate::test_answers::ok(handle(Request::Status, &old));
    runtime.shutdown().unwrap();
    assert!(matches!(
        old.reads.reader(),
        Err(StoreError::ReadPoolClosed)
    ));
    let mut next =
        test_daemon::start_runtime(&root, config(dir.path()), ArchiveMode::Standalone).unwrap();
    crate::test_answers::ok(handle(Request::Status, &next.daemon));
    next.shutdown().unwrap();
    assert!(
        matches!(handle(Request::Status, &old), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
    );
}

#[cfg(feature = "sinex")]
#[test]
fn standalone_start_does_not_read_the_sinex_configuration() {
    let dir = tempfile::tempdir().unwrap();
    let configuration = dir.path().join(".config/polylogue");
    std::fs::create_dir_all(&configuration).unwrap();
    std::fs::write(configuration.join("sinex.toml"), "invalid deployment bytes").unwrap();
    let mut runtime = test_daemon::start_runtime(
        &dir.path().join("archive"),
        config(dir.path()),
        ArchiveMode::Standalone,
    )
    .unwrap();
    assert!(!runtime.daemon.archive.is_primary());
    assert!(runtime.daemon.archive.archive_db().ends_with("archive.db"));
    runtime.shutdown().unwrap();
}

#[test]
fn stopping_releases_requests_waiting_for_a_full_pool_before_workers_or_running_readers_finish() {
    let dir = tempfile::tempdir().unwrap();
    let mut runtime = test_daemon::start_runtime(
        &dir.path().join("archive"),
        config(dir.path()),
        ArchiveMode::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    let mut requests = Vec::new();
    let entered = Arc::new(std::sync::Barrier::new(9));
    for _ in 0..8 {
        let daemon = Arc::clone(&daemon);
        let entered = Arc::clone(&entered);
        requests.push(std::thread::spawn(move || {
            entered.wait();
            handle(Request::Status, &daemon)
        }));
    }
    entered.wait();
    runtime.stop_accepting();
    for request in requests {
        assert!(
            matches!(request.join().unwrap(), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
        );
    }
    runtime.shutdown().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    assert!(
        matches!(handle(Request::Status, &daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
    );
}

#[test]
fn invalid_reader_resource_configuration_refuses_startup_and_releases_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    for (workers, memory) in [(0, 128 << 20), (2, 3 * 3072 - 1)] {
        let mut config = config(dir.path());
        config.workers = workers;
        config.reader_memory_bytes = memory;
        assert!(test_daemon::start_runtime(&root, config, ArchiveMode::Standalone).is_err());
        assert!(Archive::open_or_create(&root).is_ok());
    }
}

#[test]
fn read_and_session_query_share_stored_titles_without_changing_source_or_canonical_records() {
    let w = test_daemon::start("First human question\nMore details");
    let source = w.home.join(".claude/projects/-p/s1.jsonl");
    let original = std::fs::read(&source).unwrap();
    let records = || {
        let request = Request::Read {
            target: "session:claude-code:s1".into(),
            options: polylogue_model::read::ReadOptions::default(),
            delivery: polylogue_model::read::ReadDelivery::Stream,
            format: "jsonl".into(),
            budget: None,
            with_blobs: false,
            cursor: None,
        };
        let envelope = command_surface::Request {
            operation: "read".into(),
            arguments: serde_json::to_value(
                crate::protocol::read::Arguments::from_request(&request).unwrap(),
            )
            .unwrap(),
            request_id: uuid::Uuid::now_v7().to_string(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let mut socket =
            std::os::unix::net::UnixStream::connect(w.daemon.archive.socket()).unwrap();
        socket
            .set_read_timeout(Some(std::time::Duration::from_secs(30)))
            .unwrap();
        serde_json::to_writer(
            &mut socket,
            &crate::protocol::wire::Outgoing {
                request: &envelope,
                caller: Default::default(),
                parent_deadline_ms: None,
            },
        )
        .unwrap();
        use std::io::Write;
        socket.write_all(b"\n").unwrap();
        let mut records = Vec::new();
        crate::client::stream::receive(socket, |event| {
            if event.event == "record" {
                let transfer: polylogue_model::read::ReadRecord =
                    serde_json::from_str(&event.data).map_err(std::io::Error::other)?;
                if let Some(record) = transfer.record() {
                    records.push(record.clone());
                }
            }
            Ok(event.event == "complete")
        })
        .unwrap();
        records
    };
    let before = records();
    let canonical = before
        .iter()
        .find_map(|record| {
            if let polylogue_model::bundle::Record::Session(record) = record {
                Some(&record.session)
            } else {
                None
            }
        })
        .unwrap();
    assert!(canonical.title.is_none());
    let response = handle(
        Request::Query {
            text: "sessions".into(),
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
        &w.daemon,
    );
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    assert_eq!(data["rows"][0]["title"], "First human question");
    for (format, title) in [
        (polylogue_render::Format::Markdown, "# First human question"),
        (
            polylogue_render::Format::Html,
            "<h1>First human question</h1>",
        ),
        (polylogue_render::Format::Terminal, "First human question\n"),
    ] {
        let page = typed(read(&w.daemon, "transcript", format.name(), None, None));
        assert!(encoded(&page, format).contains(title));
    }
    assert_eq!(records(), before);
    assert_eq!(std::fs::read(source).unwrap(), original);
}

#[cfg(feature = "sinex")]
mod mode;

mod read_pages;

mod reading_shapes;

mod hook_errors;

mod codex_reports;

mod attachments;

mod judgments;

mod saved;

mod deliveries;
mod outcomes;

#[test]
fn an_invalidated_read_phase_does_not_enter_the_authored_writer() {
    let daemon = crate::test_daemon::start("native write admission scope");
    let invocation = daemon
        .daemon
        .invocation(&crate::operation::Context::worker_authorship(
            "scope-write-fixture",
        ));
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
    let response = limit.scope(|| {
        drop(daemon.daemon.reads.reader().unwrap());
        daemon.daemon.archive.advance_generation_for_test().unwrap();
        write(
            &daemon.daemon,
            &invocation,
            "stale-read-write".into(),
            UserWrite::SetSetting {
                key: "browser.actions".into(),
                value: "on".into(),
                machine: None,
            },
        )
    });
    assert!(
        matches!(response,command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("scope_changed")),
        "{response:?}"
    );
    let reader = daemon.daemon.reads.reader().unwrap();
    let current = reader
        .setting_current("browser.actions", &invocation.machine)
        .unwrap();
    assert!(current.authored.is_none());
    assert_eq!(
        current.value,
        Some(polylogue_model::settings::Value::Boolean(false))
    );
}

#[test]
fn read_derived_write_keeps_its_selected_scope_after_reader_release() {
    let daemon = crate::test_daemon::start("pinned authored admission");
    let invocation = daemon
        .daemon
        .invocation(&crate::operation::Context::worker_authorship(
            "pinned-write-fixture",
        ));
    let reader = daemon.daemon.reads.reader().unwrap();
    let selected_scope = reader.scope_token();
    drop(reader);
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    // No active ReadLimit: the queued native token is the authority here.
    let response = write_from_read(
        &daemon.daemon,
        &invocation,
        "old-generation-write".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
        selected_scope,
    );
    assert!(
        matches!(response, command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("scope_changed")),
        "{response:?}"
    );
    let reader = daemon.daemon.reads.reader().unwrap();
    assert!(
        reader
            .user_request("old-generation-write")
            .unwrap()
            .is_none()
    );
    let current = reader
        .setting_current("browser.actions", &invocation.machine)
        .unwrap();
    assert!(current.authored.is_none());
    let selected_scope = reader.scope_token();
    drop(reader);
    let response = write_from_read(
        &daemon.daemon,
        &invocation,
        "current-generation-write".into(),
        UserWrite::SetSetting {
            key: "browser.actions".into(),
            value: "on".into(),
            machine: None,
        },
        selected_scope,
    );
    crate::test_answers::ok(response);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .unwrap()
            .user_request("current-generation-write")
            .unwrap()
            .is_some()
    );
}
