use super::*;
use crate::brief::Ask;
use crate::test_daemon;
use polylogue_engine::Config;
use polylogue_model::user::UserWrite;
use polylogue_store::{Archive, StoreError};
use std::sync::Arc;
use std::sync::mpsc::sync_channel;

fn read(
    d: &Daemon,
    view: &str,
    format: &str,
    budget: Option<usize>,
    cursor: Option<&command_surface::RunnableRequest>,
) -> command_surface::Answer<serde_json::Value> {
    let Some(view) = polylogue_model::read::View::parse(view) else {
        return error(polylogue_model::Code::BadRequest, "unknown reading view");
    };
    if polylogue_render::Format::parse(format).is_none() {
        return error(polylogue_model::Code::BadRequest, "unknown reading format");
    }
    let request = if let Some(resume) = cursor {
        let mut resume = resume.clone();
        if let Some(budget) = budget {
            resume.arguments["steps"][0]["budget"] = serde_json::json!(budget);
        } else {
            resume.arguments["steps"][0]
                .as_object_mut()
                .unwrap()
                .remove("budget");
        }
        crate::test_daemon::Call::from_request(&resume)
    } else {
        crate::test_daemon::read_call(polylogue_reading::ReadRequest {
            start: Default::default(),
            references: None,
            target: "session:claude-code:s1".into(),
            options: polylogue_model::read::ReadOptions {
                view,
                ..Default::default()
            },
            delivery: polylogue_model::read::ReadDelivery::Page,
            format: polylogue_model::read::Format::parse(format).unwrap(),
            budget: budget.map_or(polylogue_model::read::ReadBudget::Default, |tokens| {
                polylogue_model::read::ReadBudget::Tokens(tokens as u64)
            }),
            alternates: Vec::new(),
            cursor: None,
        })
    };
    request.answer(d)
}

fn typed(response: command_surface::Answer<serde_json::Value>) -> polylogue_model::read::ReadPage {
    crate::test_answers::read(response)
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
    let envelope: command_surface::RunnableRequest = call.request.clone();
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
    malformed.arguments["steps"][0]["cursor"] = serde_json::json!("m0.b0.c0");
    for (view, format, cursor, expected) in [
        ("nope", "json", None, "bad_request"),
        ("transcript", "pdf", None, "bad_request"),
        ("transcript", "json", Some(&malformed), "invalid_query"),
    ] {
        assert!(
            crate::test_answers::refusal(read(&w.daemon, view, format, None, cursor))
                .code
                .as_deref()
                == Some(expected),
            "{view} {format} {cursor:?}"
        );
    }
}

#[test]
fn reading_alternates_setting_is_the_default_and_is_bound_in_the_cursor() {
    let w = test_daemon::start("a prompt long enough to need two pages of one token each");
    let set = |value: &str| {
        crate::test_answers::ok(
            crate::test_daemon::Call::new(
                "set",
                serde_json::json!({"key": "reading.alternates", "value": value}),
            )
            .answer(&w.daemon),
        );
    };
    let next = |page: &polylogue_model::read::ReadPage| -> command_surface::RunnableRequest {
        let call = page.document.footer.next.as_ref().expect("another page");
        call.request.clone()
    };
    // The continuation is the step it came from; its cursor carries the
    // alternates the first page resolved (docs/reading.md, Alternates).
    let alternates = |call: &command_surface::RunnableRequest| -> Vec<String> {
        use polylogue_model::read::{CursorSeal, PageCursor};
        let reader = w.daemon.reads.reader().unwrap();
        let seal = CursorSeal {
            key: reader.cursor_key().unwrap(),
            index_instance: reader.change_position().unwrap().index_instance,
        };
        PageCursor::open(
            call.arguments["steps"][0]["cursor"].as_str().unwrap(),
            &seal,
        )
        .unwrap()
        .alternates
    };
    set(r#"["translation:pl"]"#);
    let first = next(&typed(read(&w.daemon, "transcript", "json", Some(1), None)));
    assert_eq!(alternates(&first), ["translation:pl"]);
    // The setting changes; the reading keeps what its first page resolved,
    // even when the continuation names no alternate.
    set("[]");
    let mut bare = first.clone();
    bare.arguments["steps"][0]["alternate"] = serde_json::Value::Null;
    let second = typed(read(&w.daemon, "transcript", "json", Some(1), Some(&bare)));
    if second.document.footer.next.is_some() {
        assert_eq!(alternates(&next(&second)), ["translation:pl"]);
    }
    // A new reading follows the setting as it is now.
    let fresh = next(&typed(read(&w.daemon, "transcript", "json", Some(1), None)));
    assert!(alternates(&fresh).is_empty());
    // `none` asks for the originals whatever the setting holds.
    set(r#"["translation:pl"]"#);
    let originals = crate::test_daemon::read_call(polylogue_reading::ReadRequest {
        start: Default::default(),
        references: None,
        target: "session:claude-code:s1".into(),
        options: polylogue_model::read::ReadOptions {
            alternate: Some("none".into()),
            ..Default::default()
        },
        delivery: polylogue_model::read::ReadDelivery::Page,
        format: polylogue_model::read::Format::Json,
        budget: polylogue_model::read::ReadBudget::Tokens(1),
        alternates: Vec::new(),
        cursor: None,
    })
    .answer(&w.daemon);
    assert!(alternates(&next(&typed(originals))).is_empty());
}

#[test]
fn several_setting_alternates_survive_next_and_previous_when_the_setting_changes() {
    use polylogue_model::read::{CursorSeal, PageCursor};
    let w = test_daemon::start(&"a synthetic sentence for paging ".repeat(100));
    let set = |value: &str| {
        crate::test_answers::ok(
            crate::test_daemon::Call::new(
                "set",
                serde_json::json!({"key":"reading.alternates","value":value}),
            )
            .answer(&w.daemon),
        );
    };
    set(r#"["translation:pl","translation:de"]"#);
    let first = typed(read(&w.daemon, "transcript", "json", Some(1), None));
    let next = first.document.footer.next.as_ref().unwrap().request.clone();
    let alternates = |request: &command_surface::RunnableRequest| {
        let reader = w.daemon.reads.reader().unwrap();
        let seal = CursorSeal {
            key: reader.cursor_key().unwrap(),
            index_instance: reader.change_position().unwrap().index_instance,
        };
        PageCursor::open(
            request.arguments["steps"][0]["cursor"].as_str().unwrap(),
            &seal,
        )
        .unwrap()
        .alternates
    };
    assert_eq!(alternates(&next), ["translation:pl", "translation:de"]);
    assert!(next.arguments["steps"][0].get("alternate").is_none());
    set("[]");
    let second = typed(read(&w.daemon, "transcript", "json", Some(1), Some(&next)));
    let previous = second
        .document
        .footer
        .previous
        .as_ref()
        .unwrap()
        .request
        .clone();
    assert_eq!(alternates(&previous), ["translation:pl", "translation:de"]);
    let original = typed(read(
        &w.daemon,
        "transcript",
        "json",
        Some(1),
        Some(&previous),
    ));
    assert_eq!(original.document.items, first.document.items);
    assert_eq!(original.document.header, first.document.header);
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
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("shutdown did not return: the signal timer was not woken")
        .unwrap();
    thread.join().unwrap();
    assert!(
        matches!(crate::test_daemon::Call::new("rows", serde_json::json!({"selection": "sessions"})).answer(&daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
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
        assert!(crate::daemon::server::Server::start(Arc::clone(&runtime.daemon)).is_err());
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
    let check_empty_status = |daemon: &Daemon| {
        let answer = crate::test_daemon::Call::new("status", serde_json::json!({})).answer(daemon);
        assert!(
            matches!(
                &answer.outcome,
                command_surface::Outcome::Ok(_) | command_surface::Outcome::Degraded { .. }
            ),
            "{answer:?}"
        );
        assert_eq!(answer.value().unwrap()["store"]["sessions"], 0);
    };
    check_empty_status(&old);
    runtime.shutdown().unwrap();
    assert!(matches!(old.reads.reader(), Err(StoreError::Stopped)));
    let mut next =
        test_daemon::start_runtime(&root, config(dir.path()), ArchiveMode::Standalone).unwrap();
    check_empty_status(&next.daemon);
    next.shutdown().unwrap();
    assert!(
        matches!(crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&old), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
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
    // Every ordinary connection and the reserved interactive one are held
    // before any request can start.
    let (held, acquired) = std::sync::mpsc::channel();
    let mut holders = Vec::new();
    for _ in 0..daemon.reads.max_readers() {
        let daemon = Arc::clone(&daemon);
        let held = held.clone();
        let (release, released) = sync_channel(0);
        let holder = std::thread::spawn(move || {
            let limit =
                polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
            let reader = daemon
                .reads
                .checkout(polylogue_store::Lane::Interactive, &limit)
                .unwrap();
            assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
            held.send(()).unwrap();
            released.recv().unwrap();
            assert!(matches!(
                reader.query("SELECT 42 AS n", &[]),
                Err(StoreError::Stopped)
            ));
        });
        holders.push((release, holder));
    }
    for _ in 0..daemon.reads.max_readers() {
        acquired.recv().unwrap();
    }
    assert_eq!(daemon.reads.checked_out(), daemon.reads.max_readers() + 1);
    let mut requests = Vec::new();
    let entered = Arc::new(std::sync::Barrier::new(9));
    for _ in 0..8 {
        let daemon = Arc::clone(&daemon);
        let entered = Arc::clone(&entered);
        requests.push(std::thread::spawn(move || {
            entered.wait();
            crate::test_daemon::Call::new("rows", serde_json::json!({"selection": "sessions"}))
                .answer(&daemon)
        }));
    }
    entered.wait();
    runtime.stop_accepting();
    for request in requests {
        let answer = request.join().unwrap();
        assert!(
            matches!(&answer, command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } if error.code.as_deref() == Some("shutting_down")),
            "{answer:?}"
        );
    }
    for (release, holder) in holders {
        release.send(()).unwrap();
        holder.join().unwrap();
    }
    runtime.shutdown().unwrap();
    // A reader held across the stop ends with it.
    assert!(matches!(
        reader.query("SELECT 42 AS n", &[]),
        Err(StoreError::Stopped)
    ));
    assert!(
        matches!(crate::test_daemon::Call::new("rows", serde_json::json!({"selection": "sessions"})).answer(&daemon), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
    );
}

#[test]
fn invalid_reader_resource_configuration_refuses_startup_and_releases_the_lock() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("archive");
    for (workers, memory, refusal) in [
        (0, 128 << 20, "workers must be positive"),
        (
            2,
            3 * 3072 - 1,
            "reader memory budget needs at least 3 KiB per connection",
        ),
    ] {
        let mut config = config(dir.path());
        config.workers = workers;
        config.reader_memory_bytes = memory;
        let Err(error) = test_daemon::start_runtime(&root, config, ArchiveMode::Standalone) else {
            panic!("{workers} workers with {memory} bytes started");
        };
        assert!(error.to_string().contains(refusal), "{error}");
        assert!(Archive::open_or_create(&root).is_ok());
    }
}

#[test]
fn read_and_query_share_stored_titles() {
    let w = test_daemon::start("First human question\nMore details");
    let source = w.home.join(".claude/projects/-p/s1.jsonl");
    let original = std::fs::read(&source).unwrap();
    let records = || {
        let request = crate::test_daemon::Call::new(
            "rows",
            serde_json::json!({"selection": "session:claude-code:s1", "steps":[{"verb":"read", "delivery": "stream", "format": "jsonl"}]}),
        );
        let envelope = command_surface::Request {
            operation: "rows".into(),
            arguments: request.arguments.clone(),
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
        crate::client::receive(&mut socket, None, |event| {
            if event.event == "record" {
                let transfer: polylogue_model::read::ReadRecord =
                    serde_json::from_str(&event.data).map_err(std::io::Error::other)?;
                if let Some(record) = transfer.record() {
                    records.push(record.clone());
                }
            }
            Ok(false)
        })
        .unwrap()
        .expect("the line answers");
        records
    };
    let before = records();
    let stored = before
        .iter()
        .find_map(|record| {
            if let polylogue_model::bundle::Record::Session(record) = record {
                Some(&record.session)
            } else {
                None
            }
        })
        .unwrap();
    assert!(stored.title.is_none());
    let response =
        crate::test_daemon::Call::new("rows", serde_json::json!({"selection": "sessions"}))
            .answer(&w.daemon);
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    assert_eq!(data["reply"]["rows"][0]["title"], "First human question");
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
mod read_seen;

mod reading_shapes;

mod hook_errors;

mod index_rebuild;

mod codex_reports;

mod attachments;

mod judgments;

mod saved;

mod deliveries;
mod outcomes;

mod stopping;
mod workers;

#[test]
fn an_invalidated_read_phase_does_not_enter_the_authored_writer() {
    let daemon = crate::test_daemon::start("write scope check");
    let invocation =
        daemon
            .daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
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
    let daemon = crate::test_daemon::start("pinned authored write");
    let invocation =
        daemon
            .daemon
            .invocation(&polylogue_reading::operation::Context::worker_authorship(
                "pinned-write-fixture",
            ));
    let reader = daemon.daemon.reads.reader().unwrap();
    let selected_scope = reader.scope_token();
    drop(reader);
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    // No active ReadLimit: the queued read scope token decides here.
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

mod advisory;

mod read_annotations;
mod read_attachments;
mod read_composition;
mod read_links;
mod read_objects;
mod read_runs;
mod read_saved_queries;

mod read_failures;

mod read_trace;

mod import_provider;
mod import_refusals;
mod tags;
mod withdrawals;
