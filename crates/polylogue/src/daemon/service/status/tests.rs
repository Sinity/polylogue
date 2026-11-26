use super::*;

/// While an index is built and none can be read, status answers from the
/// build: its header and steps, no archive figures, and no error fields
/// holding the build's words (docs/operations.md#status).
#[test]
fn a_build_that_serves_no_index_is_what_status_answers_from() {
    use crate::protocol::status::AnswersFrom;
    let daemon = crate::test_daemon::start("synthetic source");
    let header = polylogue_store::archive::index_build::IndexBuild {
        serving_previous: false,
        chains: 3,
        chains_read: 1,
        bytes: 900,
        bytes_read: 100,
    };
    let steps: polylogue_store::archive::build_progress::BuildProgress =
        serde_json::from_value(serde_json::json!({"to_come": ["installing"]})).unwrap();
    for vectors in [None, Some(Err(header.to_string()))] {
        let requested = vectors.is_some();
        let answer = unread(
            &daemon.daemon,
            AnswersFrom::Build,
            Some((header.clone(), steps.clone())),
            vectors,
            false,
            0,
        );
        assert_eq!(answer.exit_code(), 5);
        let command_surface::Outcome::Degraded { value, missing } = &answer.outcome else {
            panic!("{answer:?}");
        };
        assert_eq!(missing, &["the index is being built"]);
        assert_eq!(value.answers_from, AnswersFrom::Build);
        assert_eq!(
            (&value.index_build, &value.build_progress),
            (&Some(header.clone()), &Some(steps.clone()))
        );
        assert_eq!(
            (
                value.store.is_none(),
                value.totals_error.as_deref(),
                value.notices.is_none(),
                value.running.is_none(),
                value.quota.is_empty(),
            ),
            (true, None, true, true, true)
        );
        assert!(value.vectors.is_none() && value.verification.is_none());
        assert_eq!(
            value.vectors_error,
            requested.then(|| header.to_string()),
            "a refused vector read names the build"
        );
    }
}

/// Before a snapshot is read, status says since when: the daemon's start,
/// or a built index's install; once read, it answers from the snapshot with
/// the time it was read. Whether the scope index is loaded is `warm`'s
/// (docs/operations.md#status).
#[test]
fn status_says_what_it_answers_from() {
    use crate::protocol::status::{AnswersFrom, NotReadSince};
    let daemon = crate::test_daemon::start("synthetic source");
    let d = &daemon.daemon;
    assert!(matches!(
        StatusSnapshot::default().latest(),
        Latest::NotRead(NotReadSince::Start)
    ));
    let started = unread(
        d,
        AnswersFrom::NotReadYet {
            since: NotReadSince::Start,
        },
        None,
        None,
        false,
        0,
    );
    let command_surface::Outcome::Degraded { value, missing } = &started.outcome else {
        panic!("{started:?}");
    };
    assert_eq!(missing, &["figures not read yet since the daemon started"]);
    assert_eq!(
        (
            value.store.is_none(),
            value.totals_error.as_deref(),
            value.notices.is_none(),
            value.running.is_none()
        ),
        (true, None, true, true)
    );

    d.status_snapshot.clear(d);
    let installed = answer(d, None, false);
    assert_eq!(installed.exit_code(), 5);
    let command_surface::Outcome::Degraded { value, missing } = &installed.outcome else {
        panic!("{installed:?}");
    };
    assert_eq!(
        value.answers_from,
        AnswersFrom::NotReadYet {
            since: NotReadSince::Install
        }
    );
    assert_eq!(
        missing,
        &["figures not read yet since the new index was installed"]
    );
    assert!(value.warm.scope_index_loaded);

    let before = polylogue_model::Timestamp::from(d.clock.now());
    d.status_snapshot.refresh(d, &d.reads.reader().unwrap(), 0);
    let read = answer(d, None, false);
    let value = read.value().unwrap();
    let AnswersFrom::Snapshot { read_at } = value.answers_from else {
        panic!("{:?}", value.answers_from);
    };
    assert!(read_at >= before, "{read_at:?} {before:?}");
    assert_eq!(value.store.as_ref().map(|store| store.messages), Some(1));
    assert!(value.notices.is_some() && value.running.is_some());
}

#[test]
fn verification_retains_runtime_pending_and_named_problem_outcomes() {
    let daemon = crate::test_daemon::start("synthetic source");
    let status = answer(&daemon.daemon, None, false);
    let mut status = status.map(|mut status| {
        status.pending_writes.delivery_outcomes = 2;
        status.pending_writes.last_error = Some("synthetic retry pending".into());
        status
    });
    let value = crate::protocol::into_value(status).unwrap();
    status = command_surface::Answer::degraded(value, vec!["awaiting source confirmation".into()]);
    status.envelope.notes.push("synthetic runtime note".into());
    let checked = crate::daemon::verify::run(&daemon.daemon, None);
    let found = crate::protocol::into_value(checked).unwrap();
    let expected_verification = found.clone();
    let checked =
        command_surface::Answer::degraded(found, vec!["synthetic storage problem".into()]);
    let combined = with_verification(status, checked);
    let command_surface::Outcome::Degraded { value, missing } = combined.outcome else {
        panic!("{combined:?}");
    };
    assert_eq!(
        missing,
        ["awaiting source confirmation", "synthetic storage problem"]
    );
    assert_eq!(combined.envelope.notes, ["synthetic runtime note"]);
    assert_eq!(value.pending_writes.delivery_outcomes, 2);
    assert_eq!(
        value.pending_writes.last_error.as_deref(),
        Some("synthetic retry pending")
    );
    assert_eq!(value.verification, Some(expected_verification));
}

/// `recent_log` carries the daemon's latest records, a logged warning
/// among them; `--log` returns every kept record, more than an ordinary
/// answer's twenty.
#[test]
fn status_carries_the_recent_log_and_log_returns_every_kept_record() {
    let daemon = crate::test_daemon::start("synthetic source");
    for n in 0..30 {
        polylogue_store::log!(info, "synthetic progress {n}");
    }
    polylogue_store::log!(warning, "synthetic degraded reading");
    let status = |arguments| {
        let answer = crate::test_daemon::Call::new("status", arguments).answer(&daemon.daemon);
        let value = answer.value().expect("status answers").clone();
        serde_json::from_value::<crate::protocol::status::Answer>(value)
            .unwrap()
            .recent_log
    };
    let recent = status(serde_json::json!({}));
    assert_eq!(recent.len(), crate::protocol::status::RECENT_LOG);
    let last = recent.last().unwrap();
    assert_eq!(
        (last.priority, last.text.as_str()),
        (
            polylogue_store::log::Priority::Warning,
            "synthetic degraded reading"
        )
    );
    let whole = status(serde_json::json!({"log": true}));
    assert!(whole.len() > crate::protocol::status::RECENT_LOG);
    let progress: Vec<&str> = whole
        .iter()
        .map(|record| record.text.as_str())
        .filter(|text| text.starts_with("synthetic progress"))
        .collect();
    assert_eq!(progress.len(), 30);
    assert_eq!(progress.first(), Some(&"synthetic progress 0"));
}

/// Status is answered from the daemon's snapshot and its own state, so
/// repeated requests through the socket while a pass ingests a large corpus
/// each arrive within the status class's default deadline, also while every
/// read connection is held, as reads waiting on a disk busy with the
/// writer's I/O hold them (docs/operations.md#status,
/// docs/latency.md#warm-state).
#[test]
fn status_answers_within_its_default_deadline_while_a_pass_ingests_a_large_corpus() {
    use std::sync::Arc;
    const SESSIONS: usize = 400;
    const TURNS: usize = 60;
    /// Status requests sent while every read connection is held.
    const WHILE_HELD: usize = 20;
    let daemon = crate::test_daemon::start("synthetic source");
    // Neutral generated text, about 40 MB in all, written beside the
    // session the daemon has already read so its next pass ingests it.
    let project = daemon.home.join(".claude/projects/-corpus");
    std::fs::create_dir_all(&project).unwrap();
    let words = [
        "parse", "index", "archive", "record", "stream", "column", "window",
    ];
    for s in 0..SESSIONS {
        let session = format!("corpus-{s}");
        let mut text = String::new();
        for t in 0..TURNS {
            let line: String = (0..120)
                .map(|w| words[(s + t + w) % words.len()])
                .collect::<Vec<_>>()
                .join(" ");
            text.push_str(&polylogue_testkit::claude_user(
                &format!("{session}-u{t}"),
                &session,
                &line,
            ));
            text.push_str(&polylogue_testkit::claude_assistant(
                &format!("{session}-a{t}"),
                &session,
                &line,
            ));
        }
        std::fs::write(project.join(format!("{session}.jsonl")), text).unwrap();
    }
    let total = 1 + (SESSIONS * TURNS * 2) as u64;
    let socket = daemon.daemon.archive.socket();
    let status = || {
        let request = crate::client::request(
            "status",
            serde_json::json!({}),
            polylogue_store::user_writes::new_request_id(),
        );
        assert_eq!(request.deadline, command_surface::Deadline::Default);
        crate::client::call(&socket, &request).unwrap()
    };
    let mut late = Vec::new();
    let mut during_ingest = 0;
    let mut held_replies = 0;
    loop {
        let answer = status();
        let Some(value) = answer.value() else {
            late.push(format!("{:?}", answer.outcome));
            continue;
        };
        let messages = value["store"]["messages"].as_u64();
        if value["busy"] == true && messages.is_some_and(|n| n > 1 && n < total) {
            during_ingest += 1;
        }
        if messages == Some(total) && value["busy"] == false {
            break;
        }
        if during_ingest != 1 || held_replies > 0 {
            continue;
        }
        // Every connection, the interactive one included, held by a read
        // that does not end until released.
        let readers = daemon.daemon.reads.max_readers() + 1;
        let (held, acquired) = std::sync::mpsc::channel();
        let holders: Vec<_> = (0..readers)
            .map(|_| {
                let d = Arc::clone(&daemon.daemon);
                let held = held.clone();
                let (release, released) = std::sync::mpsc::sync_channel::<()>(0);
                let thread = std::thread::spawn(move || {
                    let reader = d
                        .reads
                        .checkout(
                            polylogue_store::Lane::Interactive,
                            &polylogue_store::read::limit::ReadLimit::without_deadline(
                                Default::default(),
                            ),
                        )
                        .unwrap();
                    reader.query("SELECT 1 AS n", &[]).unwrap();
                    held.send(()).unwrap();
                    released.recv().unwrap();
                });
                (release, thread)
            })
            .collect();
        for _ in 0..readers {
            acquired.recv().unwrap();
        }
        for _ in 0..WHILE_HELD {
            let answer = status();
            if answer.value().is_none() {
                late.push(format!("while held: {:?}", answer.outcome));
            }
            held_replies += 1;
        }
        for (release, thread) in holders {
            release.send(()).unwrap();
            thread.join().unwrap();
        }
    }
    assert_eq!(
        late,
        Vec::<String>::new(),
        "status replies past the deadline"
    );
    assert!(
        during_ingest > 0 && held_replies == WHILE_HELD,
        "status replies while the pass was ingesting: {during_ingest}, while reads were held: {held_replies}"
    );
}

/// Status takes no reader (docs/operations.md, interactive work): with
/// every connection of the read pool held, the interactive one included,
/// status through the socket still answers within its default deadline.
#[test]
fn status_answers_while_every_reader_is_held() {
    let daemon = crate::test_daemon::start("synthetic source");
    let readers = daemon.daemon.reads.max_readers() + 1;
    let (taken, held) = std::sync::mpsc::channel();
    let releases: Vec<_> = (0..readers)
        .map(|_| {
            let daemon = std::sync::Arc::clone(&daemon.daemon);
            let taken = taken.clone();
            let (release, released) = std::sync::mpsc::channel::<()>();
            std::thread::spawn(move || {
                let limit =
                    polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
                let _reader = daemon
                    .reads
                    .checkout(polylogue_store::Lane::Interactive, &limit)
                    .unwrap();
                taken.send(()).unwrap();
                let _ = released.recv();
            });
            release
        })
        .collect();
    for _ in 0..readers {
        held.recv_timeout(std::time::Duration::from_secs(10))
            .expect("every reader is held");
    }
    let request = crate::client::request(
        "status",
        serde_json::json!({}),
        polylogue_store::user_writes::new_request_id(),
    );
    let answer = crate::client::call(&daemon.daemon.archive.socket(), &request).unwrap();
    drop(releases);
    let status = answer
        .value()
        .unwrap_or_else(|| panic!("{:?}", answer.outcome));
    assert_eq!(status["pid"], std::process::id());
}

/// `threads` lists each daemon thread that is doing something, with what it
/// is doing, outermost first, and how long since it made progress; a thread
/// whose work ended is not listed (docs/operations.md#status).
#[test]
fn status_lists_what_each_busy_thread_is_doing() {
    use std::sync::mpsc::channel;
    const THREAD: &str = "status-test-busy";
    let daemon = crate::test_daemon::start("synthetic source");
    let threads = || {
        let answer =
            crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&daemon.daemon);
        let value = answer.value().expect("status answers").clone();
        serde_json::from_value::<crate::protocol::status::Answer>(value)
            .unwrap()
            .threads
            .into_iter()
            .find(|thread| thread.name == THREAD)
    };
    let (busy_tx, busy) = channel::<()>();
    let (finish, finish_rx) = channel::<()>();
    let (idle_tx, idle) = channel::<()>();
    let (exit, exit_rx) = channel::<()>();
    let worker = std::thread::Builder::new()
        .name(THREAD.into())
        .spawn(move || {
            {
                let _outer = polylogue_store::activity::doing("building the index");
                let _inner = polylogue_store::activity::doing_on("indexing", "sessions");
                busy_tx.send(()).unwrap();
                finish_rx.recv().unwrap();
            }
            idle_tx.send(()).unwrap();
            exit_rx.recv().unwrap();
        })
        .unwrap();
    busy.recv().unwrap();
    let busy = threads().expect("a busy thread is listed");
    assert_eq!(
        (
            busy.doing
                .iter()
                .map(|work| (work.what.as_str(), work.subject.as_deref()))
                .collect::<Vec<_>>(),
            busy.seconds_since_progress.is_some(),
            busy.statements,
        ),
        (
            vec![("building the index", None), ("indexing", Some("sessions"))],
            true,
            Vec::new(),
        )
    );
    finish.send(()).unwrap();
    idle.recv().unwrap();
    assert_eq!(threads(), None, "an idle thread is not listed");
    exit.send(()).unwrap();
    worker.join().unwrap();
}

#[test]
fn lasting_worker_failure_is_a_status_problem() {
    use polylogue_model::{Cause, Code, Failure};
    use polylogue_store::{StoreError, completion::CommitEvents};
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    };

    let daemon = crate::test_daemon::start("synthetic source");
    for cause in [Cause::Transient, Cause::Environment] {
        let events = CommitEvents::default();
        let subscription = events.subscribe();
        let wake = subscription.wake_handle();
        let stop = Arc::new(AtomicBool::new(false));
        let stopped = stop.clone();
        let (sent, changed) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("status failure test".into())
            .spawn(move || {
                polylogue_store::follow::follow(
                    "status failure test",
                    Some(&subscription),
                    &stopped,
                    || polylogue_model::Timestamp::from_unix_ms(1_700_000_000_000).unwrap(),
                    || sent.send(()).unwrap(),
                    || {
                        Err(StoreError::Failed(Failure::stated(
                            "",
                            cause,
                            Code::Io,
                            "synthetic worker failure",
                        )))
                    },
                )
            })
            .unwrap();
        changed.recv().unwrap();
        let state = if cause == Cause::Transient {
            "failing"
        } else {
            "halted"
        };
        let expected = format!("status failure test: {state}: synthetic worker failure");
        for reply in [
            answer(&daemon.daemon, None, false),
            unread(
                &daemon.daemon,
                crate::protocol::status::AnswersFrom::NotReadYet {
                    since: crate::protocol::status::NotReadSince::Start,
                },
                None,
                None,
                false,
                0,
            ),
        ] {
            let command_surface::Outcome::Degraded { value, missing } = reply.outcome else {
                panic!("worker failure must degrade status");
            };
            assert!(missing.contains(&expected), "{missing:?}");
            let worker = value
                .threads
                .iter()
                .find(|thread| thread.name == "status failure test")
                .unwrap()
                .worker
                .as_ref()
                .unwrap();
            assert_eq!(
                worker.problem("status failure test"),
                Some(expected.clone())
            );
        }
        stop.store(true, Ordering::Release);
        let _ = wake.try_send(());
        assert!(thread.join().unwrap());
    }
}
