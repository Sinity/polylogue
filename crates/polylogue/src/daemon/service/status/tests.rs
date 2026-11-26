use super::*;

#[test]
fn building_status_keeps_progress_and_does_not_invent_empty_vectors() {
    let daemon = crate::test_daemon::start("synthetic source");
    let progress = polylogue_store::archive::index_build::IndexBuild {
        reason: polylogue_store::index::file::RebuildReason::NoIndex,
        chains: 3,
        chains_read: 1,
        bytes: 900,
        bytes_read: 100,
    };
    for vectors in [None, Some(Err(progress.to_string()))] {
        let requested = vectors.is_some();
        let answer = unread(&daemon.daemon, Some(progress.clone()), vectors, false, 0);
        assert_eq!(answer.exit_code(), 5);
        let status = answer.value().unwrap();
        assert_eq!(status.index_build, Some(progress.clone()));
        assert_eq!(status.totals_error, Some(progress.to_string()));
        assert!(!status.warming, "a build names itself, it is not warming");
        assert!(status.vectors.is_none() && status.verification.is_none());
        assert_eq!(
            status.vectors_error,
            requested.then(|| progress.to_string()),
            "the unread reason is the build's progress"
        );
    }
}

/// Status is warming until the first reads are ready: with no snapshot and
/// no build it says so, and once the daemon has read its scope index and
/// snapshot it does not.
#[test]
fn status_is_warming_until_the_first_reads_are_ready() {
    let daemon = crate::test_daemon::start("synthetic source");
    daemon.daemon.status_snapshot.clear(&daemon.daemon);
    let warming = answer(&daemon.daemon, None, false);
    assert_eq!(warming.exit_code(), 5);
    assert!(warming.value().unwrap().warming);
    assert!(
        daemon
            .daemon
            .reads
            .reader()
            .map(|reader| daemon
                .daemon
                .status_snapshot
                .refresh(&daemon.daemon, &reader, 0))
            .is_ok()
    );
    assert!(daemon.daemon.scope.is_loaded());
    assert!(!answer(&daemon.daemon, None, false).value().unwrap().warming);
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
