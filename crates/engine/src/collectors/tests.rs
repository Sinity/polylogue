use super::*;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};

static WAITING: Collector = Collector {
    setting: None,
    background: true,
    settle: None,
    sync: |paths, _stop| {
        let mut socket = UnixStream::connect(paths.spool).unwrap();
        let mut reply = [0];
        socket.read_exact(&mut reply).unwrap();
        vec!["finished request".into()]
    },
};

#[test]
fn a_waiting_collector_runs_off_the_loop_thread_and_does_not_start_twice() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("collector.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut collectors = Collectors::default();
    let paths = CollectorPaths {
        home: temp.path(),
        state_dir: temp.path(),
        spool: &socket,
    };
    assert!(collectors.run("waiting", &WAITING, &paths).is_empty());
    // A blocking accept waits for the worker while the calling thread is free.
    listener.set_nonblocking(false).unwrap();
    let (mut request, _) = listener.accept().unwrap();
    assert!(collectors.run("waiting", &WAITING, &paths).is_empty());
    assert_eq!(collectors.tasks.len(), 1);
    listener.set_nonblocking(true).unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    request.write_all(&[1]).unwrap();
    let task = collectors.tasks.remove("waiting").unwrap();
    assert_eq!(task.join().unwrap(), ["finished request"]);
}

static STOPPING: Collector = Collector {
    setting: None,
    background: true,
    settle: None,
    sync: |paths, stop| {
        let mut socket = UnixStream::connect(paths.spool).unwrap();
        while !stop.load(Ordering::Relaxed) {
            std::thread::park_timeout(std::time::Duration::from_millis(5));
        }
        socket.write_all(&[1]).unwrap();
        Vec::new()
    },
};

#[test]
fn dropping_collectors_signals_stop_before_joining_the_workers() {
    let temp = tempfile::tempdir().unwrap();
    let socket = temp.path().join("collector.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let mut collectors = Collectors::default();
    assert!(
        collectors
            .run(
                "stopping",
                &STOPPING,
                &CollectorPaths {
                    home: temp.path(),
                    state_dir: temp.path(),
                    spool: &socket,
                }
            )
            .is_empty()
    );
    let (mut request, _) = listener.accept().unwrap();
    drop(collectors);
    let mut stopped = [0];
    request.read_exact(&mut stopped).unwrap();
    assert_eq!(stopped, [1]);
}

static ARCHIVE_ACCEPTANCE: Collector = Collector {
    setting: None,
    background: false,
    sync: |_, _| Vec::new(),
    settle: Some(|paths, accepted| {
        let requested =
            std::fs::read(paths.state_dir.join("expected.json")).map_err(|e| e.to_string())?;
        if accepted(
            &paths.spool.join("1AbC.json"),
            polylogue_model::BlobHash::of(&requested),
            requested.len() as u64,
        )? {
            Ok(())
        } else {
            Err("revision is not archived".into())
        }
    }),
};

#[test]
fn collector_acceptance_requires_the_complete_exact_archived_body() {
    let home = polylogue_testkit::ClaudeHome::new();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64 << 10);
    let spool = engine.config.state_dir().join("drive/ai-studio");
    std::fs::create_dir_all(&spool).unwrap();
    let paths = CollectorPaths {
        home: &home.home,
        state_dir: home.dir.path(),
        spool: &spool,
    };
    let path = spool.join("1AbC.json");
    let small = br#"{"runSettings":{},"chunkedPrompt":{"chunks":[]}}"#;
    let complete = br#"{"runSettings":{},"chunkedPrompt":{"chunks":[{"role":"user","text":"full retained bytes"}]}}"#;
    std::fs::write(paths.state_dir.join("expected.json"), complete).unwrap();
    assert!(
        engine
            .settle_collector("ai-studio", &ARCHIVE_ACCEPTANCE, &paths)
            .is_err()
    );
    std::fs::write(&path, small).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(
        engine
            .settle_collector("ai-studio", &ARCHIVE_ACCEPTANCE, &paths)
            .is_err()
    );
    std::fs::write(&path, complete).unwrap();
    assert!(
        engine
            .settle_collector("ai-studio", &ARCHIVE_ACCEPTANCE, &paths)
            .is_err()
    );
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    engine
        .settle_collector("ai-studio", &ARCHIVE_ACCEPTANCE, &paths)
        .unwrap();
    assert!(
        engine
            .settle_collector("codex", &ARCHIVE_ACCEPTANCE, &paths)
            .is_err()
    );
    let mut wrong = complete.to_vec();
    let at = wrong.iter().position(|byte| *byte == b'f').unwrap();
    wrong[at] = b'F';
    std::fs::write(paths.state_dir.join("expected.json"), &wrong).unwrap();
    assert!(
        engine
            .settle_collector("ai-studio", &ARCHIVE_ACCEPTANCE, &paths)
            .is_err()
    );
    // Missing local publication needs republishing; it is never an acceptance.
    std::fs::remove_file(&path).unwrap();
    assert!(
        engine
            .settle_collector("ai-studio", &ARCHIVE_ACCEPTANCE, &paths)
            .is_err()
    );
}
