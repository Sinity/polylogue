use super::*;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};

static WAITING: Collector = Collector {
    setting: None,
    background: true,
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
