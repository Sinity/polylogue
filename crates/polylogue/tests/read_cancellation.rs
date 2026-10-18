use polylogue::read_cancellation::ReadCancellation;
use serde_json::json;
use std::{
    io::{BufRead, BufReader, Read},
    os::unix::net::UnixListener,
    sync::mpsc,
    thread,
    time::Duration,
};

#[test]
fn cancel_closes_the_socket_of_each_client_request() {
    for operation in ["model", "query", "status"] {
        let root = tempfile::tempdir_in("/tmp").unwrap();
        let socket = root.path().join("socket");
        let listener = UnixListener::bind(&socket).unwrap();
        let cancellation = ReadCancellation::default();
        let task_cancel = cancellation.clone();
        let (sent, received) = mpsc::channel();
        let work = thread::spawn(move || {
            let result = match operation {
                "model" => polylogue::model_read::request_cancellable(
                    &socket,
                    &json!({"command":"model-read","reference":"test:neutral","presentation":"mcp"}),
                    task_cancel,
                ),
                "query" => polylogue::session_query::request_cancellable(&socket, "", task_cancel),
                _ => polylogue::request_value_cancellable(
                    &socket,
                    &json!({"command":"status"}),
                    task_cancel,
                ),
            };
            sent.send(result).unwrap();
        });
        let (mut peer, _) = listener.accept().unwrap();
        peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let mut line = String::new();
        BufReader::new(&peer).read_line(&mut line).unwrap();
        assert!(line.contains("command"));
        cancellation.cancel();
        // The peer never answers and the request's own deadline is 30 s, so
        // any other error comes from the cancellation.
        let error = received.recv().unwrap().unwrap_err();
        assert_ne!(error.kind(), std::io::ErrorKind::TimedOut, "{operation}");
        let mut b = [0];
        assert_eq!(
            peer.read(&mut b).unwrap(),
            0,
            "{operation} kept its read socket open"
        );
        work.join().unwrap();
    }
}

#[test]
fn cancel_stops_a_blocked_connect_before_its_deadline() {
    let root = tempfile::tempdir_in("/tmp").unwrap();
    let socket = root.path().join("socket");
    let listener =
        socket2::Socket::new(socket2::Domain::UNIX, socket2::Type::STREAM, None).unwrap();
    listener
        .bind(&socket2::SockAddr::unix(&socket).unwrap())
        .unwrap();
    listener.listen(0).unwrap();
    let _queued = std::os::unix::net::UnixStream::connect(&socket).unwrap();
    let cancellation = ReadCancellation::default();
    let task_cancel = cancellation.clone();
    let (sent, received) = mpsc::channel();
    let work = thread::spawn(move || {
        sent.send(polylogue::model_read::request_cancellable(
            &socket,
            &json!({"command":"model-read","reference":"test:neutral"}),
            task_cancel,
        ))
        .unwrap()
    });
    cancellation.cancel();
    // The listener's queue stays full, so the connect can end only through the
    // cancellation (ConnectionAborted) or its 30 s deadline (TimedOut).
    assert_eq!(
        received.recv().unwrap().unwrap_err().kind(),
        std::io::ErrorKind::ConnectionAborted
    );
    work.join().unwrap();
}
