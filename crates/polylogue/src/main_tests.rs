use super::*;

#[test]
fn read_transport_timeouts_exit_seven_with_a_suggestion() {
    for kind in [io::ErrorKind::TimedOut, io::ErrorKind::WouldBlock] {
        let failure = model_read_io_error(io::Error::new(kind, "synthetic deadline"));
        assert_eq!(failure.kind, ErrorKind::Timeout);
        assert_eq!(failure.outcome().exit_code(), 7);
        assert!(failure.suggestion.is_some());
        assert!(!failure.retryable);
        assert!(failure.message.contains("smaller budget"));
    }
    assert_eq!(
        model_read_io_error(io::Error::new(
            io::ErrorKind::InvalidData,
            "synthetic frame"
        ))
        .kind,
        ErrorKind::Io
    );
}
#[test]
fn daemon_replies_map_to_the_documented_exit_codes() {
    for kind in [
        ErrorKind::InvalidQuery,
        ErrorKind::NotYetAvailable,
        ErrorKind::NotFound,
        ErrorKind::Ambiguous,
        ErrorKind::Conflict,
        ErrorKind::Refused {
            reason: "unsupported".into(),
        },
        ErrorKind::Timeout,
        ErrorKind::NotRunning,
        ErrorKind::Io,
    ] {
        let expected = kind.exit_code();
        let reply = json!({"outcome":"error","error":Failure::new(kind,"failed")});
        assert_eq!(reply_exit_code(&reply).unwrap(), expected);
    }
    for (reply, code) in [
        (json!({"outcome":"ok"}), 0),
        (json!({"outcome":"empty"}), 1),
        (json!({"outcome":"degraded","missing":["input"]}), 5),
    ] {
        assert_eq!(reply_exit_code(&reply).unwrap(), code);
    }
    for reply in [
        json!({}),
        json!({"outcome":"error"}),
        json!({"outcome":"unknown"}),
    ] {
        assert_eq!(reply_exit_code(&reply).unwrap_err().kind, ErrorKind::Io);
    }
}
