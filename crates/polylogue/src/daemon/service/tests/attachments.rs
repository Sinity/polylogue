use super::*;

#[test]
fn exact_attachment_socket_refuses_once_the_daemon_stops_accepting() {
    let daemon = test_daemon::start("synthetic stopped reads");
    daemon
        .daemon
        .stopping
        .store(true, std::sync::atomic::Ordering::Release);
    let (_client, server) = std::os::unix::net::UnixStream::pair().unwrap();
    let response = crate::daemon::attachment::socket(
        &server,
        Request::Attachment {
            message: polylogue_model::MessageId::parse("claude-code:synthetic").unwrap(),
            block: 0,
            attachment: 0,
            input: None,
            reveal: true,
            length: 1,
            cursor: None,
        },
        &daemon.daemon,
    );
    assert!(matches!(response,Response::Error { kind,.. } if kind=="daemon_stopping"));
}
