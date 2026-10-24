use super::*;

#[test]
fn exact_attachment_socket_honors_daemon_read_admission_stop() {
    let world = test_daemon::start("synthetic read admission");
    world
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
            asset: None,
            reveal: true,
            length: 1,
            cursor: None,
        },
        &world.daemon,
    );
    assert!(matches!(response,Response::Error { kind,.. } if kind=="daemon_stopping"));
}
