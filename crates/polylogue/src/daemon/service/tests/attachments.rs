use super::*;

#[test]
fn attachment_read_refuses_once_the_daemon_stops_accepting() {
    let daemon = test_daemon::start("synthetic stopped reads");
    daemon
        .daemon
        .stopping
        .store(true, std::sync::atomic::Ordering::Release);
    let response = crate::daemon::attachment::read(
        crate::daemon::attachment::Range {
            message: polylogue_model::MessageId::parse("claude-code:synthetic").unwrap(),
            block: 0,
            attachment: 0,
            input: None,
            length: 1,
            cursor: None,
        },
        &daemon.daemon,
    );
    assert!(
        matches!(response,command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("shutting_down"))
    );
}

#[test]
fn attachment_pages_bound_each_transfer_and_keep_every_requested_byte_reachable() {
    use base64::Engine as _;
    let daemon = test_daemon::start("synthetic bounded attachment");
    let bytes = vec![b'Q'; crate::daemon::attachment::PAGE_BYTES + 17];
    let message = test_daemon::assets::seed_bytes(&daemon, &bytes);
    let mut cursor = None;
    let mut actual = Vec::new();
    loop {
        let response = crate::daemon::attachment::read(
            crate::daemon::attachment::Range {
                message: polylogue_model::MessageId::parse(message).unwrap(),
                block: 0,
                attachment: 0,
                input: None,
                length: usize::MAX,
                cursor: cursor.clone(),
            },
            &daemon.daemon,
        );
        let page = crate::test_answers::ok(response);
        assert_eq!(page.offset, actual.len() as u64);
        let part = base64::engine::general_purpose::STANDARD
            .decode(page.content_base64)
            .unwrap();
        assert!(part.len() <= crate::daemon::attachment::PAGE_BYTES);
        actual.extend(part);
        cursor = page.next;
        if cursor.is_none() {
            break;
        }
    }
    assert_eq!(actual, bytes);
}
