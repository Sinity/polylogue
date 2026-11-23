use super::*;

#[test]
fn a_panicking_stream_answers_internal_before_or_after_its_first_bytes() {
    let daemon = crate::test_daemon::start("stream panic fixture");
    for prefix in [false, true] {
        let invocation =
            daemon
                .daemon
                .invocation(&polylogue_reading::operation::Context::worker_authorship(
                    "stream-panic",
                ));
        let input = start(
            &daemon.daemon,
            Class::Ordinary,
            ReadLimit::without_deadline(Default::default()),
            invocation,
            Box::new(move |_, _, _, output| {
                if prefix {
                    output.write_all(b"synthetic-prefix\n").unwrap();
                }
                panic!("synthetic stream panic");
            }),
        )
        .unwrap();
        let mut bytes = Vec::new();
        loop {
            match input.next_blocking().unwrap() {
                super::super::stream_buffer::Next::Bytes(next) => bytes.extend(next),
                super::super::stream_buffer::Next::Failed(failure) => {
                    assert!(!prefix);
                    assert_eq!(failure.code(), "internal");
                    assert!(failure.message.contains("synthetic stream panic"));
                    break;
                }
                super::super::stream_buffer::Next::End => {
                    assert!(prefix);
                    let text = String::from_utf8(bytes).unwrap();
                    assert!(text.starts_with("synthetic-prefix\n"));
                    assert!(text.contains("internal"), "{text}");
                    assert!(text.contains("synthetic stream panic"), "{text}");
                    break;
                }
            }
        }
    }
}

#[test]
fn a_stream_write_after_a_scope_change_is_refused_even_before_a_reader_binds_the_limit() {
    let daemon = crate::test_daemon::start("stream activation fence");
    let scope = daemon.daemon.reads.scope_token().unwrap();
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
    assert!(limit.check().is_ok());
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let (mut output, input) = crate::daemon::stream_buffer::buffer(directory.path().to_owned());
    let mut writer = GuardedWriter::new(&mut output, limit, scope);
    assert!(std::io::Write::write_all(&mut writer, b"stale complete event").is_err());
    let failure = writer
        .failure()
        .expect("typed scope failure")
        .answer::<serde_json::Value>();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = failure
    else {
        panic!("scope changed failure");
    };
    assert_eq!(error.code.as_deref(), Some("scope_changed"));
    drop(writer);
    output.finish(Ok(()));
    assert!(matches!(
        input.next_blocking().unwrap(),
        crate::daemon::stream_buffer::Next::End
    ));
}

#[test]
fn shutdown_cancels_a_stream_read_and_keeps_its_producers_writes_admissible() {
    let mut daemon = crate::test_daemon::start("stream shutdown");
    let parent = daemon.daemon.archive.stop().child();
    let limit = ReadLimit::without_deadline(parent.clone());
    let authorship: polylogue_model::authorship::Authorship = command_surface::author_of(
        &command_surface::CallerEvidence::default(),
        command_surface::Via::Cli,
        None,
    )
    .into();
    let invocation = daemon.daemon.invocation(&authorship);
    let (entered, ready) = std::sync::mpsc::channel();
    let (cancelled, stopped) = std::sync::mpsc::channel();
    let input = start(
        &daemon.daemon,
        Class::Ordinary,
        limit,
        invocation,
        Box::new(move |_, _, _, _| {
            let limit = ReadLimit::current().unwrap();
            let (wake, wait) = std::sync::mpsc::channel();
            let _wake = limit.cancellation.on_cancel(move || {
                let _ = wake.send(());
            });
            entered.send(()).unwrap();
            wait.recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            cancelled.send(()).unwrap();
            limit.check().map_err(Failure::from)
        }),
    )
    .unwrap();
    ready
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    daemon.stop().unwrap();
    stopped
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    // A write its producer already handed over is still accepted.
    assert!(
        polylogue_reading::operation::ParentLifetime::capture(None, parent)
            .check()
            .is_ok()
    );
    let super::super::stream_buffer::Next::Failed(failure) = input.next_blocking().unwrap() else {
        panic!("the stopped read must fail");
    };
    assert_eq!(failure.code(), "shutting_down");
}
