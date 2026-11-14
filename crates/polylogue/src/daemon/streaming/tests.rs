use super::*;

#[test]
fn a_stream_write_after_a_scope_change_is_refused_even_before_a_reader_binds_the_limit() {
    let daemon = crate::test_daemon::start("native stream activation fence");
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
fn shutdown_cancels_a_stream_read_without_cancelling_its_producing_parent() {
    let mut daemon = crate::test_daemon::start("stream shutdown");
    let parent = polylogue_store::read::limit::ReadCancellation::default();
    let limit = ReadLimit::without_deadline(parent.clone());
    let authorship = polylogue_model::authorship::compute_authorship(
        polylogue_model::authorship::Channel::Cli,
        &Default::default(),
        None,
    );
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
    assert!(!parent.is_cancelled());
    assert!(matches!(
        input.next_blocking().unwrap(),
        super::super::stream_buffer::Next::Failed(_)
    ));
}
