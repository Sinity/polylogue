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
