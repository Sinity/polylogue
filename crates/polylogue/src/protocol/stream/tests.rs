use super::*;
#[test]
fn data_activation_is_scope_fenced_even_before_a_reader_binds_the_limit() {
    let daemon = crate::test_daemon::start("native stream activation fence");
    let scope = daemon.daemon.reads.scope_token().unwrap();
    let limit = polylogue_store::read::limit::ReadLimit::without_deadline(Default::default());
    assert!(limit.check().is_ok());
    daemon.daemon.archive.advance_generation_for_test().unwrap();
    let mut bytes = Vec::new();
    let mut writer = GuardedWriter::new(&mut bytes, limit, scope);
    assert!(writer.write_all(b"stale complete event").is_err());
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
    assert!(bytes.is_empty());
}
