use super::super::derive_thread::Unreported;

#[test]
fn repeated_background_failures_are_kept_as_the_latest_and_a_count() {
    let mut unreported = Unreported::default();
    let mut errors = vec!["earlier".to_owned()];
    unreported.drain_into(&mut errors);
    assert_eq!(errors, ["earlier"]);

    unreported.push("derive: first".into());
    unreported.drain_into(&mut errors);
    assert_eq!(errors, ["earlier", "derive: first"]);

    for n in 0..1000 {
        unreported.push(format!("derive: failure {n}"));
    }
    let mut errors = Vec::new();
    unreported.drain_into(&mut errors);
    assert_eq!(errors, ["derive: failure 999 (and 999 earlier failures)"]);

    unreported.drain_into(&mut errors);
    assert_eq!(errors.len(), 1);
}
