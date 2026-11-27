use super::*;

#[test]
fn a_caught_panic_keeps_its_message() {
    let panic = caught(|| -> u32 { panic!("record 7 has no type") }).unwrap_err();
    assert_eq!(panic.message, "record 7 has no type");
    assert_eq!(caught(|| 5), Ok(5));
}

#[test]
fn the_panic_hook_gives_caught_its_location_and_reports_only_uncaught_panics() {
    let reported = std::sync::Arc::new(std::sync::Mutex::new(Vec::<(String, Panicked)>::new()));
    let sink = std::sync::Arc::clone(&reported);
    install_panic_hook(move |thread, panic, _backtrace| {
        sink.lock()
            .unwrap()
            .push((thread.to_owned(), panic.clone()));
    });
    let line = line!() + 1;
    let panic = caught(|| -> () { panic!("inside") }).unwrap_err();
    assert_eq!(panic.message, "inside");
    assert_eq!(
        panic.location,
        Some(format!("crates/model/src/cause/tests.rs:{line}"))
    );
    assert!(reported.lock().unwrap().is_empty());

    let thread = std::thread::Builder::new()
        .name("polylogue-test-uncaught".into())
        .spawn(|| panic!("outside"))
        .unwrap();
    assert!(thread.join().is_err());
    let reported = reported.lock().unwrap();
    assert_eq!(reported.len(), 1);
    assert_eq!(reported[0].0, "polylogue-test-uncaught");
    assert_eq!(reported[0].1.message, "outside");
    let _ = std::panic::take_hook();
}

#[test]
fn chain_failures_name_content_stored_data_and_bugs() {
    let failure = ChainFailure::from(caught(|| -> () { panic!("bad") }).unwrap_err());
    assert_eq!(failure.kind, ChainFailureKind::ParserPanicked);
    assert_eq!(
        (failure.cause(), failure.code()),
        (Cause::Bug, Code::Internal)
    );
    let mismatch = ChainFailure::from(crate::CountMismatch {
        records: 3,
        counted: 2,
    });
    assert_eq!(mismatch.kind, ChainFailureKind::CountsDiffer);
    assert_eq!(
        (mismatch.cause(), mismatch.code()),
        (Cause::Bug, Code::Internal)
    );
    let bytes = ChainFailure::bytes_differ();
    assert_eq!(
        (bytes.cause(), bytes.code()),
        (Cause::StoredData, Code::Internal)
    );
    let malformed = ChainFailure::malformed("invalid JSON");
    assert_eq!(
        (malformed.cause(), malformed.code()),
        (Cause::Content, Code::Import)
    );
    assert_eq!(
        Panicked {
            message: "m".into(),
            location: None
        }
        .cause(),
        Cause::Bug
    );
}

#[test]
fn io_failures_are_classified_by_kind() {
    use std::io::ErrorKind;
    for (kind, cause) in [
        (ErrorKind::Interrupted, Cause::Transient),
        (ErrorKind::TimedOut, Cause::Transient),
        (ErrorKind::PermissionDenied, Cause::Environment),
        (ErrorKind::StorageFull, Cause::Environment),
        (ErrorKind::NotFound, Cause::Environment),
        (ErrorKind::Other, Cause::Transient),
        (ErrorKind::InvalidData, Cause::Content),
        (ErrorKind::InvalidInput, Cause::Bug),
    ] {
        assert_eq!(Cause::of_io(&kind.into()), cause, "{kind:?}");
    }
}

#[test]
fn a_failure_is_the_same_condition_whatever_its_sentence_says() {
    let first = Failure {
        subject: "/home/op/a.jsonl".into(),
        cause: Cause::Environment,
        code: Code::Io,
        message: "read 10 bytes".into(),
    };
    let again = Failure {
        message: "read 20 bytes".into(),
        ..first.clone()
    };
    assert_eq!(first.key(), again.key());
    assert_eq!(first.to_string(), "/home/op/a.jsonl: read 10 bytes");
}
