use super::*;
#[test]
fn an_unpublished_old_generation_is_rebuilt_before_the_next_activation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let fixture =
        polylogue_store::fixture::Fixture::new(&directory.path().join("archive")).unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let current = reader.change_position().unwrap();
    let pointer = completion::CatalogPointer::new(path.clone());
    let old = Metadata {
        archive: "synthetic".into(),
        position: ChangePosition {
            generation: "retired-generation".into(),
            index: 0,
            archive: 0,
        },
        written_at: 1,
    };
    let writer = pointer
        .build(&old, |writer| {
            writer.put_catalog(&super::super::base_catalog())?;
            writer.put_values(
                "old",
                &BTreeMap::from([(
                    "repo".into(),
                    BTreeMap::from([("old-scope-value".into(), 1)]),
                )]),
            )
        })
        .unwrap();
    let mut state = State::new(false, super::super::tests::request_metadata());
    // A build may finish before its activation is rejected by the scope fence.
    state.writer = Some((old.position.generation, writer));
    assert!(state.position.is_none());
    state.refresh(&reader, &path, "synthetic").unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.position, current);
    assert!(cache.values("repo", "").unwrap().is_empty());
}

#[test]
fn publication_uses_the_captured_clock_and_machine_scoped_setting() {
    use polylogue_model::{
        Author, WallClock,
        authorship::{Authorship, Channel},
        user::{RequestMetadata, UserWrite},
    };
    let archive = polylogue_store::fixture::writer::WriterArchive::new();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("catalog");
    let instant: jiff::Timestamp = "2026-02-03T04:05:06Z".parse().unwrap();
    let requests = RequestMetadata::new("captured-machine".into(), WallClock::fixed(instant));
    for (id, machine, value) in [
        ("selected", "captured-machine", "3"),
        ("other", "different-machine", "9"),
    ] {
        archive
            .writer
            .user(requests.request(
                id.into(),
                UserWrite::SetSetting {
                    key: "signal.interrupts_per_hour".into(),
                    value: value.into(),
                    machine: Some(machine.into()),
                },
                Authorship {
                    author: Author::Operator,
                    via: Channel::Cli,
                },
            ))
            .unwrap();
    }
    let reads = polylogue_store::ReadPool::open(&archive.archive).unwrap();
    let mut state = State::new(false, requests);
    state
        .refresh(&reads.reader().unwrap(), &path, "synthetic")
        .unwrap();
    let cache = completion::CatalogReader::open(&path, "synthetic")
        .unwrap()
        .unwrap();
    assert_eq!(cache.metadata.written_at, instant.as_millisecond());
    let settings = cache.settings("signal.interrupts_per_hour").unwrap();
    let [setting] = &settings[..] else {
        panic!("one setting")
    };
    let current = &setting.value;
    assert_eq!(
        current.value,
        Some(polylogue_model::settings::Value::InterruptBudget(3))
    );
    assert!(current.conflicts.is_empty());
}

#[test]
fn explicit_shutdown_reports_a_panicked_publisher_once() {
    let mut publisher = Publisher {
        stop: Arc::new(AtomicBool::new(false)),
        expected_exit: Arc::new(AtomicBool::new(false)),
        wake: None,
        thread: Some(std::thread::spawn(|| panic!("synthetic publisher failure"))),
    };
    let error = publisher.shutdown().unwrap_err();
    assert_eq!(error.to_string(), "completion worker panicked");
    assert!(publisher.thread.is_none());
    publisher.shutdown().unwrap();
}

#[test]
fn explicit_shutdown_accepts_only_a_recorded_expected_return() {
    for (expected, succeeds) in [(true, true), (false, false)] {
        let mut publisher = Publisher {
            stop: Arc::new(AtomicBool::new(false)),
            expected_exit: Arc::new(AtomicBool::new(expected)),
            wake: None,
            thread: Some(std::thread::spawn(|| {})),
        };
        assert_eq!(publisher.shutdown().is_ok(), succeeds);
        assert!(publisher.thread.is_none());
    }
}
