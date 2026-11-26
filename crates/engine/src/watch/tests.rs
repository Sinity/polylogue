use super::*;
use std::fs;
use std::time::Duration;

#[test]
fn missing_roots_advance_without_watching_unrelated_subtrees_or_reads() {
    let home = tempfile::tempdir().unwrap();
    let unrelated = home.path().join("unrelated/deep");
    fs::create_dir_all(&unrelated).unwrap();
    let mut config = crate::test_config();
    config.home = home.path().to_path_buf();
    config.tool_dirs.clear();
    let (tx, rx) = crossbeam_channel::unbounded();
    let mut roots = Roots::new(&config, tx).unwrap();
    fs::write(unrelated.join("ignored"), "unrelated").unwrap();
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());

    let spool = config.spool_dir(polylogue_sources::browser_capture::ORIGIN);
    fs::create_dir_all(&spool).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).expect("a wake-up");
    roots.take();
    roots.refresh(&config).unwrap();
    let provider = spool.join("synthetic");
    fs::create_dir_all(&provider).unwrap();
    fs::write(provider.join("capture.json"), "{}").unwrap();
    rx.recv_timeout(Duration::from_secs(10)).expect("a wake-up");
    roots.take();
    // Drain the finite write burst before checking read access.
    while rx.recv_timeout(Duration::from_millis(100)).is_ok() {}
    roots.take();
    fs::read(provider.join("capture.json")).unwrap();
    fs::read_dir(&spool).unwrap().for_each(drop);
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
}

#[test]
fn repeated_paths_are_coalesced_and_watch_failures_request_a_full_scan() {
    use notify::event::{CreateKind, Flag};
    let mut changes = Changes::default();
    let event = notify::Event::new(notify::EventKind::Create(CreateKind::Folder))
        .add_path(PathBuf::from("/synthetic/new"));
    changes.record(Ok(event.clone()));
    changes.record(Ok(event));
    assert_eq!(changes.paths.len(), 1);
    assert!(!changes.full);
    changes.record(Err(notify::Error::generic("synthetic watch failure")));
    assert!(changes.full);
    assert_eq!(changes.errors.len(), 1);
    changes.full = false;
    changes.record(Ok(
        notify::Event::new(notify::EventKind::Other).set_flag(Flag::Rescan)
    ));
    assert!(changes.full);
    changes.full = false;
    changes.record(Ok(notify::Event::new(notify::EventKind::Any)));
    assert!(changes.full);
}

#[test]
fn removing_a_watched_root_marks_it_for_rewatch_but_a_change_below_it_does_not() {
    let home = tempfile::tempdir().unwrap();
    let mut config = crate::test_config();
    config.home = home.path().to_path_buf();
    config.tool_dirs.clear();
    fs::create_dir_all(config.spool_dir(polylogue_sources::browser_capture::ORIGIN)).unwrap();
    let (tx, _rx) = crossbeam_channel::unbounded();
    let mut roots = Roots::new(&config, tx).unwrap();
    let root = roots.active.keys().next().cloned().expect("a watched root");
    roots.take();
    roots.refresh(&config).unwrap();
    assert!(!roots.retry);
    let unrelated = notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::File))
        .add_path(root.join("child"));
    roots.changes.locked().record(Ok(unrelated));
    roots.take();
    assert!(!roots.retry, "a change below a root keeps its watch");
    let removed = notify::Event::new(notify::EventKind::Remove(notify::event::RemoveKind::Folder))
        .add_path(root);
    roots.changes.locked().record(Ok(removed));
    roots.take();
    assert!(roots.retry);
}
