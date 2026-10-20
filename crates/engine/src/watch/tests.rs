use super::*;
use std::fs;
use std::sync::mpsc::channel;
use std::time::Duration;

#[test]
fn missing_roots_advance_without_watching_unrelated_subtrees_or_reads() {
    let home = tempfile::tempdir().unwrap();
    let unrelated = home.path().join("unrelated/deep");
    fs::create_dir_all(&unrelated).unwrap();
    let mut config = Config::from_env();
    config.home = home.path().to_path_buf();
    config.tool_dirs.clear();
    let (tx, rx) = channel();
    let mut roots = Roots::new(&config, tx).unwrap();
    fs::write(unrelated.join("ignored"), "unrelated").unwrap();
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());

    let spool = config.spool_dir(polylogue_sources::browser_capture::ORIGIN);
    fs::create_dir_all(&spool).unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    roots.refresh(&config).unwrap();
    let provider = spool.join("synthetic");
    fs::create_dir_all(&provider).unwrap();
    fs::write(provider.join("capture.json"), "{}").unwrap();
    rx.recv_timeout(Duration::from_secs(2)).unwrap();
    // Drain the finite write burst before checking read access.
    while rx.recv_timeout(Duration::from_millis(100)).is_ok() {}
    fs::read(provider.join("capture.json")).unwrap();
    fs::read_dir(&spool).unwrap().for_each(drop);
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
}
