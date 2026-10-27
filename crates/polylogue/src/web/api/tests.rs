use super::*;
use std::io::Read;
use std::os::unix::fs::MetadataExt;

#[test]
fn finite_batches_are_scrubbed_once_and_reopened_with_independent_positions() {
    let daemon = crate::test_daemon::start("synthetic changes");
    let web = Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    };
    let mut first = crate::daemon::service::change_feed(&web.daemon, None, true).unwrap();
    let mut second = crate::daemon::service::change_feed(&web.daemon, None, true).unwrap();
    assert_eq!(
        first.metadata().unwrap().ino(),
        second.metadata().unwrap().ino()
    );
    let mut a = String::new();
    let mut b = String::new();
    first.read_to_string(&mut a).unwrap();
    second.read_to_string(&mut b).unwrap();
    assert_eq!(a, b);
    assert!(!a.contains(&web.daemon.archive.root.to_string_lossy().to_string()));
    let lines = a
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(lines.last().unwrap()["data"]["type"], "complete");
    let root = web.daemon.archive.root.to_string_lossy().into_owned();
    let value = super::super::scrubbed(
        &web,
        serde_json::json!({"nested":{root:"path key","[archive]":"existing key"}}),
    );
    assert_eq!(value["nested"]["[archive]"], "existing key");
    assert_eq!(value["nested"]["[archive] #2"], "path key");
}
