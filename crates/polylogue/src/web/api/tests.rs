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
    let mut first = crate::daemon::service::change_feed(
        &crate::daemon::Request::Changes { since: None },
        &web.daemon,
        Some(super::super::access::scrub),
    )
    .unwrap()
    .unwrap();
    let mut second = crate::daemon::service::change_feed(
        &crate::daemon::Request::Changes { since: None },
        &web.daemon,
        Some(super::super::access::scrub),
    )
    .unwrap()
    .unwrap();
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

#[test]
fn different_feed_positions_prepare_without_holding_the_cache_lock() {
    use std::sync::mpsc;
    let daemon = crate::test_daemon::start("synthetic feed");
    let d = Arc::clone(&daemon.daemon);
    let (entered, reached) = mpsc::channel();
    let (release, resume) = mpsc::channel();
    static PAUSE: std::sync::Mutex<Option<(mpsc::Sender<()>, mpsc::Receiver<()>)>> =
        std::sync::Mutex::new(None);
    fn paused(text: &str, _: &std::path::Path) -> String {
        if let Some((entered, resume)) = PAUSE.lock().unwrap().take() {
            entered.send(()).unwrap();
            resume.recv().unwrap();
        }
        text.to_owned()
    }
    *PAUSE.lock().unwrap() = Some((entered, resume));
    let request = crate::daemon::Request::Changes { since: None };
    let first = std::thread::spawn(move || {
        crate::daemon::service::change_feed(&request, &d, Some(paused))
            .unwrap()
            .unwrap()
    });
    reached.recv().unwrap();
    let second = crate::daemon::service::change_feed(
        &crate::daemon::Request::Changes { since: None },
        &daemon.daemon,
        None,
    )
    .unwrap()
    .unwrap();
    assert!(second.metadata().unwrap().len() > 0);
    release.send(()).unwrap();
    first.join().unwrap();
    assert!(
        std::fs::read_dir(daemon.daemon.archive.staging_dir())
            .unwrap()
            .count()
            >= 1
    );
}
