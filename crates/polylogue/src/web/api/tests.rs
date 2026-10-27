use super::*;
use std::io::{Read, Seek, SeekFrom, Write};

#[test]
fn finite_change_output_scrubs_decoded_paths_before_transport_without_changing_completion() {
    let daemon = crate::test_daemon::start("synthetic changes");
    let web = Web {
        daemon: Arc::clone(&daemon.daemon),
        key: "synthetic".into(),
        token: "synthetic".into(),
        port: 8765,
    };
    let root = web.daemon.archive.root.to_string_lossy().into_owned();
    let data = serde_json::json!({"type":"change","component":"index","position":null,"change":{"ref":format!("session:claude-code:{root}"),"seq":1}});
    let mut file = tempfile::tempfile().unwrap();
    for data in [
        data,
        serde_json::json!({"type":"complete","position":"synthetic:1:0","count":1,"replacement":false}),
    ] {
        serde_json::to_writer(&mut file, &crate::daemon::Response::Ok { data }).unwrap();
        file.write_all(b"\n").unwrap();
    }
    file.seek(SeekFrom::Start(0)).unwrap();
    let mut clean = scrub_changes(&web, file).unwrap();
    let mut body = String::new();
    clean.read_to_string(&mut body).unwrap();
    assert!(!body.contains(&root));
    let lines = body
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[1]["data"]["position"], "synthetic:1:0");
    assert_eq!(lines[1]["data"]["count"], 1);
    let value = super::super::scrubbed(
        &web,
        serde_json::json!({"nested":{root:"path key","[archive]":"existing key"}}),
    );
    assert_eq!(value["nested"]["[archive]"], "existing key");
    assert_eq!(value["nested"]["[archive] #2"], "path key");
}
