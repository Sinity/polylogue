use super::*;
use crate::commands::{Action, Caller, Surface};
use serde_json::json;

#[test]
fn infer_and_unread_share_typed_requests_and_keep_local_output_off_remote_surfaces() {
    let (command, values) = crate::cli::parse(&[
        "polylogue",
        "schema",
        "infer",
        "claude-code",
        "--input",
        "opaque:input",
        "--revision",
        "2",
        "--field",
        "named",
        "--output",
        "schema.json",
    ])
    .unwrap();
    let Action::Request(build) = command.action else {
        panic!("inference needs the daemon");
    };
    let caller = Caller {
        surface: Surface::Cli,
        cwd: None,
        json: true,
    };
    let request = build(command, &values, &caller).unwrap();
    let wire = serde_json::to_value(&request).unwrap();
    assert_eq!(
        wire,
        json!({"op":"schema_infer","origin":"claude-code","inputs":["opaque:input"],"revision":2,"since":null,"fields":["named"]})
    );
    assert!(command.mcp && command.http);
    assert!(
        !command
            .args_for(Surface::Mcp)
            .any(|arg| arg.name == "output")
    );
    assert!(
        !command
            .args_for(Surface::Http)
            .any(|arg| arg.name == "output")
    );
    for args in [
        vec!["polylogue", "schema", "infer", "--revision", "0"],
        vec!["polylogue", "schema", "infer", "--since", "2026-01-01"],
        vec!["polylogue", "schema", "infer", "claude-code", "--all"],
    ] {
        let (command, values) = crate::cli::parse(&args).unwrap();
        let Action::Request(build) = command.action else {
            unreachable!()
        };
        assert!(build(command, &values, &caller).is_err());
    }
    let (command, values) =
        crate::cli::parse(&["polylogue", "schema", "unread", "claude-code"]).unwrap();
    let Action::Request(build) = command.action else {
        unreachable!()
    };
    assert_eq!(
        serde_json::to_value(build(command, &values, &caller).unwrap()).unwrap(),
        json!({"op":"schema_unread","origin":"claude-code"})
    );
}

#[test]
fn explicit_local_artifacts_are_reusable_for_diff_without_archive_access() {
    let dir = tempfile::tempdir().unwrap();
    let before = dir.path().join("before.json");
    let after = dir.path().join("after.json");
    let diff_output = dir.path().join("diff.json");
    let infer = |bytes: &[u8]| {
        let mut observer = polylogue_sources::schema::Observer::default();
        observer
            .read(
                io::Cursor::new(bytes),
                polylogue_sources::schema::Framing::Lines,
            )
            .unwrap();
        let schema = observer.finish("synthetic".into(), vec![]);
        serde_json::json!({"selection":{"origin":"synthetic","explicit_inputs":true,"revision":null,"since_ms":null,"unknown_time_inputs":0,"missing_revisions":[],"selected_inputs":1,"selected_bytes":bytes.len(),"read_bytes":bytes.len()},"origins":{"synthetic":schema}})
    };
    std::fs::write(&before, serde_json::to_vec(&infer(b"{\"a\":1}\n")).unwrap()).unwrap();
    std::fs::write(
        &after,
        serde_json::to_vec(&infer(b"{\"a\":null,\"b\":true}\n")).unwrap(),
    )
    .unwrap();
    let (command, values) = crate::cli::parse(&[
        "polylogue",
        "schema",
        "diff",
        before.to_str().unwrap(),
        after.to_str().unwrap(),
        "--output",
        diff_output.to_str().unwrap(),
    ])
    .unwrap();
    assert_eq!(diff(command, &values, true), ExitCode::SUCCESS);
    let diff: serde_json::Value =
        serde_json::from_slice(&std::fs::read(diff_output).unwrap()).unwrap();
    assert_eq!(diff["comparable"], false);
    assert!(
        diff["reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r == "different_selection_coverage")
    );
    let changes = diff["origins"]["synthetic"]["changes"].as_array().unwrap();
    assert!(
        changes
            .iter()
            .any(|c| c["path"] == "/a" && c["kind"] == "types")
    );
    assert!(
        changes
            .iter()
            .any(|c| c["path"] == "/b" && c["kind"] == "added")
    );
}
