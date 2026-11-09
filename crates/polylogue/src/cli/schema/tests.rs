use super::*;
use crate::commands::Surface;
use serde_json::json;

#[test]
fn infer_and_unread_share_typed_requests_and_keep_local_output_off_remote_surfaces() {
    let arguments = |argv: &[&str]| {
        let (command, _, arguments) =
            crate::cli::request_arguments(argv, std::path::Path::new("/")).unwrap();
        (command, serde_json::Value::Object(arguments))
    };
    let (command, wire) = arguments(&[
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
    ]);
    assert_eq!(
        wire,
        json!({"origin":"claude-code","input":["opaque:input"],"revision":2,"field":["named"]})
    );
    assert!(command.tool == Some(crate::commands::Tool::Read) && command.http);
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
    let daemon = crate::test_daemon::start("schema arguments");
    for args in [
        vec!["polylogue", "schema", "infer", "--revision", "0"],
        vec!["polylogue", "schema", "infer", "--since", "2026-01-01"],
        vec!["polylogue", "schema", "infer", "claude-code", "--all"],
    ] {
        let (_, wire) = arguments(&args);
        let answer = crate::test_daemon::call(&daemon.daemon, "schema infer", wire);
        assert_eq!(
            answer.failure().and_then(|failure| failure.code.as_deref()),
            Some("bad_request"),
            "{args:?}: {answer:?}"
        );
    }
    let (_, wire) = arguments(&["polylogue", "schema", "unread", "claude-code"]);
    assert_eq!(wire, json!({"origin":"claude-code"}));
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
    assert_eq!(
        diff(
            command,
            &values,
            crate::config::OutputConfig::new(false, false, false, None),
            true
        ),
        ExitCode::SUCCESS
    );
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

#[test]
fn artifact_diff_reports_declared_type_changes_in_empty_native_tables() {
    let dir = tempfile::tempdir().unwrap();
    let before = dir.path().join("before.json");
    let after = dir.path().join("after.json");
    let result = dir.path().join("diff.json");
    for (path, declared) in [(&before, "TEXT"), (&after, "INTEGER")] {
        let header = json!({"schema":{"empty":"private SQL body"},"table_metadata":{"empty":{"columns":[{"cid":0,"name":"value","declared_type":declared,"not_null":false,"primary_key":0,"hidden":0}],"without_rowid":false}}});
        let mut observer = polylogue_sources::schema::Observer::default();
        observer
            .read(
                io::Cursor::new(serde_json::to_vec(&header).unwrap()),
                polylogue_sources::schema::Framing::RowDump,
            )
            .unwrap();
        let schema = observer.finish("tool-database".into(), vec![]);
        let artifact = json!({"selection":{"origin":"tool-database","explicit_inputs":true,"revision":null,"since_ms":null,"unknown_time_inputs":0,"missing_revisions":[],"selected_inputs":1,"selected_bytes":1,"read_bytes":1},"origins":{"tool-database":schema}});
        std::fs::write(path, serde_json::to_vec(&artifact).unwrap()).unwrap();
    }
    let (command, values) = crate::cli::parse(&[
        "polylogue",
        "schema",
        "diff",
        before.to_str().unwrap(),
        after.to_str().unwrap(),
        "--output",
        result.to_str().unwrap(),
    ])
    .unwrap();
    assert_eq!(
        diff(
            command,
            &values,
            crate::config::OutputConfig::new(false, false, false, None),
            true
        ),
        ExitCode::SUCCESS
    );
    let bytes = std::fs::read(result).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let origin = &report["origins"]["tool-database"];
    assert!(origin["changes"].as_array().unwrap().is_empty());
    assert_eq!(origin["tables"].as_array().unwrap().len(), 1);
    let table = &origin["tables"][0];
    assert_eq!(table["table"], "empty");
    assert_eq!(
        table["before"]["metadata"][0]["columns"][0]["declared_type"],
        "TEXT"
    );
    assert_eq!(
        table["after"]["metadata"][0]["columns"][0]["declared_type"],
        "INTEGER"
    );
    assert_eq!(
        table["after"]["unobserved_declared_columns"],
        json!(["value"])
    );
    assert!(
        !String::from_utf8(bytes)
            .unwrap()
            .contains("private SQL body")
    );
}
