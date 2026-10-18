use std::process::Command;

#[test]
fn help_version_and_schema_work_offline_without_creating_an_archive() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("must-not-exist");
    for args in [
        vec!["--help"],
        vec!["status", "--help"],
        vec!["import", "-h"],
        vec!["--version"],
        vec!["config", "schema"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_polylogue"))
            .args(["--archive", archive.to_str().unwrap()])
            .args(&args)
            .env_remove("XDG_RUNTIME_DIR")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        if args.contains(&"--help") || args.contains(&"-h") {
            assert!(String::from_utf8_lossy(&output.stdout).starts_with("Polylogue:"));
        }
        if args == ["config", "schema"] {
            let schema: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(schema["additionalProperties"], false);
            assert!(schema["properties"]["archive"].is_object());
        }
        assert!(!archive.exists());
    }
}
#[test]
fn unsupported_commands_and_bad_arguments_fail_with_exit_two() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("not-created");
    for args in [
        vec!["search", "anything"],
        vec!["mcp", "--unknown"],
        vec!["daemon", "--until-idle"],
        vec!["--unknown"],
        vec!["config", "set", "secret", "value"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_polylogue"))
            .args(["--archive", archive.to_str().unwrap()])
            .args(args)
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(!archive.exists());
    }
    assert_eq!(
        Command::new(env!("CARGO_BIN_EXE_polylogue"))
            .arg("--archive")
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
    assert_eq!(
        Command::new(env!("CARGO_BIN_EXE_polylogue"))
            .args(["--archive", "", "daemon"])
            .output()
            .unwrap()
            .status
            .code(),
        Some(2)
    );
}

#[test]
fn daemon_restarts_on_the_same_archive() {
    use std::process::{Child, Stdio};
    use std::time::Duration;
    struct OwnedChild(Child);
    impl Drop for OwnedChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    // Unix socket paths are limited to 108 bytes, so only the socket goes in /tmp.
    let runtime_owner = tempfile::tempdir_in("/tmp").unwrap();
    let runtime = runtime_owner.path().join("runtime");
    std::fs::create_dir(&runtime).unwrap();
    let binary = env!("CARGO_BIN_EXE_polylogue");
    let start = || {
        OwnedChild(
            Command::new(binary)
                .args(["--archive", archive.to_str().unwrap(), "daemon"])
                .env("XDG_RUNTIME_DIR", &runtime)
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        )
    };
    let mut first = start();
    let wait_status = |child: &mut OwnedChild| {
        loop {
            let output = Command::new(binary)
                .args(["--archive", archive.to_str().unwrap(), "status"])
                .env("XDG_RUNTIME_DIR", &runtime)
                .output()
                .unwrap();
            if output.status.success() {
                let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(value["archive"], archive.to_str().unwrap());
                assert!(value["store"].is_object(), "{value}");
                break;
            }
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "daemon exited before status: {output:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    wait_status(&mut first);
    let invalid = Command::new(binary)
        .args(["--archive", archive.to_str().unwrap(), "progress", ""])
        .env("XDG_RUNTIME_DIR", &runtime)
        .output()
        .unwrap();
    let reply: serde_json::Value = serde_json::from_slice(&invalid.stdout).unwrap();
    assert_eq!(reply["outcome"], "error");
    assert_eq!(reply["error"]["kind"], "invalid_query");
    assert_eq!(invalid.status.code(), Some(2), "{invalid:?}");
    first.0.kill().unwrap();
    first.0.wait().unwrap();
    let mut second = start();
    wait_status(&mut second);
}

#[test]
fn mcp_lists_its_tools_without_a_daemon_or_archive() {
    use serde_json::{Value, json};
    use std::{io::Write, process::Stdio};
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("not-created");
    let mut child = Command::new(env!("CARGO_BIN_EXE_polylogue"))
        .args(["--archive", archive.to_str().unwrap(), "mcp"])
        .env_remove("XDG_RUNTIME_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = child.stdin.take().unwrap();
    let meta = json!({"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{}});
    for (id, method) in [(1, "server/discover"), (2, "tools/list")] {
        writeln!(
            input,
            "{}",
            json!({"jsonrpc":"2.0","id":id,"method":method,"params":{"_meta":meta}})
        )
        .unwrap();
    }
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let responses: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let discovery = responses.iter().find(|r| r["id"] == 1).unwrap();
    assert!(
        discovery["result"]["supportedVersions"]
            .as_array()
            .unwrap()
            .contains(&json!("2026-07-28"))
    );
    let listing = responses.iter().find(|r| r["id"] == 2).unwrap();
    assert_eq!(listing["result"]["resultType"], "complete");
    let tools = listing["result"]["tools"].as_array().unwrap();
    assert_eq!(
        tools
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["status", "search", "read", "note"]
    );
    for tool in tools {
        assert_eq!(tool["outputSchema"]["required"], json!(["outcome"]));
        assert_eq!(tool["annotations"]["readOnlyHint"], tool["name"] != "note");
        assert_eq!(
            tool["annotations"]["idempotentHint"],
            tool["name"] != "note"
        );
        assert_eq!(tool["inputSchema"]["additionalProperties"], false);
    }
    let note = &tools[3];
    assert_eq!(note["inputSchema"]["required"], json!(["value"]));
    assert_eq!(
        note["inputSchema"]["properties"]["kind"]["enum"],
        json!(["note", "correction", "withdrawal"])
    );
    assert!(note["inputSchema"]["properties"].get("author").is_none());
    assert!(!archive.exists());
}
