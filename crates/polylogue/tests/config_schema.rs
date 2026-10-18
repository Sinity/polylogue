use serde_json::{Value, json};
use std::{
    fs,
    path::Path,
    process::{Child, Command, Output, Stdio},
    time::Duration,
};

const BINARY: &str = env!("CARGO_BIN_EXE_polylogue");

fn startup(config: &Value, root: &Path) -> Output {
    let file = root.join("source.json");
    fs::write(&file, serde_json::to_vec(config).unwrap()).unwrap();
    let output = Command::new(BINARY)
        .args(["--archive", root.join("archive").to_str().unwrap()])
        .args(["daemon", "--config", file.to_str().unwrap(), "--until-idle"])
        .env("XDG_RUNTIME_DIR", root)
        .output()
        .unwrap();
    assert!(!root.join("archive").exists(), "{output:?}");
    output
}

#[test]
fn config_schema_matches_what_daemon_config_accepts() {
    let root = tempfile::tempdir().unwrap();
    let output = Command::new(BINARY)
        .args(["config", "schema"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let schema: Value = serde_json::from_slice(&output.stdout).unwrap();
    let startup_schema = &schema["$defs"]["observedStartup"];
    assert_eq!(startup_schema["additionalProperties"], false);
    assert_eq!(
        startup_schema["oneOf"],
        json!([
            {"required":["claude_main"],"properties":{"claude_main":{"type":"object"}}},
            {"required":["claude_projects"],"properties":{"claude_projects":{"type":"object"}}},
            {"required":["codex_home"],"properties":{"codex_home":{"type":"object"}}}
        ])
    );
    let properties = &startup_schema["properties"];
    for name in ["claude_main", "claude_projects", "codex_home"] {
        assert_eq!(properties[name]["type"], json!(["object", "null"]));
    }
    let projects = &properties["claude_projects"];
    assert_eq!(projects["required"], json!(["root", "machine", "host"]));
    assert_eq!(projects["additionalProperties"], false);
    assert_eq!(projects["properties"]["root"]["pattern"], "^/");
    for name in ["machine", "host"] {
        assert_eq!(
            projects["properties"][name],
            json!({"type":"string","minLength":1})
        );
    }
    for source in ["claude_main", "claude_projects"] {
        assert_eq!(
            properties[source]["properties"]["acquisition_quantum_bytes"],
            json!({"type":"integer","minimum":65536,"maximum":8388608,"default":8388608})
        );
    }

    let valid = json!({"root":root.path().join("projects"),"machine":"synthetic","host":"test"});
    // The file parses and validates, then --until-idle is refused, before any
    // archive is opened.
    for quantum in [None, Some(json!(65536)), Some(json!(8388608))] {
        let mut source = valid.clone();
        if let Some(quantum) = quantum {
            source["acquisition_quantum_bytes"] = quantum;
        }
        let output = startup(
            &json!({"claude_projects":source,"claude_main":null,"codex_home":null}),
            root.path(),
        );
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("--until-idle is not supported for claude_projects"),
            "{output:?}"
        );
    }
    for (field, value) in [
        ("root", json!("relative")),
        ("machine", json!("")),
        ("host", json!("")),
    ] {
        let mut source = valid.clone();
        source[field] = value;
        let output = startup(&json!({"claude_projects":source}), root.path());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("absolute Claude projects root"),
            "{output:?}"
        );
    }
    for source in ["claude_projects", "claude_main"] {
        for quantum in [
            json!(0),
            json!(65535),
            json!(8388609),
            json!(-1),
            json!(1.5),
            json!("65536"),
            Value::Null,
        ] {
            let mut value = if source == "claude_projects" {
                valid.clone()
            } else {
                json!({"path":root.path().join("main.jsonl"),"machine":"synthetic","host":"test"})
            };
            value["acquisition_quantum_bytes"] = quantum;
            let output = startup(&json!({source:value}), root.path());
            assert!(!output.status.success(), "{output:?}");
            let error = String::from_utf8_lossy(&output.stderr);
            assert!(
                error.contains("acquisition quantum")
                    || error.contains("acquisition_quantum_bytes")
                    || error.contains("configuration file"),
                "{output:?}"
            );
        }
    }
    let main = json!({"path":root.path().join("main.jsonl"),"machine":"synthetic","host":"test"});
    let codex = json!({"home":root.path().join("codex"),"machine":"synthetic","host":"test"});
    for config in [
        json!({}),
        json!({"claude_projects":null}),
        json!({"claude_projects":valid,"claude_main":main}),
        json!({"claude_projects":valid,"codex_home":codex}),
        json!({"claude_main":main,"codex_home":codex}),
    ] {
        let output = startup(&config, root.path());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("configure exactly one"),
            "{output:?}"
        );
    }
    for config in [
        json!({"claude_projects":valid,"unknown":true}),
        json!({"claude_projects":{"root":"/synthetic","machine":"synthetic","host":"test","unknown":true}}),
    ] {
        let output = startup(&config, root.path());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("unknown field"),
            "{output:?}"
        );
    }
}

struct OwnedDaemon(Child);
impl Drop for OwnedDaemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn chunk_size_defaults_and_limits_match_daemon_status() {
    for source in ["claude_projects", "claude_main"] {
        for quantum in [None, Some(65536), Some(8388608)] {
            let root = tempfile::tempdir().unwrap();
            let runtime = tempfile::tempdir_in("/tmp").unwrap();
            let provider = root.path().join("provider");
            fs::create_dir(&provider).unwrap();
            let main = provider.join("schema-main.jsonl");
            fs::write(&main, b"{\"type\":\"user\",\"uuid\":\"schema-u\",\"sessionId\":\"schema-main\",\"message\":{\"role\":\"user\",\"content\":\"Neutral schema fixture.\"}}\n").unwrap();
            let mut value = json!({"machine":"synthetic-schema","host":"test"});
            value[if source == "claude_projects" {
                "root"
            } else {
                "path"
            }] = json!(if source == "claude_projects" {
                &provider
            } else {
                &main
            });
            if let Some(quantum) = quantum {
                value["acquisition_quantum_bytes"] = json!(quantum);
            }
            let config = root.path().join("source.json");
            fs::write(&config, serde_json::to_vec(&json!({source:value})).unwrap()).unwrap();
            let archive = root.path().join("archive");
            let mut daemon = OwnedDaemon(
                Command::new(BINARY)
                    .args([
                        "--archive",
                        archive.to_str().unwrap(),
                        "daemon",
                        "--config",
                        config.to_str().unwrap(),
                    ])
                    .env("XDG_RUNTIME_DIR", runtime.path())
                    .stdout(Stdio::null())
                    .stderr(Stdio::piped())
                    .spawn()
                    .unwrap(),
            );
            loop {
                let output = Command::new(BINARY)
                    .args(["--archive", archive.to_str().unwrap(), "status"])
                    .env("XDG_RUNTIME_DIR", runtime.path())
                    .output()
                    .unwrap();
                if output.status.success() {
                    let status: Value = serde_json::from_slice(&output.stdout).unwrap();
                    assert_eq!(
                        status["claude_continuation"]["quantum_bytes"],
                        quantum.unwrap_or(8388608)
                    );
                    break;
                }
                assert!(
                    daemon.0.try_wait().unwrap().is_none(),
                    "daemon exited: {output:?}"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
            assert_eq!(unsafe { libc::kill(daemon.0.id() as i32, libc::SIGINT) }, 0);
            assert!(daemon.0.wait().unwrap().success());
        }
    }
}
