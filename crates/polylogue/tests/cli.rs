use std::process::Command;

#[test]
fn packaged_offline_commands_do_not_create_an_archive() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("must-not-exist");
    for args in [vec!["--help"], vec!["--version"], vec!["config", "schema"]] {
        let output = Command::new(env!("CARGO_BIN_EXE_polylogue"))
            .args(["--archive", archive.to_str().unwrap()])
            .args(&args)
            .env_remove("XDG_RUNTIME_DIR")
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        if args == ["config", "schema"] {
            let schema: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
            assert_eq!(schema["additionalProperties"], false);
            assert!(schema["properties"]["archive"].is_object());
        }
        assert!(!archive.exists());
    }
}
#[test]
fn unsupported_features_and_bad_arguments_never_return_empty_success() {
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("not-created");
    for args in [
        vec!["search", "anything"],
        vec!["mcp"],
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
fn actual_daemon_and_client_restart_in_owned_state() {
    use std::process::{Child, Stdio};
    use std::time::{Duration, Instant};
    struct OwnedChild(Child);
    impl Drop for OwnedChild {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let temp = tempfile::tempdir().unwrap();
    let archive = temp.path().join("archive");
    // Unix socket names have a kernel length limit independent of TMPDIR.
    // Only socket dentries live here; database/blob work stays in managed TMPDIR.
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
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let output = Command::new(binary)
                .args(["--archive", archive.to_str().unwrap(), "status"])
                .env("XDG_RUNTIME_DIR", &runtime)
                .output()
                .unwrap();
            if output.status.success() {
                let value: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
                assert_eq!(value["archive"], archive.to_str().unwrap());
                assert_eq!(value["product_path"], "not_implemented");
                break;
            }
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "daemon exited before status: {output:?}"
            );
            assert!(
                Instant::now() < deadline,
                "foundation startup timed out: {output:?}"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    };
    wait_status(&mut first);
    first.0.kill().unwrap();
    first.0.wait().unwrap();
    let mut second = start();
    wait_status(&mut second);
}
