use super::*;

fn old_cli(binary: &Path, env: &[(&str, &Path)], args: &[&str]) -> Output {
    let mut command = Command::new(binary);
    command.args(args);
    for (key, value) in env {
        command.env(key, value);
    }
    command.output().unwrap()
}

fn old_daemon(binary: &Path, env: &[(&str, &Path)]) -> Daemon {
    let mut command = Command::new(binary);
    command.arg("daemon").stderr(Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut daemon = Daemon(polylogue_testkit::spawn_daemon(&mut command).unwrap());
    let mut first = String::new();
    BufReader::new(daemon.0.stderr.as_mut().unwrap())
        .read_line(&mut first)
        .unwrap();
    assert!(first.contains("serving"), "{first}");
    daemon
}

fn batches(archive: &Path) -> Vec<(polylogue_store::sinex_rows::Unit, Vec<u8>)> {
    let source = polylogue_store::sinex_rows::Published::open(archive).unwrap();
    let inventory = source.inventory().unwrap();
    (1..=inventory.len())
        .map(|position| {
            let unit = inventory.unit(position).unwrap();
            let mut batch = source.read(&unit).unwrap();
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut batch.file, &mut bytes).unwrap();
            (unit, bytes)
        })
        .collect()
}

#[test]
#[ignore = "requires a preserved native Archive9 frontend in POLYLOGUE_OLD9"]
fn actual_archive9_legacy_imports_rebuild_after_a_new_kept_observation() {
    let binary = std::env::var_os("POLYLOGUE_OLD9").expect("set POLYLOGUE_OLD9");
    for add_kept in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        let archive = dir.path().join("archive");
        let config = dir.path().join("config");
        std::fs::create_dir(&home).unwrap();
        let env = [
            ("POLYLOGUE_HOME", home.as_path()),
            ("POLYLOGUE_ARCHIVE", archive.as_path()),
            ("XDG_CONFIG_HOME", config.as_path()),
        ];
        let bytes = format!(
            r#"{{"metadata":"{}","conversations":[{{"conversation":{{"id":"legacy-native"}},"responses":[{{"responseId":"one","sender":"human","message":"native legacy interpretation"}}]}}]}}"#,
            "synthetic ".repeat(12)
        );
        let opaque = dir.path().join("opaque.json");
        let standard = dir.path().join("prod-grok-backend.json");
        for path in [&opaque, &standard] {
            std::fs::write(path, &bytes).unwrap();
        }
        let daemon = old_daemon(Path::new(&binary), &env);
        for path in [&opaque, &standard] {
            let output = old_cli(
                Path::new(&binary),
                &env,
                &["--json", "import", path.to_str().unwrap()],
            );
            assert!(output.status.success(), "{output:?}");
            assert!(
                json(&output)["pass"]["errors"]
                    .as_array()
                    .unwrap()
                    .is_empty()
            );
        }
        let before = old_cli(
            Path::new(&binary),
            &env,
            &["--json", "read", "grok:legacy-native"],
        );
        assert!(before.status.success(), "{before:?}");
        let messages = json(&before)["messages"].clone();
        drop(daemon);
        let old_batches = batches(&archive);
        if add_kept {
            let daemon = start_daemon(&env);
            let output = cli(&env, &["--json", "import", opaque.to_str().unwrap()]);
            assert!(output.status.success(), "{output:?}");
            assert_eq!(json(&output)["unrecognized"].as_array().unwrap().len(), 1);
            drop(daemon);
        }
        std::fs::remove_file(opaque).unwrap();
        std::fs::remove_file(standard).unwrap();
        polylogue_testkit::delete_index(&archive);
        let _daemon = start_daemon(&env);
        let started = Instant::now();
        let status = wait_status(&env, |status| {
            assert!(started.elapsed() < Duration::from_secs(10), "{status}");
            status["store"]["messages"] == 1
        });
        assert!(status["last_pass"]["errors"].as_array().unwrap().is_empty());
        assert_eq!(
            status["last_pass"]["legacy_imports"][0]["candidates"][0]["format"],
            "grok"
        );
        let after = cli(&env, &["--json", "read", "grok:legacy-native"]);
        assert!(after.status.success(), "{after:?}");
        assert_eq!(json(&after)["messages"], messages);
        let source = polylogue_store::sinex_rows::Published::open(&archive).unwrap();
        for (unit, expected) in old_batches {
            let mut batch = source.read(&unit).unwrap();
            let mut bytes = Vec::new();
            std::io::Read::read_to_end(&mut batch.file, &mut bytes).unwrap();
            assert_eq!(
                bytes, expected,
                "migration must preserve exact acknowledged units"
            );
        }
    }
}
