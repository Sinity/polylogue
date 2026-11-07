use super::*;

#[test]
fn a_sample_name_cannot_escape_its_benchmark_root() {
    for value in [
        "",
        ".",
        "..",
        "../other",
        "/absolute",
        "one/two",
        "one/.",
        "one/",
    ] {
        assert!(sample_name(value).is_err(), "{value}");
    }
    assert_eq!(sample_name("small-1").unwrap(), "small-1");
}

#[cfg(unix)]
#[test]
fn first_build_uses_the_selected_root_lock() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let binary = dir.path().join("fixture");
    let shell = env::var("SHELL").expect("test environment declares SHELL");
    let result = json!({
        "errors": 0,
        "input_mb": 1,
        "seconds": 1,
        "messages_written": 1,
        "unreadable_root_count": 0,
    });
    fs::write(&binary, format!("#!{shell}\nprintf '%s\\n' '{result}'\n")).unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    let lock = dir.path().join("bench.lock");
    first_build(&binary, dir.path(), &dir.path().join("out"), true, &lock).unwrap();
    assert!(lock.is_file());
}

#[test]
fn sample_cli_keeps_manifests_under_flag_environment_and_xdg_roots() {
    for mode in ["flag", "environment", "xdg"] {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("bench state");
        let claude = dir.path().join("claude");
        let codex = dir.path().join("codex");
        fs::create_dir_all(claude.join("projects/synthetic")).unwrap();
        fs::create_dir_all(&codex).unwrap();
        fs::write(
            claude.join("projects/synthetic/session.jsonl"),
            b"{\"type\":\"synthetic\"}\n",
        )
        .unwrap();
        let mut child = Command::new(env::current_exe().unwrap());
        child
            .args(["--exact", "bench::tests::sample_cli_child", "--nocapture"])
            .env("DEVKIT_TEST_BENCH_ROOT", &root)
            .env("DEVKIT_TEST_BENCH_MODE", mode)
            .env("CLAUDE_CONFIG_DIR", &claude)
            .env("CODEX_HOME", &codex)
            .env_remove("POLYLOGUE_BENCH_HOME");
        if mode == "environment" {
            child.env("POLYLOGUE_BENCH_HOME", &root);
        }
        if mode == "xdg" {
            child.env("XDG_DATA_HOME", &root);
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = if mode == "xdg" {
            root.join("polylogue-bench")
        } else {
            root
        };
        let manifest = Manifest::read(&expected.join("manifests/polylogue-tiny.json")).unwrap();
        assert_eq!(manifest.entries.len(), 1);
        assert!(
            manifest.entries[0]
                .source
                .starts_with(expected.join("tiny/source/.claude"))
        );
    }
}

#[test]
fn sample_cli_child() {
    use clap::Parser;
    let Some(root) = env::var_os("DEVKIT_TEST_BENCH_ROOT") else {
        return;
    };
    let mut args = vec![
        "xtask".to_owned(),
        "bench".into(),
        "sample".into(),
        "--name".into(),
        "tiny".into(),
        "--mb".into(),
        "1".into(),
    ];
    if env::var("DEVKIT_TEST_BENCH_MODE").unwrap() == "flag" {
        args.extend(["--root".into(), root.to_string_lossy().into_owned()]);
    }
    let crate::Cli::Bench(options) = crate::Cli::try_parse_from(args).unwrap() else {
        panic!("benchmark expected");
    };
    assert_eq!(execute(options), ExitCode::SUCCESS);
}

#[test]
fn first_build_refuses_ingestion_errors_or_empty_work() {
    let good = json!({"errors":0,"input_mb":8.0,"seconds":1.0,"messages_written":10,"unreadable_root_count":0});
    assert_eq!(
        first_build_result(&format!("progress\n{good}\n"), true).unwrap(),
        good
    );
    for (field, value) in [
        ("errors", json!(2)),
        ("unreadable_root_count", json!(1)),
        ("messages_written", json!(0)),
        ("input_mb", json!(0)),
        ("seconds", json!(0)),
    ] {
        let mut bad = good.clone();
        bad[field] = value;
        assert!(first_build_result(&bad.to_string(), true).is_err(), "{bad}");
    }
    let mut idle = good.clone();
    idle["messages_written"] = json!(0);
    assert_eq!(first_build_result(&idle.to_string(), false).unwrap(), idle);
    idle["errors"] = json!(1);
    assert!(first_build_result(&idle.to_string(), false).is_err());
    let mut unknown = good;
    unknown.as_object_mut().unwrap().remove("errors");
    assert!(first_build_result(&unknown.to_string(), true).is_err());
}

#[test]
fn a_pair_needs_two_labelled_builds() {
    assert_eq!(
        labelled("base=/bin/a").unwrap(),
        ("base".to_owned(), PathBuf::from("/bin/a"))
    );
    for bad in ["base", "=/bin/a", "base=", "../base=/bin/a"] {
        assert!(labelled(bad).is_err(), "{bad}");
    }
}

#[test]
fn a_drawn_sample_keeps_its_bytes_when_the_source_grows() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("session.jsonl");
    fs::write(&source, "first\n").unwrap();
    let chosen = [SampleFile {
        path: source.clone(),
        rel: ".claude/projects/session.jsonl".into(),
        stratum: "claude/main@2026".into(),
        size: 6,
        grows: true,
    }];
    let copied = copy_sample(&directory.path().join("copies"), &chosen).unwrap();
    fs::write(&source, "first\nnext\n").unwrap();
    let manifest = Manifest::build("polylogue", &chosen, &copied, 6, 1, 1.0).unwrap();
    let home = directory.path().join("home");
    manifest.materialize(&home).unwrap();
    assert_eq!(
        fs::read_to_string(home.join(&chosen[0].rel)).unwrap(),
        "first\n"
    );
    assert_eq!(copied[0].size, 6);
    assert_eq!(copied[0].stratum, chosen[0].stratum);
    assert!(copied[0].grows);
}

#[test]
fn a_database_sample_includes_committed_wal_rows_only() {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, Stdio};

    struct Writer(Child);
    impl Drop for Writer {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("source.sqlite");
    let mut writer = Writer(
        Command::new("sqlite3")
            .arg(&source)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut output = BufReader::new(writer.0.stdout.take().unwrap());
    writer.0.stdin.as_mut().unwrap().write_all(b".bail on\nPRAGMA journal_mode=WAL;\nCREATE TABLE item(value); INSERT INTO item VALUES (1);\nBEGIN; INSERT INTO item VALUES (2);\nSELECT 'ready';\n").unwrap();
    for expected in ["wal", "ready"] {
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        assert_eq!(line.trim(), expected);
    }
    let chosen = [SampleFile {
        path: source,
        rel: ".codex/thread's.sqlite".into(),
        stratum: "codex/database".into(),
        size: 0,
        grows: false,
    }];
    let copied = copy_sample(&directory.path().join("copies"), &chosen).unwrap();
    let query = || {
        let result = Command::new("sqlite3")
            .arg("-readonly")
            .arg(&copied[0].path)
            .arg("SELECT sum(value) FROM item; PRAGMA integrity_check;")
            .output()
            .unwrap();
        assert!(result.status.success(), "{result:?}");
        String::from_utf8(result.stdout).unwrap()
    };
    assert_eq!(query(), "1\nok\n");
    writer
        .0
        .stdin
        .as_mut()
        .unwrap()
        .write_all(b"COMMIT;\n")
        .unwrap();
    drop(writer.0.stdin.take());
    assert!(writer.0.wait().unwrap().success());
    assert_eq!(query(), "1\nok\n");
    let manifest = Manifest::build("polylogue", &chosen, &copied, 1, 1, 1.0).unwrap();
    let home = directory.path().join("home");
    manifest.materialize(&home).unwrap();
    assert_eq!(
        fs::read(home.join(&chosen[0].rel)).unwrap(),
        fs::read(&copied[0].path).unwrap()
    );
}
