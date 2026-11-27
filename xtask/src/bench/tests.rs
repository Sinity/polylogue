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

#[test]
fn first_build_refuses_ingestion_errors_or_empty_work() {
    let good = json!({"errors":0,"input_mb":8.0,"seconds":1.0,"messages_written":10,"unreadable_root_count":0,"stages":{"read":0.5,"write":1.5}});
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
        ("stages", json!(null)),
        ("stages", json!({"read": null})),
        ("stages", json!({"read": "fast"})),
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

fn listed_file(path: &Path, origin: &str, layout: &str) -> Listed {
    Listed {
        path: path.to_owned(),
        origin: origin.into(),
        layout: layout.into(),
        bytes: fs::metadata(path).map_or(3, |m| m.len()),
        database: false,
    }
}

#[test]
fn a_session_is_grouped_with_its_subagents_and_tool_results() {
    let missing = Path::new("/nonexistent");
    let listed = [
        listed_file(missing, "claude-code", ".claude/projects/-w/s1.jsonl"),
        listed_file(
            missing,
            "claude-code",
            ".claude/projects/-w/s1/subagents/agent-a.jsonl",
        ),
        listed_file(
            missing,
            "claude-code-side",
            ".claude/projects/-w/s1/tool-results/t1.txt",
        ),
        listed_file(missing, "claude-code", ".claude/projects/-w/s2.jsonl"),
        listed_file(
            missing,
            "claude-code-side",
            ".claude/projects/-w/sessions-index.json",
        ),
        listed_file(
            missing,
            "codex",
            ".codex/sessions/2026/01/02/rollout-t.jsonl.zst",
        ),
    ];
    let groups: Vec<(String, Option<String>, String)> = candidates(&listed)
        .into_iter()
        .map(|c| (c.rel.display().to_string(), c.group, c.stratum))
        .collect();
    let s1 = Some(".claude/projects/-w/s1".to_owned());
    assert_eq!(
        groups,
        [
            // Paths order by component, so a session's directory comes
            // before its own file.
            (
                ".claude/projects/-w/s1/subagents/agent-a.jsonl".into(),
                s1.clone(),
                "claude-code".into()
            ),
            (
                ".claude/projects/-w/s1/tool-results/t1.txt".into(),
                s1.clone(),
                "claude-code".into()
            ),
            (
                ".claude/projects/-w/s1.jsonl".into(),
                s1,
                "claude-code".into()
            ),
            (
                ".claude/projects/-w/s2.jsonl".into(),
                Some(".claude/projects/-w/s2".into()),
                "claude-code".into()
            ),
            (
                ".claude/projects/-w/sessions-index.json".into(),
                None,
                "claude-code-side".into()
            ),
            (
                ".codex/sessions/2026/01/02/rollout-t.jsonl.zst".into(),
                None,
                "codex".into()
            ),
        ]
    );
}

#[test]
fn a_drawn_sample_keeps_its_bytes_when_the_source_grows() {
    let directory = tempfile::tempdir().unwrap();
    let source = directory.path().join("session.jsonl");
    fs::write(&source, "first\n").unwrap();
    let chosen = [Candidate {
        path: source.clone(),
        rel: ".claude/projects/-w/session.jsonl".into(),
        stratum: "claude-code@2026".into(),
        size: 6,
        group: Some(".claude/projects/-w/session".into()),
    }];
    let home = directory.path().join("home");
    let copied = copy_sample(&home, &chosen, &[]).unwrap();
    fs::write(&source, "first\nnext\n").unwrap();
    let manifest = Manifest::build("polylogue", &chosen, &copied, 6, 1).unwrap();
    assert_eq!(
        fs::read_to_string(home.join(&chosen[0].rel)).unwrap(),
        "first\n"
    );
    assert_eq!(manifest.entries[0].size, 6);
    assert_eq!(manifest.entries[0].source, home.join(&chosen[0].rel));
    assert_eq!(copied[0].stratum, chosen[0].stratum);
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
    let rel = PathBuf::from(".codex/thread's.sqlite");
    let chosen = [Candidate {
        path: source,
        rel: rel.clone(),
        stratum: "codex-database".into(),
        size: 0,
        group: None,
    }];
    let copied = copy_sample(&directory.path().join("home"), &chosen, &[rel.as_path()]).unwrap();
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
}

#[test]
fn a_sample_size_too_large_for_bytes_is_refused() {
    use clap::Parser;
    let args = ["xtask", "bench", "sample", "--mb", "18446744073709551615"];
    let crate::Cli::Bench(options) = crate::Cli::try_parse_from(args).unwrap() else {
        panic!("benchmark expected");
    };
    assert_eq!(execute(options), ExitCode::FAILURE);
}

#[test]
fn a_storage_other_than_disk_or_memory_is_refused_by_the_parser() {
    use clap::Parser;
    let args = ["xtask", "bench", "run", "--storage", "tape"];
    assert!(crate::Cli::try_parse_from(args).is_err());
}

/// A synthetic home's inputs and a stand-in `first_build` that lists them
/// and, asked to build, prints a result with `cpu_ticks` CPU ticks.
struct Synthetic {
    directory: tempfile::TempDir,
    layouts: Vec<&'static str>,
}

impl Synthetic {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let files = [
            (".claude/projects/-w/s1.jsonl", "claude-code", "{\"a\":1}\n"),
            (
                ".claude/projects/-w/s1/subagents/agent-a.jsonl",
                "claude-code",
                "{\"b\":2}\n",
            ),
            (
                ".claude/projects/-w/s1/tool-results/t1.txt",
                "claude-code-side",
                "output\n",
            ),
            (
                ".codex/sessions/2026/01/02/rollout-t.jsonl.zst",
                "codex",
                "compressed",
            ),
        ];
        let mut listing = String::new();
        for (layout, origin, text) in files {
            let path = directory.path().join("real").join(layout);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(&path, text).unwrap();
            listing.push_str(
                &json!({"path": path, "origin": origin, "layout": layout,
                        "bytes": text.len(), "database": false})
                .to_string(),
            );
            listing.push('\n');
        }
        fs::write(directory.path().join("listing"), listing).unwrap();
        Self {
            directory,
            layouts: files.iter().map(|(layout, _, _)| *layout).collect(),
        }
    }

    #[cfg(unix)]
    fn binary(&self, name: &str, cpu_ticks: u64) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let shell = env::var("SHELL").expect("test environment declares SHELL");
        let listing = self.directory.path().join("listing");
        let result = json!({
            "errors": 0, "unreadable_root_count": 0, "input_mb": 0.001, "seconds": 0.5,
            "messages_written": 3, "cpu_ticks": cpu_ticks, "stages": {"read": 0.25},
        });
        let path = self.directory.path().join(name);
        fs::write(
            &path,
            format!(
                "#!{shell}\nif [ \"$1\" = --list-inputs ]; then cat '{}'; exit 0; fi\nmkdir -p \"$4\"\nprintf '%s\\n' '{result}'\n",
                listing.display()
            ),
        )
        .unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
        path
    }

    /// Runs `xtask bench ARGS` in a child with `environment`, so the
    /// benchmark root and the history store are the test's own.
    fn xtask(&self, args: &[&str], environment: &[(&str, &Path)]) {
        let mut child = Command::new(env::current_exe().unwrap());
        child
            .args([
                "--exact",
                "bench::tests::xtask_bench_child",
                "--ignored",
                "--nocapture",
            ])
            .env("XTASK_TEST_BENCH_ARGS", json!(args).to_string())
            .env("XDG_STATE_HOME", self.directory.path().join("state"))
            .env_remove("POLYLOGUE_BENCH_HOME")
            .env_remove("XDG_DATA_HOME");
        for (key, value) in environment {
            child.env(key, value);
        }
        let output = child.output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "child of the bench CLI tests, which launch it with their environment"]
fn xtask_bench_child() {
    use clap::Parser;
    let Ok(args) = env::var("XTASK_TEST_BENCH_ARGS") else {
        return;
    };
    let args: Vec<String> = serde_json::from_str(&args).unwrap();
    let crate::Cli::Bench(options) =
        crate::Cli::try_parse_from(std::iter::once("xtask".to_owned()).chain(args)).unwrap()
    else {
        panic!("benchmark expected");
    };
    assert_eq!(execute(options), ExitCode::SUCCESS);
}

#[cfg(unix)]
#[test]
fn sample_cli_keeps_manifests_under_flag_environment_and_xdg_roots() {
    let synthetic = Synthetic::new();
    let binary = synthetic.binary("first_build", 100);
    let binary = binary.to_str().unwrap();
    for mode in ["flag", "environment", "xdg"] {
        let root = synthetic
            .directory
            .path()
            .join(format!("bench state {mode}"));
        let root_text = root.to_str().unwrap();
        let mut args = vec![
            "bench", "sample", "--name", "tiny", "--mb", "1", "--binary", binary,
        ];
        if mode == "flag" {
            args.extend(["--root", root_text]);
        }
        let environment: Vec<(&str, &Path)> = match mode {
            "environment" => vec![("POLYLOGUE_BENCH_HOME", root.as_path())],
            "xdg" => vec![("XDG_DATA_HOME", root.as_path())],
            _ => vec![],
        };
        synthetic.xtask(&args, &environment);
        let expected = if mode == "xdg" {
            root.join("polylogue-bench")
        } else {
            root
        };
        let manifest = Manifest::read(&expected.join("manifests/polylogue-tiny.json")).unwrap();
        let mut drawn: Vec<String> = manifest
            .entries
            .iter()
            .map(|e| e.rel.display().to_string())
            .collect();
        drawn.sort();
        assert_eq!(drawn, synthetic.layouts, "{mode}: every group drawn whole");
        for entry in &manifest.entries {
            assert_eq!(entry.source, expected.join("tiny/home").join(&entry.rel));
            assert_eq!(
                fs::read(&entry.source).unwrap(),
                fs::read(synthetic.directory.path().join("real").join(&entry.rel)).unwrap()
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn a_pair_over_a_drawn_sample_records_both_builds_and_their_comparison() {
    let synthetic = Synthetic::new();
    let base = synthetic.binary("base", 100);
    let slow = synthetic.binary("slow", 300);
    let root = synthetic.directory.path().join("bench");
    let root_text = root.to_str().unwrap();
    synthetic.xtask(
        &[
            "bench",
            "--root",
            root_text,
            "sample",
            "--name",
            "tiny",
            "--mb",
            "1",
            "--binary",
            base.to_str().unwrap(),
        ],
        &[],
    );
    synthetic.xtask(
        &[
            "bench",
            "--root",
            root_text,
            "pair",
            "--name",
            "tiny",
            "--repetitions",
            "2",
            "--memory-max",
            "256M",
            "--a",
            &format!("base={}", base.display()),
            "--b",
            &format!("slow={}", slow.display()),
        ],
        &[],
    );
    let store = Store::open_at(&synthetic.directory.path().join("state/devkit/runs.db")).unwrap();
    let recorded = bench::results(&store, "polylogue", 1).unwrap();
    let result = &recorded[0].result;
    assert_eq!(recorded[0].bench, "first_build");
    assert_eq!(
        result["builds"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["label"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["base", "slow"]
    );
    assert_eq!(
        result["repetitions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["build"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["base", "slow", "base", "slow"]
    );
    let verdict = |figure: &str| {
        result["comparison"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["figure"] == figure)
            .unwrap()["verdict"]
            .clone()
    };
    assert_eq!(verdict("process_cpu_seconds"), "3.000x (3.000 to 3.000)");
    assert_eq!(verdict("messages_written"), "no difference");
    assert_eq!(result["limit"]["memory_max"], "256M");
    let unit = &result["repetitions"][0]["record"]["measured"]["accounting"];
    assert!(
        unit["unit"].is_string() || unit["not_applied"].is_string(),
        "{unit}"
    );
    assert!(root.join("tiny/out-base/").is_dir() && root.join("tiny/out-slow/").is_dir());
}
