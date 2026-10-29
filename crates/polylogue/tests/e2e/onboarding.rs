use serde_json::Value;

#[test]
fn public_morrow_tour_matches_current_query_examples() {
    let temp = tempfile::tempdir().unwrap();
    let actual = temp.path().join("a");
    let alias = temp.path().join("b");
    std::fs::create_dir(&actual).unwrap();
    std::os::unix::fs::symlink(&actual, &alias).unwrap();
    let directory = alias.join("t\u{9b}\x7f\u{202e}");
    let operator_archive = temp.path().join("operator-archive");
    let output = crate::Fixture::new(
        &operator_archive,
        &temp.path().join("home"),
        &temp.path().join("config"),
    )
    .command()
    .args(["--json", "demo", "tour"])
    .arg(&directory)
    .output()
    .unwrap();
    assert!(!operator_archive.exists());
    assert!(output.status.success(), "{output:?}");
    let text = std::str::from_utf8(&output.stdout).unwrap();
    assert!(!text.contains(['\u{9b}', '\x7f', '\u{202e}']));
    let tour: Value = serde_json::from_slice(&output.stdout).unwrap();
    let directory = directory.canonicalize().unwrap();
    let steps = tour["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 10);
    assert_eq!(tour["first_query_seconds"], steps[0]["seconds"]);
    assert!(tour["preparation_seconds"].as_f64().unwrap() >= 0.0);
    let examples: Value = serde_json::from_str(include_str!(
        "../../../../examples/morrow/expected/main/expected.json"
    ))
    .unwrap();
    assert_eq!(
        steps[0]["result"]["rows"].as_array().unwrap().len(),
        examples["queries"]["13 messages where written_by:human \"chrono\""]
            .as_array()
            .unwrap()
            .len()
    );
    let short = |row: &Value| {
        polylogue_testkit::short(
            row["session"]
                .as_str()
                .or_else(|| row["id"].as_str())
                .unwrap(),
        )
    };
    let mut errors: Vec<_> = steps[2]["result"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                short(r),
                r["command"].as_str().unwrap().to_owned(),
                r["exit_code"].as_i64().unwrap(),
            )
        })
        .collect();
    let mut expected_errors: Vec<_> = examples["queries"]["15 tool_calls where outcome:error"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                r[0].as_str().unwrap().to_owned(),
                r[1].as_str().unwrap().to_owned(),
                r[2].as_i64().unwrap(),
            )
        })
        .collect();
    errors.sort();
    expected_errors.sort();
    assert_eq!(errors, expected_errors);
    let mut children: Vec<_> = steps[3]["result"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(short)
        .collect();
    children.sort();
    assert_eq!(children, ["#3", "#4"]);
    let unknown: Vec<_> = steps[5]["result"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(short)
        .collect();
    let expected_unknown: Vec<_> = examples["queries"]["9 cost:unknown"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    assert_eq!(unknown, expected_unknown);
    assert_eq!(steps[8]["result"]["rows"].as_array().unwrap().len(), 1);
    assert!(
        steps[9]["result"]
            .to_string()
            .contains("Use chrono for date parsing.")
    );
    assert!(
        steps
            .iter()
            .all(|s| !s["text"].as_str().unwrap().is_empty())
    );
    assert_eq!(
        std::fs::read_to_string(directory.join("transcript.md")).unwrap(),
        tour["transcript"].as_str().unwrap()
    );
    // Bundled collected files must exactly match Testkit's Main layout.
    let main = polylogue_testkit::Morrow::lay_out(polylogue_testkit::Set::Main);
    let home = directory.join("home");
    for path in main.provider_files() {
        let relative = path.strip_prefix(&main.home).unwrap();
        assert_eq!(
            std::fs::read(home.join(relative)).unwrap(),
            std::fs::read(path).unwrap()
        );
    }
    fn files(root: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                files(&path, out);
            } else {
                out.push(path);
            }
        }
    }
    let mut bundled = Vec::new();
    files(&home, &mut bundled);
    bundled.retain(|p| !p.starts_with(home.join("exports")));
    assert_eq!(bundled.len(), main.provider_files().len());

    // The README daemon uses this machine's identity and the same absolute paths.
    let archive = directory.join("archive");
    let config = directory.join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let before = {
        let pool =
            polylogue_store::ReadPool::open(&polylogue_store::Archive::at(&archive)).unwrap();
        let reader = pool.reader().unwrap();
        reader.status().unwrap()
    };
    let daemon = super::cli::start_daemon(&env);
    let status = super::cli::wait_status(&env, |_| true);
    assert_eq!(status["store"]["sessions"], 8);
    assert_eq!(status["store"]["inputs"], before.inputs);
    assert_eq!(status["store"]["revisions"], before.revisions);
    assert_eq!(status["store"]["stored_bytes"], before.stored_bytes);
    let output = super::cli::cli(&env, &["--json", "tool_calls where outcome:error"]);
    assert!(output.status.success(), "{output:?}");
    let restarted: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(restarted["rows"], steps[2]["result"]["rows"]);
    drop(daemon);
    let marker = directory.join("keep.txt");
    std::fs::write(&marker, "keep this file").unwrap();
    let retry = crate::command(temp.path())
        .args(["demo", "tour"])
        .arg(&directory)
        .output()
        .unwrap();
    assert!(!retry.status.success());
    assert_eq!(std::fs::read_to_string(marker).unwrap(), "keep this file");
}
