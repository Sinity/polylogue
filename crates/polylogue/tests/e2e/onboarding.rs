use serde_json::Value;

#[test]
fn demo_steps_run_through_rows() {
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
    .args(["--json", "demo"])
    .arg(&directory)
    .output()
    .unwrap();
    assert!(!operator_archive.exists());
    assert!(output.status.success(), "{output:?}");
    let text = std::str::from_utf8(&output.stdout).unwrap();
    assert!(!text.contains(['\u{9b}', '\x7f', '\u{202e}']));
    let tour: Value = serde_json::from_slice(&output.stdout).unwrap();
    let declared_directory = directory.clone();
    let directory = directory.canonicalize().unwrap();
    let steps = tour["steps"].as_array().unwrap();
    assert_eq!(steps.len(), 10);
    for (index, step) in steps.iter().enumerate() {
        if index != 6 {
            assert_eq!(step["command"], "rows", "step {index}");
            assert!(step["arguments"]["selection"].is_string(), "step {index}");
            if let Some(reports) = step["result"]["reply"]["steps"].as_array() {
                for report in reports {
                    assert_eq!(
                        report["outcome"]["status"], "done",
                        "step {index}: {report}"
                    );
                }
            }
        }
    }
    assert_eq!(
        steps[0]["command_line"],
        "polylogue 'messages where written_by:human chrono'"
    );
    assert_eq!(
        steps[1]["command_line"],
        "polylogue session:claude-code:7f3a2c10-0002-4000-8000-000000000002 read --format markdown"
    );
    assert_eq!(
        steps[7]["command_line"],
        "polylogue session:claude-code:7f3a2c10-0002-4000-8000-000000000002 note 'Use chrono for date parsing.' --as decision"
    );
    assert_eq!(
        steps[9]["command_line"],
        "polylogue session:claude-code:7f3a2c10-0002-4000-8000-000000000002 brief"
    );
    assert_eq!(tour["first_query_seconds"], steps[0]["seconds"]);
    // Preparation imports the bundled exports, so it takes measurable time, and
    // the transcript header reports the same figures as the tour.
    let preparation = tour["preparation_seconds"].as_f64().unwrap();
    assert!(preparation > 0.0);
    assert!(
        tour["transcript"].as_str().unwrap().contains(&format!(
            "Preparation: {preparation:.3}s. First query: {:.3}s.",
            tour["first_query_seconds"].as_f64().unwrap()
        )),
        "{}",
        tour["transcript"]
    );
    let examples: Value = serde_json::from_str(include_str!(
        "../../../../examples/morrow/expected/main/expected.json"
    ))
    .unwrap();
    let short = |row: &Value| {
        polylogue_testkit::short(
            row["session"]
                .as_str()
                .or_else(|| row["id"].as_str())
                .unwrap(),
        )
    };
    let mut chrono_sessions: Vec<_> = steps[0]["result"]["reply"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(short)
        .collect();
    let mut expected_chrono_sessions: Vec<_> =
        examples["queries"]["13 messages where written_by:human \"chrono\""]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r[0].as_str().unwrap().to_owned())
            .collect();
    chrono_sessions.sort();
    expected_chrono_sessions.sort();
    assert_eq!(chrono_sessions, expected_chrono_sessions);
    let mut errors: Vec<_> = steps[2]["result"]["reply"]["rows"]
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
    let mut children: Vec<_> = steps[3]["result"]["reply"]["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(short)
        .collect();
    children.sort();
    assert_eq!(children, ["#3"]);
    let unknown: Vec<_> = steps[5]["result"]["reply"]["rows"]
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
    let notes = steps[8]["result"]["reply"]["rows"].as_array().unwrap();
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert_eq!(notes[0]["value"], "Use chrono for date parsing.");
    // A takeover names an annotation by another author that the operator has
    // not accepted, without quoting it (docs/agents.md, Brief).
    let takeover = steps[9]["result"]["reply"]["steps"][0]["outcome"]["reply"]["reply"]["prepared"]
        ["text"]
        .as_str()
        .unwrap();
    let session = "session:claude-code:7f3a2c10-0002-4000-8000-000000000002";
    let pointer = format!(
        "Annotations by other authors, not yet accepted: 1; list them with `polylogue 'annotations where NOT author:operator judged:unknown about:\"{session}\"'`."
    );
    assert_eq!(
        takeover
            .lines()
            .filter(|line| line.starts_with("Annotations by other authors"))
            .collect::<Vec<_>>(),
        [pointer.as_str()],
        "{takeover}"
    );
    assert!(
        !takeover.contains("Use chrono for date parsing."),
        "{takeover}"
    );
    assert_eq!(
        std::fs::read_to_string(directory.join("transcript.md")).unwrap(),
        tour["transcript"].as_str().unwrap()
    );
    // Bundled collected files must exactly match Testkit's Main layout.
    let main = polylogue_testkit::Home::morrow(polylogue_testkit::Set::Main);
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

    // The README daemon uses the same declared absolute aliases as the tour.
    // Canonical paths above are only for inspecting the bundled bytes.
    let home = declared_directory.join("home");
    let archive = declared_directory.join("archive");
    let config = declared_directory.join("config");
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
    let restarted: command_surface::Answer<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        crate::cli::listing(restarted.value().expect("query succeeded").clone())["rows"],
        steps[2]["result"]["reply"]["rows"]
    );
    drop(daemon);
    let marker = directory.join("keep.txt");
    std::fs::write(&marker, "keep this file").unwrap();
    let retry = crate::command(temp.path())
        .args(["demo"])
        .arg(&directory)
        .output()
        .unwrap();
    assert!(!retry.status.success());
    assert_eq!(std::fs::read_to_string(marker).unwrap(), "keep this file");
}
