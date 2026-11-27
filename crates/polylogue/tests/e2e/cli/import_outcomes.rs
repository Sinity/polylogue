use super::*;

#[test]
fn file_import_reports_what_was_stored_and_logs_failures() {
    use command_surface::{Answer, FailureKind, Outcome};

    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir(&home).unwrap();
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let export = dir.path().join("export");
    let project = export.join(".claude/projects/-p");
    std::fs::create_dir_all(&project).unwrap();
    let first = project.join("first.jsonl");
    std::fs::write(&first, line("u1", "first", "the first import")).unwrap();
    let _daemon = start_daemon(&env);

    let first_path = first.to_str().unwrap();
    let out = cli(&env, &["--json", "import", first_path]);
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    let answer: Answer<polylogue_engine::ImportReport> =
        serde_json::from_slice(&out.stdout).unwrap();
    let Outcome::Ok(report) = answer.outcome else {
        panic!("{answer:?}");
    };
    assert_eq!(report.imported.as_slice(), std::slice::from_ref(&first));
    assert_eq!(report.readable, polylogue_engine::Readable::Now);

    let out = cli(&env, &["--json", "import", first_path]);
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    let answer: Answer<polylogue_engine::ImportReport> =
        serde_json::from_slice(&out.stdout).unwrap();
    let Outcome::Ok(report) = answer.outcome else {
        panic!("{answer:?}");
    };
    assert_eq!(
        report.already_stored.as_slice(),
        std::slice::from_ref(&first)
    );
    std::fs::remove_file(first).unwrap();
    assert_eq!(
        answer.envelope.notes,
        ["Everything named was already stored."]
    );

    let second = project.join("second.jsonl");
    std::fs::write(&second, line("u2", "second", "the second import")).unwrap();
    let unknown = export.join("unknown.json");
    std::fs::write(&unknown, b"{}").unwrap();
    let unknown_path = unknown.to_str().unwrap();
    let refusal = format!("refused {unknown_path}: no declared structure matched");
    let out = cli(&env, &["--json", "import", export.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(5), "{out:?}");
    let answer: Answer<polylogue_engine::ImportReport> =
        serde_json::from_slice(&out.stdout).unwrap();
    let Outcome::Degraded { value, missing } = answer.outcome else {
        panic!("{answer:?}");
    };
    assert_eq!(value.imported, [second]);
    assert_eq!(value.refused.len(), 1);
    assert_eq!(value.refused[0].path, unknown);
    assert_eq!(value.refused[0].reason, "no declared structure matched");
    assert_eq!(missing.as_slice(), std::slice::from_ref(&refusal));

    let third = project.join("third.jsonl");
    std::fs::write(&third, line("u3", "third", "the third import")).unwrap();
    let out = cli(&env, &["import", third.to_str().unwrap(), unknown_path]);
    assert_eq!(out.status.code(), Some(5), "{out:?}");
    assert_eq!(
        String::from_utf8(out.stdout).unwrap(),
        format!("stored: {}\n", third.display())
    );
    assert_eq!(
        String::from_utf8(out.stderr).unwrap(),
        format!("polylogue: note: {refusal}\n")
    );

    let out = cli(&env, &["--json", "sessions"]);
    let rows = listing(json(&out));
    let mut ids: Vec<_> = rows["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect();
    ids.sort_unstable();
    assert_eq!(
        ids,
        [
            "claude-code:first",
            "claude-code:second",
            "claude-code:third"
        ]
    );

    let out = cli(&env, &["--json", "import", unknown_path]);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let answer: Answer<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
    let failure = answer.failure().unwrap();
    assert_eq!(failure.kind, FailureKind::BadValue);
    assert_eq!(failure.message, format!("no files were stored\n{refusal}"));

    let absent = export.join("absent.json");
    let absent_path = absent.to_str().unwrap();
    let out = cli(&env, &["--json", "import", absent_path]);
    assert_eq!(out.status.code(), Some(4), "{out:?}");
    let answer: Answer<serde_json::Value> = serde_json::from_slice(&out.stdout).unwrap();
    let failure = answer.failure().unwrap();
    assert_eq!(failure.kind, FailureKind::Io);
    let cause = format!(
        "failed {absent_path}: {}",
        std::fs::File::open(&absent).unwrap_err()
    );
    assert_eq!(failure.message, cause);

    let status = cli(&env, &["--json", "status", "--log"]);
    assert_eq!(status.status.code(), Some(0), "{status:?}");
    let status = json(&status);
    let imports: Vec<_> = status["recent_log"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|record| {
            let text = record["text"].as_str().unwrap();
            text.starts_with("import ")
                .then_some((record["priority"].as_str().unwrap(), text))
        })
        .collect();
    assert_eq!(
        imports,
        [
            ("warning", format!("import incomplete: {refusal}").as_str()),
            ("warning", format!("import incomplete: {refusal}").as_str()),
            (
                "error",
                format!("import failed (bad_value): no files were stored | {refusal}").as_str()
            ),
            ("error", format!("import failed (io): {cause}").as_str()),
        ]
    );
}

#[test]
fn file_import_reports_a_mistyped_origin_once() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let export = dir.path().join("export");
    std::fs::create_dir(&home).unwrap();
    std::fs::create_dir(&export).unwrap();
    for member in ["first.json", "second.json", "third.json"] {
        std::fs::write(export.join(member), b"{}").unwrap();
    }
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    let args = ["import", export.to_str().unwrap(), "--origin", "nosuch"];

    let out = cli(&env, &args);
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let text = String::from_utf8(out.stderr).unwrap();
    assert_eq!(text.lines().count(), 1, "{text}");
    assert!(text.contains("--origin"), "{text}");
    assert!(text.contains("python-vectors"), "{text}");
    assert_eq!(text.matches("nosuch").count(), 1, "{text}");

    let out = cli(
        &env,
        &["--json"].into_iter().chain(args).collect::<Vec<_>>(),
    );
    assert_eq!(out.status.code(), Some(2), "{out:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&out.stdout).unwrap();
    let failure = answer.failure().unwrap();
    assert_eq!(failure.kind, command_surface::FailureKind::BadValue);
    assert_eq!(failure.code.as_deref(), Some("bad_value"));
    assert!(failure.message.contains("claude-code"), "{failure:?}");
}

#[test]
fn import_prints_a_runnable_next_call() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    std::fs::create_dir(&home).unwrap();
    let env = crate::Fixture::new(
        &dir.path().join("archive"),
        &home,
        &dir.path().join("config"),
    );
    let _daemon = start_daemon(&env);
    let out = cli(&env, &["import", "grok", "--wait", "1ms"]);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let text = String::from_utf8(out.stderr).unwrap();
    let next = text
        .strip_prefix("polylogue: wait again: polylogue ")
        .unwrap();
    let args: Vec<_> = ["--json"]
        .into_iter()
        .chain(next.split_whitespace())
        .collect();
    let out = cli(&env, &args);
    assert_eq!(out.status.code(), Some(1), "{out:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(answer.value().unwrap()["job"]["status"], "waiting");
    assert_eq!(answer.envelope.next.len(), 1);
    assert_eq!(answer.envelope.next[0].request.operation, "import");
    assert_eq!(
        answer.envelope.next[0].request.arguments["paths"],
        serde_json::json!(["grok"])
    );
}
