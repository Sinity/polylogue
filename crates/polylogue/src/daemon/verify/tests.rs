use polylogue_store::verify::Verification;

#[test]
fn verify_through_the_daemon_checks_the_archive_and_writes_a_new_manifest() {
    let daemon = crate::test_daemon::start("words to verify");
    let manifest = daemon._dir.path().join("manifest.tsv");
    let request = crate::client::request(
        "verify",
        serde_json::json!({"manifest": manifest}),
        "verify-manifest".into(),
    );
    // A verification reads at idle priority, without the default deadline.
    let row = crate::commands::command("verify").unwrap();
    let policy = row
        .registration()
        .unwrap()
        .admit(
            &request.arguments,
            row,
            &crate::commands::Caller {
                surface: crate::commands::Surface::Cli,
                cwd: None,
            },
        )
        .unwrap();
    assert!(policy.read_only && !policy.default_deadline);
    let socket = daemon.daemon.archive.socket();
    let answer = crate::client::call(&socket, &request).unwrap();
    let Some(value) = answer.value().cloned() else {
        panic!("{answer:?}");
    };
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let found: Verification = serde_json::from_value(value).unwrap();
    assert_eq!(found.problems, []);
    assert!(found.counts.inputs >= 1);
    assert_eq!(found.blobs.checked, found.counts.blobs);
    let text = std::fs::read_to_string(&manifest).unwrap();
    assert_eq!(text.lines().count() as u64, found.counts.revisions);
    // The transcript's bytes, hashed here, are one of the revisions.
    let transcript = std::fs::read(daemon.home.join(".claude/projects/-p/s1.jsonl")).unwrap();
    let expected = format!(
        "\t1\t{}\t{}",
        transcript.len(),
        blake3::hash(&transcript).to_hex()
    );
    assert!(text.lines().any(|line| line.ends_with(&expected)), "{text}");

    // An existing file is never replaced, and no partial file is left.
    let again = crate::client::call(&socket, &request).unwrap();
    assert!(again.value().is_none(), "{again:?}");
    assert_eq!(std::fs::read_to_string(&manifest).unwrap(), text);
    let left: Vec<_> = std::fs::read_dir(daemon._dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|name| name.starts_with(".polylogue-manifest-"))
        .collect();
    assert_eq!(left, Vec::<String>::new());
}

#[test]
fn a_manifest_path_must_be_absolute_on_the_daemon() {
    let daemon = crate::test_daemon::start("words to verify");
    // A caller without a working directory (HTTP) names an absolute path.
    let answer =
        crate::test_daemon::Call::new("verify", serde_json::json!({"manifest": "relative.tsv"}))
            .surface(crate::commands::Surface::Http)
            .answer(&daemon.daemon);
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = &answer
    else {
        panic!("{answer:?}");
    };
    assert!(error.message.contains("absolute path"), "{answer:?}");
}
