fn failure(answer: &command_surface::Answer<serde_json::Value>) -> &command_surface::Failure {
    match &answer.outcome {
        command_surface::Outcome::Error(error) => error,
        _ => panic!("expected a refusal: {answer:?}"),
    }
}

#[test]
fn verify_through_the_daemon_checks_the_archive_and_writes_a_new_manifest() {
    let daemon = crate::test_daemon::start("words to verify");
    let manifest = daemon._dir.path().join("manifest.tsv");
    let request = crate::client::request(
        "status",
        serde_json::json!({"verify": true, "manifest": manifest}),
        "verify-manifest".into(),
    );
    // A verification reads at idle priority, without the default deadline.
    let policy = crate::daemon::handler::policy(
        crate::commands::command("status").unwrap(),
        &request.arguments,
        polylogue_model::authorship::Channel::Cli,
    )
    .unwrap();
    assert!(!policy.changes_archive);
    assert_eq!(policy.bound, crate::daemon::handler::Bound::CallerOnly);
    assert_eq!(policy.lane, crate::daemon::handler::Lane::Background);
    assert_eq!(crate::client::connection_deadline(&request, None), None);
    let socket = daemon.daemon.archive.socket();
    let answer = crate::client::call(&socket, &request).unwrap();
    let Some(value) = answer.value().cloned() else {
        panic!("{answer:?}");
    };
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let status: crate::protocol::status::Answer = serde_json::from_value(value).unwrap();
    assert!(status.vectors.is_some());
    let found = status.verification.unwrap();
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
fn verification_is_cli_only_and_manifest_requires_verification() {
    let daemon = crate::test_daemon::start("words to verify");
    let manifest = daemon._dir.path().join("not-created.tsv");
    for surface in [
        command_surface::Via::Mcp,
        command_surface::Via::Web,
        command_surface::Via::Hook,
    ] {
        let answer = crate::test_daemon::Call::new(
            "status",
            serde_json::json!({"verify": true, "manifest": manifest}),
        )
        .channel(surface)
        .answer(&daemon.daemon);
        assert!(
            failure(&answer).message.contains("command line"),
            "{answer:?}"
        );
        assert!(!manifest.exists());
    }
    for arguments in [
        serde_json::json!({"manifest": manifest}),
        serde_json::json!({"line": true, "verify": true}),
        serde_json::json!({"line": true, "vectors": true}),
    ] {
        let answer = crate::test_daemon::Call::new("status", arguments).answer(&daemon.daemon);
        assert_eq!(
            failure(&answer).kind,
            command_surface::FailureKind::BadRequest
        );
    }
    let answer = crate::test_daemon::Call::new(
        "status",
        serde_json::json!({"verify": true, "manifest": "relative.tsv"}),
    )
    .answer(&daemon.daemon);
    assert!(failure(&answer).message.contains("absolute path"));
}

#[test]
fn explicit_vector_status_counts_all_models_without_a_provider() {
    use polylogue_store::vectors::{KeyStatus, Vector, VectorStatus, Vectors};
    let daemon = crate::test_daemon::start("synthetic words");
    let clock = &daemon.daemon.clock;
    let mut vectors = Vectors::writer(&daemon.daemon.archive, 256).unwrap();
    vectors
        .put(
            "model-a",
            &[
                Vector {
                    text_hash: vec![1; 32],
                    chunk: 0,
                    values: vec![1.0, 0.0],
                },
                Vector {
                    text_hash: vec![1; 32],
                    chunk: 1,
                    values: vec![0.0, 1.0],
                },
                Vector {
                    text_hash: vec![2; 32],
                    chunk: 0,
                    values: vec![1.0, 0.0],
                },
            ],
            clock,
        )
        .unwrap();
    vectors
        .put(
            "model-b",
            &[Vector {
                text_hash: vec![1; 32],
                chunk: 0,
                values: vec![1.0],
            }],
            clock,
        )
        .unwrap();
    vectors
        .fail("model-a", &[3; 32], "synthetic refusal", clock)
        .unwrap();
    drop(vectors);
    for surface in [
        command_surface::Via::Cli,
        command_surface::Via::Mcp,
        command_surface::Via::Web,
    ] {
        let ordinary = crate::test_daemon::Call::new("status", serde_json::json!({}))
            .channel(surface)
            .answer(&daemon.daemon);
        let ordinary: crate::protocol::status::Answer =
            serde_json::from_value(ordinary.value().unwrap().clone()).unwrap();
        assert!(
            ordinary.vectors.is_none()
                && ordinary.vectors_error.is_none()
                && ordinary.verification.is_none()
        );
        let answer = crate::test_daemon::Call::new("status", serde_json::json!({"vectors": true}))
            .channel(surface)
            .answer(&daemon.daemon);
        let status: crate::protocol::status::Answer =
            serde_json::from_value(answer.value().unwrap().clone()).unwrap();
        assert_eq!(
            status.vectors,
            Some(VectorStatus {
                keys: vec![
                    KeyStatus {
                        key: "model-a".into(),
                        dims: Some(2),
                        texts: 2,
                        vectors: 3,
                        failures: 1
                    },
                    KeyStatus {
                        key: "model-b".into(),
                        dims: Some(1),
                        texts: 1,
                        vectors: 1,
                        failures: 0
                    },
                ]
            })
        );
        assert!(status.vectors_error.is_none());
    }
    let line = crate::test_daemon::Call::new("status", serde_json::json!({"line": true}))
        .answer(&daemon.daemon);
    assert!(line.value().unwrap().get("vectors").is_none());
    // A broken auxiliary store must not break ordinary status or look empty.
    let broken = crate::test_daemon::start("synthetic unreadable vector store");
    std::fs::remove_file(broken.daemon.archive.embeddings_db()).unwrap();
    let ordinary =
        crate::test_daemon::Call::new("status", serde_json::json!({})).answer(&broken.daemon);
    assert!(ordinary.value().is_some(), "{ordinary:?}");
    let detail = crate::test_daemon::Call::new("status", serde_json::json!({"vectors": true}))
        .answer(&broken.daemon);
    assert_eq!(detail.exit_code(), 5, "{detail:?}");
    assert!(detail.value().unwrap().get("vectors").is_none());
    assert!(detail.value().unwrap()["vectors_error"].as_str().is_some());
}

#[test]
fn manifest_resolves_the_cli_directory_and_refuses_symlinks() {
    let daemon = crate::test_daemon::start("synthetic original bytes");
    let directory = tempfile::tempdir().unwrap();
    let answer = crate::test_daemon::Call::new(
        "status",
        serde_json::json!({"verify":true,"manifest":"manifest.tsv"}),
    )
    .cwd(directory.path().into())
    .answer(&daemon.daemon);
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let manifest = directory.path().join("manifest.tsv");
    let original = std::fs::read(&manifest).unwrap();
    let link = directory.path().join("link.tsv");
    std::os::unix::fs::symlink(&manifest, &link).unwrap();
    let answer =
        crate::test_daemon::Call::new("status", serde_json::json!({"verify":true,"manifest":link}))
            .answer(&daemon.daemon);
    assert!(answer.value().is_none(), "{answer:?}");
    assert_eq!(std::fs::read(&manifest).unwrap(), original);
    // A link whose target is missing exists too: writing through it would
    // create the target.
    let missing = directory.path().join("missing.tsv");
    let dangling = directory.path().join("dangling.tsv");
    std::os::unix::fs::symlink(&missing, &dangling).unwrap();
    let answer = crate::test_daemon::Call::new(
        "status",
        serde_json::json!({"verify":true,"manifest":dangling}),
    )
    .answer(&daemon.daemon);
    assert!(answer.value().is_none(), "{answer:?}");
    assert!(!missing.exists());
}
