use crate::daemon::Request;
use polylogue_store::verify::Verification;

#[test]
fn verify_through_the_daemon_checks_the_archive_and_writes_a_new_manifest() {
    let daemon = crate::test_daemon::start("words to verify");
    let manifest = daemon._dir.path().join("manifest.tsv");
    let request = Request::Verify {
        manifest: Some(manifest.clone()),
    };
    assert!(request.read_only());
    assert!(crate::daemon::execution::request_deadline(&request, None).is_none());
    let socket = daemon.daemon.archive.socket();
    let answer = crate::client::send(&socket, &request).unwrap();
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
    let again = crate::client::send(&socket, &request).unwrap();
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
    let answer = crate::client::send(
        &daemon.daemon.archive.socket(),
        &Request::Verify {
            manifest: Some("relative.tsv".into()),
        },
    )
    .unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Error(error),
        ..
    } = &answer
    else {
        panic!("{answer:?}");
    };
    assert!(error.message.contains("must be absolute"), "{answer:?}");
}
