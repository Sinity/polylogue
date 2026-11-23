use super::*;

#[test]
fn a_flag_is_raised_once_and_lowered_idempotently() {
    let dir = tempfile::tempdir().unwrap();
    let session = SessionId::parse("claude-code:0199a1b2-3c4d-7e8f-9012-345678901234").unwrap();
    assert!(!raised(dir.path(), &session));
    raise(dir.path(), &session).unwrap();
    raise(dir.path(), &session).unwrap();
    assert!(raised(dir.path(), &session));
    assert!(
        flag(dir.path(), &session).ends_with("claude-code/0199a1b2-3c4d-7e8f-9012-345678901234")
    );
    lower(dir.path(), &session).unwrap();
    lower(dir.path(), &session).unwrap();
    assert!(!raised(dir.path(), &session));
}

#[test]
fn a_native_id_with_path_characters_stays_one_file_in_its_harness_directory() {
    let dir = tempfile::tempdir().unwrap();
    let mail = dir.path().join("mail");
    for native in ["../../outside", "a/b", "..", "50%:off", "zażółć"] {
        let session = SessionId::parse(&format!("claude-code:{native}")).unwrap();
        raise(&mail, &session).unwrap();
        assert_eq!(
            flag(&mail, &session).parent(),
            Some(mail.join("claude-code").as_path())
        );
        assert!(raised(&mail, &session));
        assert_eq!(flagged(&mail).unwrap(), std::slice::from_ref(&session));
        lower(&mail, &session).unwrap();
        assert!(flagged(&mail).unwrap().is_empty());
    }
    // Nothing was written beside the mailbox directory.
    let beside: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(beside, ["mail"]);
}

#[test]
fn only_files_the_mailbox_wrote_count_as_flags() {
    let dir = tempfile::tempdir().unwrap();
    assert!(flagged(&dir.path().join("absent")).unwrap().is_empty());
    let origin = dir.path().join("claude-code");
    std::fs::create_dir_all(&origin).unwrap();
    // Lowercase hex and raw punctuation are not names `flag` produces.
    for name in ["a%2fb", "a.b", "a b"] {
        std::fs::write(origin.join(name), "").unwrap();
    }
    assert!(flagged(dir.path()).unwrap().is_empty());
    std::fs::write(origin.join("a%2Fb"), "").unwrap();
    let session = SessionId::parse("claude-code:a/b").unwrap();
    assert_eq!(flagged(dir.path()).unwrap(), [session]);
}
