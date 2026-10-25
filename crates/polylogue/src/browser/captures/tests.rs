use super::*;

fn frame(bytes: &[u8]) -> Vec<u8> {
    let mut frame = Header::of(bytes).encode();
    frame.extend(bytes);
    frame
}

#[test]
fn retries_complete_partial_headers_and_bodies_without_changing_accepted_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.json");
    let a = br#"{"observed":"accepted-a"}"#;
    let b = br#"{"observed":"pending-b"}"#;
    append(&path, a).unwrap();
    let accepted = fs::read(&path).unwrap();
    let pending = frame(b);
    let mut expected = accepted.clone();
    expected.extend(&pending);
    for cut in [
        1,
        20,
        21,
        HEADER_LEN - 1,
        HEADER_LEN,
        HEADER_LEN + 1,
        pending.len() - 1,
    ] {
        let mut interrupted = accepted.clone();
        interrupted.extend(&pending[..cut]);
        fs::write(&path, &interrupted).unwrap();
        assert!(append(&path, b).unwrap(), "cut {cut}");
        let completed = fs::read(&path).unwrap();
        assert!(completed.starts_with(&interrupted), "cut {cut}");
        assert_eq!(completed, expected, "cut {cut}");
        assert!(!append(&path, b).unwrap());
        assert_eq!(fs::read(&path).unwrap(), expected);
    }
}

#[test]
fn initial_magic_prefix_can_complete_without_discarding_any_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.json");
    let a = br#"{"observed":"a"}"#;
    let mut expected = MAGIC.to_vec();
    expected.extend(frame(a));
    for cut in 1..MAGIC.len() {
        fs::write(&path, &MAGIC[..cut]).unwrap();
        assert!(append(&path, a).unwrap());
        assert_eq!(fs::read(&path).unwrap(), expected);
        assert!(!append(&path, a).unwrap());
    }
}

#[test]
fn differing_tails_and_corrupt_accepted_lengths_preserve_all_existing_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.json");
    let a = br#"{"observed":"a"}"#;
    let b = br#"{"observed":"b"}"#;
    append(&path, a).unwrap();
    let accepted = fs::read(&path).unwrap();
    let pending = frame(a);
    for tail in [
        b"POLYX".to_vec(),
        {
            let mut bytes = accepted.clone();
            bytes.extend(&pending[..HEADER_LEN - 1]);
            bytes
        },
        {
            let mut bytes = accepted.clone();
            bytes.extend(&pending[..pending.len() - 1]);
            bytes
        },
        {
            let mut bytes = accepted.clone();
            let mut header = Header::of(a);
            header.length += 1;
            bytes[MAGIC.len()..MAGIC.len() + HEADER_LEN].copy_from_slice(&header.encode());
            bytes
        },
    ] {
        fs::write(&path, &tail).unwrap();
        assert!(append(&path, b).is_err());
        assert_eq!(fs::read(&path).unwrap(), tail);
    }
    let mut corrupt = accepted.clone();
    let mut header = Header::of(a);
    header.length += 1;
    corrupt[MAGIC.len()..MAGIC.len() + HEADER_LEN].copy_from_slice(&header.encode());
    fs::write(&path, &corrupt).unwrap();
    assert!(append(&path, a).is_err());
    assert_eq!(fs::read(&path).unwrap(), corrupt);
}

#[test]
fn a_late_body_mismatch_refuses_before_writing_any_missing_suffix() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.json");
    append(&path, br#"{"observed":"accepted"}"#).unwrap();
    let accepted = fs::read(&path).unwrap();
    let body = format!("{{\"observed\":\"{}\"}}", "x".repeat(20_000));
    let pending = frame(body.as_bytes());
    let mut interrupted = accepted.clone();
    interrupted.extend(&pending[..HEADER_LEN + 9_000]);
    let mut mismatched = interrupted.clone();
    mismatched[accepted.len() + HEADER_LEN + 8_500] = b'y';
    fs::write(&path, &mismatched).unwrap();
    assert!(append(&path, body.as_bytes()).is_err());
    assert_eq!(fs::read(&path).unwrap(), mismatched);
    fs::write(&path, &interrupted).unwrap();
    assert!(append(&path, body.as_bytes()).unwrap());
    let mut expected = accepted;
    expected.extend(pending);
    assert_eq!(fs::read(&path).unwrap(), expected);
}
