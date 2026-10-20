use super::*;

#[test]
fn incomplete_or_corrupt_lengths_preserve_all_existing_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("capture.json");
    let a = br#"{"observed":"a"}"#;
    publish(&path, a).unwrap();
    let accepted = fs::read(&path).unwrap();
    for tail in [
        MAGIC[..5].to_vec(),
        accepted[..MAGIC.len() + 10].to_vec(),
        {
            let mut corrupt = accepted.clone();
            let mut header = Header::of(a);
            header.length += 1;
            corrupt[MAGIC.len()..MAGIC.len() + HEADER_LEN].copy_from_slice(&header.encode());
            corrupt
        },
    ] {
        fs::write(&path, &tail).unwrap();
        assert!(publish(&path, br#"{"observed":"b"}"#).is_err());
        assert_eq!(fs::read(&path).unwrap(), tail);
    }
}
