use super::*;

#[test]
fn streaming_verifier_matches_slice_verification_for_every_chunk_size() {
    let body = b"synthetic capture";
    let header = Header::of(body);
    for bytes in [
        body.as_slice(),
        b"synthetic captur",
        b"synthetic capturX",
        b"synthetic capture extra",
        b"",
    ] {
        for size in [1, 3, 8192] {
            let mut verifier = header.verifier();
            for chunk in bytes.chunks(size) {
                verifier.update(chunk);
            }
            assert_eq!(verifier.verifies(), header.verifies(bytes));
        }
    }
}

#[test]
fn streaming_verifier_requires_the_complete_frame() {
    let header = Header::of(b"capture");
    assert!(!header.verifier().verifies());
    let mut verifier = header.verifier();
    verifier.update(b"cap");
    assert!(!verifier.verifies());
    assert!(Header::of(b"").verifier().verifies());
}
