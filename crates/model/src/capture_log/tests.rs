use super::*;
use std::io::Cursor;

#[test]
fn streaming_verifier_matches_slice_verification_and_stops_at_the_frame_end() {
    let body = b"synthetic capture";
    let header = Header::of(body);
    let mut reader = Cursor::new([body.as_slice(), b"next frame"].concat());
    let mut verifier = header.verifier(&mut reader);
    let mut bytes = [0; 3];
    while verifier.read(&mut bytes).unwrap() != 0 {}
    assert!(verifier.verifies());
    assert!(header.verifies(body));
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, b"next frame");

    for bytes in [b"synthetic captur".as_slice(), b"synthetic capturX"] {
        let mut verifier = header.verifier(bytes);
        io::copy(&mut verifier, &mut io::sink()).unwrap();
        assert_eq!(verifier.verifies(), header.verifies(bytes));
        assert!(!header.verifies(bytes));
    }
    assert!(!header.verifier(body.as_slice()).verifies());
    assert!(Header::of(b"").verifier(b"".as_slice()).verifies());
}

#[test]
fn streaming_verifier_preserves_input_errors() {
    let header = Header::of(b"capture");
    let mut input = header.verifier(io::repeat(0).take(1).chain(Fails));
    let mut bytes = Vec::new();
    assert_eq!(
        input.read_to_end(&mut bytes).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    assert!(!input.verifies());
}

struct Fails;
impl Read for Fails {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::from(io::ErrorKind::PermissionDenied))
    }
}
