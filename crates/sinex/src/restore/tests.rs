use super::*;

#[test]
fn positions_must_run_from_one_to_the_last_without_a_gap() {
    let mut positions = Positions::default();
    assert_eq!(positions.complete().unwrap(), 0);
    for position in [3, 1, 2, 2, 130, 65] {
        positions.insert(position).unwrap();
    }
    assert!(matches!(
        positions.complete(),
        Err(Error::Invalid(message)) if message.contains("5 of batches 1 to 130")
    ));
    for position in 4..=130 {
        positions.insert(position).unwrap();
    }
    assert_eq!(positions.complete().unwrap(), 130);
    assert!(positions.insert(0).is_err());
}

#[test]
fn checked_bytes_must_match_their_name() {
    let bytes = b"synthetic batch";
    let mut good = Checked::new(&bytes[..], hashed_blobs::Hash::of(bytes));
    std::io::copy(&mut good, &mut std::io::sink()).unwrap();
    good.finish().unwrap();
    // Bytes left unread still count.
    Checked::new(&bytes[..], hashed_blobs::Hash::of(bytes))
        .finish()
        .unwrap();
    let wrong = Checked::new(&bytes[..], hashed_blobs::Hash::of(b"other"));
    assert!(matches!(wrong.finish(), Err(Error::Invalid(_))));
}
