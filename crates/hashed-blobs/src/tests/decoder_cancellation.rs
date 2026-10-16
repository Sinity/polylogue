use super::*;

#[test]
fn zero_output_blocks_poll_input_and_preserve_the_callers_typed_stop() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(dir.path(), vec![], 0o600).unwrap();
    let hash = store.put(&b""[..], 1).unwrap();
    let path = file_of(&store, &hash);
    let original = fs::read(&path).unwrap();
    assert_eq!(&original[4..6], &[0x24, 0]);
    let table = original.len() - 25;
    // A nonfinal empty raw zstd block is three zero bytes. Inserting these
    // does not change the decoded empty content or its existing checksum.
    let padding = 3 * 262144;
    let mut encoded = Vec::with_capacity(original.len() + padding);
    encoded.extend_from_slice(&original[..6]);
    encoded.resize(6 + padding, 0);
    encoded.extend_from_slice(&original[6..]);
    encoded[table + padding + 8..table + padding + 12]
        .copy_from_slice(&((table + padding) as u32).to_le_bytes());
    fs::write(path, encoded).unwrap();
    assert_eq!(store.verify(&hash).unwrap(), Verified::Intact);

    #[derive(Debug)]
    enum Stop {
        Storage(Error),
        Cancelled,
    }
    impl From<Error> for Stop {
        fn from(error: Error) -> Self {
            Self::Storage(error)
        }
    }
    let reader = Reader::new(dir.path(), vec![]);
    let mut polls = 0;
    let result = reader.verify_plain::<Stop>(&hash, 0, |chunk| {
        assert!(chunk.is_empty());
        polls += 1;
        if polls == 8 {
            Err(Stop::Cancelled)
        } else {
            Ok(())
        }
    });
    match result {
        Err(Stop::Cancelled) => {}
        Err(Stop::Storage(error)) => panic!("storage error replaced caller stop: {error}"),
        Ok(_) => panic!("decoder consumed the zero-output frame without polling its input"),
    }
    assert_eq!(polls, 8);
    assert!(reader.slot().unwrap().is_none());
    eprintln!("ZERO_OUTPUT_CANCEL compressed_padding={padding} decoded=0 callback_polls={polls} no_retained_proof=true");
}
