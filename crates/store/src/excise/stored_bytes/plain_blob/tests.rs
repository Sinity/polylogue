use super::*;
use std::io;

#[test]
fn every_chunk_size_preserves_nonoverlapping_literal_matches() {
    for hay in [
        b"aaaaaaa".as_slice(),
        b"ababababa",
        b"quote\" and \\ escapes",
        "one ☺ and two ☺".as_bytes(),
        b"\xff\0\xff\0\xff",
        b"",
        b"xyz",
    ] {
        for start in 0..=hay.len() {
            for end in start..=hay.len() {
                let needle = &hay[start..end];
                let expected = super::super::find_all(hay, &[needle.to_vec()]);
                for chunk in 1..=hay.len() + 2 {
                    assert_eq!(
                        collected(hay, needle, chunk).unwrap(),
                        expected,
                        "hay={hay:?},needle={needle:?},chunk={chunk}"
                    );
                }
            }
        }
    }
}

#[test]
fn long_patterns_and_partial_or_failed_reads_keep_their_meaning() {
    let needle = vec![b'x'; 70_000];
    let mut hay = needle.clone();
    hay.push(b'y');
    hay.extend_from_slice(&needle);
    assert_eq!(
        collected(hay.as_slice(), &needle, 8192).unwrap(),
        [(0, 70_000), (70_001, 140_001)]
    );
    struct Fails(bool);
    impl Read for Fails {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.0 {
                return Err(io::Error::other("synthetic file failure"));
            }
            self.0 = true;
            out[0] = b'x';
            Ok(1)
        }
    }
    assert!(collected(Fails(false), b"x", 4).is_err());
    assert!(
        matches!(literal_ranges(b"xx".as_slice(),b"x",1,|_,_| Err(StoreError::NotFound("synthetic".into()))), Err(StoreError::NotFound(message)) if message=="synthetic")
    );
}

fn collected(reader: impl Read, needle: &[u8], chunk: usize) -> Result<Vec<(usize, usize)>> {
    let mut ranges = Vec::new();
    literal_ranges(reader, needle, chunk, |start, end| {
        ranges.push((start, end));
        Ok(())
    })?;
    Ok(ranges)
}
