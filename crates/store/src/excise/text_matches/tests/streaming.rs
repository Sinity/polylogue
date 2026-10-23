use super::*;
use std::collections::BTreeSet;

fn streamed(bytes: &[u8], needle: &str, chunk: usize) -> Result<Vec<(u64, u64)>> {
    let mut scan = TextMatches::new(needle)?;
    let mut raw = BTreeSet::new();
    let mut accepted = BTreeSet::new();
    let mut collect = |batch| match batch {
        Matches::Undecided(ranges) => raw.extend(ranges),
        Matches::Plain(ranges) => {
            accepted.append(&mut raw);
            accepted.extend(ranges);
        }
        Matches::Json(ranges) => {
            raw.clear();
            accepted.extend(ranges);
        }
    };
    for part in bytes.chunks(chunk) {
        scan.feed(part)?;
        collect(scan.drain());
    }
    collect(scan.complete()?);
    Ok(accepted.into_iter().collect())
}

#[test]
fn tentative_whitespace_is_discarded_for_json_and_kept_for_plain_or_eof() {
    for tail in [b"{}".as_slice(), b"plain", b""] {
        let mut bytes = vec![b' '; 128 * 1024];
        bytes.extend(tail);
        let actual = streamed(&bytes, " ", 73).unwrap();
        assert_eq!(actual.len(), if tail == b"{}" { 0 } else { 128 * 1024 });
        assert_eq!(actual, in_bytes(&bytes, " ").unwrap());
    }
}

#[test]
fn streaming_preserves_keys_incomplete_values_nested_json_and_overlaps() {
    let cases: Vec<(&[u8], &str)> = vec![
        (br#"{"secret":"value"}"#, "secret"),
        (br#"{"v":"secret","unfinished":"\u00"#, "secret"),
        (br#"{"v":"{\"v\":\"s\\u0065cret\"}","keep":"x"}"#, "secret"),
        (br#"{"v":"{\"v\":\"s\\u0065cret\""}"#, "secret"),
        (br#"{"v":"{\"v\":\"s\\u0065cret\u00"}"#, "secret"),
        (br#"{"v":"secret"      :"value"}"#, "secret"),
        (br#"{"v":"secret"      ,"w":"secret"}"#, "secret"),
        (br#"{"v":"secret"      "#, "secret"),
        (b"ababa", "aba"),
        (br#"{"v":"ababa"}"#, "aba"),
        (r#"{"v":"é\uD83D\uDE00"}"#.as_bytes(), "é😀"),
    ];
    for (bytes, needle) in cases {
        let reference = in_bytes(bytes, needle);
        for chunk in 1..=bytes.len().max(1) {
            let actual = streamed(bytes, needle, chunk);
            match &reference {
                Ok(reference) => {
                    assert_eq!(actual.unwrap(), *reference, "chunk {chunk}: {bytes:?}")
                }
                Err(_) => assert!(actual.is_err(), "chunk {chunk}: {bytes:?}"),
            }
        }
    }
}

#[test]
fn completed_values_do_not_accumulate_with_chain_length() {
    let mut scan = TextMatches::new("secret").unwrap();
    scan.feed(b"[").unwrap();
    let mut count = 0;
    for _ in 0..20000 {
        scan.feed(br#""secret","#).unwrap();
        let Matches::Json(ranges) = scan.drain() else {
            panic!("JSON selected")
        };
        count += ranges.len();
        assert!(scan.decoded_ranges.is_empty());
        assert!(scan.raw_ranges.is_empty());
    }
    scan.feed(b"]").unwrap();
    let Matches::Json(ranges) = scan.complete().unwrap() else {
        panic!("JSON selected")
    };
    count += ranges.len();
    assert_eq!(count, 20000);
}
