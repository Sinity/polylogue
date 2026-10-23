use super::*;

fn streamed(bytes: &[u8], needle: &str, chunk: usize) -> Result<Vec<(u64, u64)>> {
    let mut scan = TextMatches::new(needle)?;
    for part in bytes.chunks(chunk) {
        scan.feed(part)?;
    }
    scan.finish()
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
fn a_pending_key_waits_through_long_whitespace_but_a_value_can_commit() {
    let value = "secret".repeat(2000);
    for suffix in [":0]", ",0]", ""] {
        let source = format!("[\"{value}\"{}{}", " ".repeat(128 * 1024), suffix);
        let actual = streamed(source.as_bytes(), "secret", 137);
        if suffix.starts_with(':') {
            assert!(actual.is_err());
        } else {
            assert_eq!(actual.unwrap().len(), 2000);
        }
    }
}

#[test]
fn nested_only_hits_require_successful_inner_completion() {
    for (inner, expected) in [
        (r#"{"v":"s\u0065cret"}"#, 1),
        (r#"{"v":"s\u0065cret","later":"unterminated}"#, 0),
    ] {
        let source = serde_json::to_vec(&serde_json::json!({"payload":inner})).unwrap();
        for chunk in [1, 7, 64] {
            assert_eq!(streamed(&source, "secret", chunk).unwrap().len(), expected);
        }
    }
    let source =
        serde_json::to_vec(&serde_json::json!({"payload":r#"{"secret":"value"}"#})).unwrap();
    assert!(streamed(&source, "secret", 1).is_err());
}
