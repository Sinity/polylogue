use super::*;

fn blanked(bytes: &[u8], ranges: &[(u64, u64)]) -> Vec<u8> {
    let mut out = bytes.to_vec();
    for &(start, end) in ranges {
        out[start as usize..end as usize].fill(b' ');
    }
    out
}

#[test]
fn every_chunk_boundary_preserves_json_spans_and_other_values() {
    let bytes = br#"{"secret":"not a value match","v":"x s\u0065cr\u0065t y","keep":"unchanged"}"#;
    let expected =
        br#"{"secret":"not a value match","v":"x                  y","keep":"unchanged"}"#;
    for split in 0..=bytes.len() {
        let mut scan = TextMatches::new("secret").unwrap();
        scan.feed(&bytes[..split]).unwrap();
        scan.feed(&bytes[split..]).unwrap();
        let ranges = scan.finish().unwrap();
        let actual = blanked(bytes, &ranges);
        assert_eq!(actual, expected, "split {split}");
        let value: serde_json::Value = serde_json::from_slice(&actual).unwrap();
        assert_eq!(value["keep"], "unchanged");
    }
}

#[test]
fn mixed_case_escapes_surrogates_and_utf8_match_identically() {
    for bytes in [
        r#"{"v":"a\u00E9\uD83D\uDE00z"}"#.as_bytes(),
        "{\"v\":\"aé😀z\"}".as_bytes(),
    ] {
        for chunk in 1..=bytes.len() {
            let mut scan = TextMatches::new("é😀").unwrap();
            for part in bytes.chunks(chunk) {
                scan.feed(part).unwrap();
            }
            let ranges = scan.finish().unwrap();
            assert_eq!(ranges.len(), 1);
            let value: serde_json::Value =
                serde_json::from_slice(&blanked(bytes, &ranges)).unwrap();
            assert!(value["v"].as_str().unwrap().starts_with('a'));
            assert!(value["v"].as_str().unwrap().ends_with('z'));
        }
    }
}

#[test]
fn strings_do_not_match_across_value_boundaries_and_keys_stay_exact() {
    let bytes = br#"{"secret":"se","v":"cret","w":"secret"}"#;
    let ranges = in_bytes(bytes, "secret").unwrap();
    assert_eq!(ranges.len(), 1);
    assert_eq!(
        blanked(bytes, &ranges),
        br#"{"secret":"se","v":"cret","w":"      "}"#
    );
}

#[test]
fn escaping_quotes_backslashes_and_controls_keeps_valid_json() {
    for value in ["a\"b", "a\\b", "a\nb", "a\tb"] {
        let bytes =
            serde_json::to_vec(&serde_json::json!({"v": format!("before {value} after")})).unwrap();
        let ranges = in_bytes(&bytes, value).unwrap();
        let value: serde_json::Value = serde_json::from_slice(&blanked(&bytes, &ranges)).unwrap();
        assert!(value["v"].as_str().unwrap().starts_with("before "));
        assert!(value["v"].as_str().unwrap().ends_with(" after"));
    }
}

#[test]
fn plain_text_handles_overlapping_matches_and_empty_patterns_refuse() {
    assert_eq!(in_bytes(b"ababa", "aba").unwrap(), [(0, 3), (2, 5)]);
    assert!(TextMatches::new("").is_err());
    assert!(in_bytes(br#"{"v":"\uD800"}"#, "secret").is_err());
    assert!(in_bytes(&[0xff], "secret").is_err());
}

#[test]
fn json_text_inside_json_strings_keeps_original_source_spans() {
    let payload = r#"{"text":"s\u0065cret","keep":"unchanged"}"#;
    let bytes =
        serde_json::to_vec(&serde_json::json!({"payload": payload, "other": "exact"})).unwrap();
    for split in 0..=bytes.len() {
        let mut scan = TextMatches::new("secret").unwrap();
        scan.feed(&bytes[..split]).unwrap();
        scan.feed(&bytes[split..]).unwrap();
        let ranges = scan.finish().unwrap();
        assert_eq!(ranges.len(), 1, "split {split}");
        let outer: serde_json::Value = serde_json::from_slice(&blanked(&bytes, &ranges)).unwrap();
        let inner: serde_json::Value =
            serde_json::from_str(outer["payload"].as_str().unwrap()).unwrap();
        assert_eq!(inner["keep"], "unchanged");
        assert_eq!(outer["other"], "exact");
        assert!(!inner["text"].as_str().unwrap().contains("secret"));
    }
}

#[test]
fn replacement_cannot_create_another_match_at_the_span_edge() {
    let needle = "ab ";
    let fill = replacement(needle).unwrap();
    assert!(!needle.as_bytes().contains(&fill));
    let mut bytes = b"abab ".to_vec();
    let ranges = in_bytes(&bytes, needle).unwrap();
    for (start, end) in ranges {
        bytes[start as usize..end as usize].fill(fill);
    }
    assert!(in_bytes(&bytes, needle).unwrap().is_empty());
}

#[test]
fn generated_escape_spellings_agree_with_serde_in_plain_and_embedded_values() {
    let target = "secret😀é";
    let chars: Vec<char> = target.chars().collect();
    for mask in 0..(1 << chars.len()) {
        let mut spelling = String::new();
        for (n, ch) in chars.iter().enumerate() {
            if mask & (1 << n) == 0 {
                spelling.push(*ch);
            } else {
                for unit in ch.encode_utf16(&mut [0u16; 2]) {
                    spelling.push_str(&format!("\\u{unit:04X}"));
                }
            }
        }
        let source = format!("{{\"keep\":\"exact\",\"v\":\"before {spelling} after\"}}");
        let original: serde_json::Value = serde_json::from_str(&source).unwrap();
        assert_eq!(original["v"], format!("before {target} after"));
        for embedded in [false, true] {
            let bytes = if embedded {
                serde_json::to_vec(&serde_json::json!({"payload": source})).unwrap()
            } else {
                source.as_bytes().to_vec()
            };
            for chunk in [1, 2, 5, 7, 16, 64] {
                let mut scan = TextMatches::new(target).unwrap();
                for bytes in bytes.chunks(chunk) {
                    scan.feed(bytes).unwrap();
                }
                let ranges = scan.finish().unwrap();
                assert_eq!(
                    ranges.len(),
                    1,
                    "mask {mask}, embedded {embedded}, chunk {chunk}"
                );
                let out = blanked(&bytes, &ranges);
                assert_eq!(out.len(), bytes.len());
                let value: serde_json::Value = serde_json::from_slice(&out).unwrap();
                let value = if embedded {
                    serde_json::from_str(value["payload"].as_str().unwrap()).unwrap()
                } else {
                    value
                };
                assert_eq!(value["keep"], "exact");
                let value = value["v"].as_str().unwrap();
                assert!(value.starts_with("before ") && value.ends_with(" after"));
                assert!(!value.contains(target));
            }
        }
    }
}
