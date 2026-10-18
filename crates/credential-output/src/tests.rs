use super::*;
fn masked(parts: &[&str]) -> String {
    let mut masker = Masker::new();
    let mut out = Vec::new();
    for text in parts {
        masker.push(text, &mut out).unwrap();
    }
    masker.finish(&mut out).unwrap();
    String::from_utf8(out).unwrap()
}
fn normalized(text: &str) -> String {
    let mut rest = text;
    let mut out = String::new();
    while let Some(start) = rest.find("[credential: ") {
        out.push_str(&rest[..start]);
        out.push_str("<masked>");
        let end = rest[start..].find(", masked]").unwrap() + start + ", masked]".len();
        rest = &rest[end..];
    }
    out.push_str(rest);
    out
}
/// Feeds `input` in byte chunks the way a UTF-8 decoder would: only complete
/// characters reach `push`, with at most three bytes carried between chunks.
fn byte_chunks(input: &str, chunks: &[usize]) -> String {
    let mut masker = Masker::new();
    let mut out = Vec::new();
    let mut offset = 0;
    let mut buffered = Vec::new();
    for &size in chunks {
        let end = (offset + size).min(input.len());
        buffered.extend_from_slice(&input.as_bytes()[offset..end]);
        offset = end;
        let valid = match std::str::from_utf8(&buffered) {
            Ok(s) => s.len(),
            Err(e) => {
                assert!(e.error_len().is_none());
                e.valid_up_to()
            }
        };
        masker
            .push(std::str::from_utf8(&buffered[..valid]).unwrap(), &mut out)
            .unwrap();
        buffered.drain(..valid);
        assert!(buffered.len() <= 3);
    }
    assert_eq!(offset, input.len());
    assert!(buffered.is_empty());
    masker.finish(&mut out).unwrap();
    String::from_utf8(out).unwrap()
}
#[test]
fn conformance_cases_mask_the_same_at_every_split() {
    let corpus: serde_json::Value = serde_json::from_str(include_str!(
        "../tests/fixtures/credential-output-v1/conformance.json"
    ))
    .unwrap();
    assert_eq!(corpus["catalog_version"], "credential-output-v1");
    assert_eq!(corpus["fixture_encoding"], "text-parts-v1");
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 132);
    for case in cases {
        let text: String = case["text_parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|part| part.as_str().unwrap())
            .collect();
        let text = text.as_str();
        let id = case["id"].as_str().unwrap();
        let mut expected = String::new();
        let mut offset = 0;
        for region in case["regions_utf8"].as_array().unwrap() {
            let a = region[0].as_u64().unwrap() as usize;
            let b = region[1].as_u64().unwrap() as usize;
            expected.push_str(&text[offset..a]);
            expected.push_str("<masked>");
            offset = b;
        }
        expected.push_str(&text[offset..]);
        for cut in 0..=text.len() {
            assert_eq!(
                normalized(&byte_chunks(text, &[cut, text.len() - cut])),
                expected,
                "{id}, byte cut {cut}"
            );
        }
        for width in 1..=8 {
            let chunks: Vec<_> = (0..text.len())
                .step_by(width)
                .map(|i| width.min(text.len() - i))
                .collect();
            assert_eq!(
                normalized(&byte_chunks(text, &chunks)),
                expected,
                "{id}, chunk width {width}"
            );
        }
    }
}
#[test]
fn word_boundaries_follow_python_letters_digits_and_whitespace() {
    let token = format!("ghp_{}", "A".repeat(36));
    for prefix in ['\u{345}', '\u{301}', '\u{203f}', '\u{200d}'] {
        assert_eq!(
            normalized(&masked(&[&format!("{prefix}{token}")])),
            format!("{prefix}<masked>")
        );
    }
    for prefix in ['π', '雪', 'Ⅷ', '²', '_', '9'] {
        let text = format!("{prefix}{token}");
        assert_eq!(masked(&[&text]), text);
    }
    for prefix in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{85}', '\u{a0}'] {
        assert_eq!(
            normalized(&masked(&[&format!("{prefix}--secret=x")])),
            format!("{prefix}--secret=<masked>")
        );
    }
}
#[test]
fn cli_flags_mask_only_their_values() {
    for (input, expected) in [
        (
            "run --secret=x --user iris --key /home/iris/id.pub",
            "run --secret=<masked> --user iris --key /home/iris/id.pub",
        ),
        ("curl --user='a:b'", "curl --user=<masked>"),
        ("--secret='' --token=\"\"", "--secret='' --token=\"\""),
        (
            "--secret=\"outer --secret=inner",
            "--secret=\"outer --secret=<masked>",
        ),
        (
            "token=ast_node secret=chapter user@example.invalid",
            "token=ast_node secret=chapter user@example.invalid",
        ),
    ] {
        assert_eq!(normalized(&masked(&[input])), expected);
    }
}
#[test]
fn overlapping_matches_merge_and_adjacent_matches_stay_separate() {
    let input = format!("xoxb-A-ghp_{}!tail", "A".repeat(36));
    assert_eq!(normalized(&masked(&[&input])), "<masked>!tail");
    let input = format!(
        "-----BEGIN PRIVATE KEY-----x-----END PRIVATE KEY-----ghp_{}",
        "A".repeat(36)
    );
    assert_eq!(normalized(&masked(&[&input])), "<masked><masked>");
    let input = format!("https://u:ghp_{}!private@host/path", "A".repeat(36));
    assert_eq!(normalized(&masked(&[&input])), "https://<masked>@host/path");
    let crossing =
        "-----BEGIN PRIVATE KEY----- --secret=\"inside -----END PRIVATE KEY----- after\" tail";
    // Expected regions: PEM [0,70) and CLI secret [37,77) merge into [0,77).
    assert_eq!(&crossing[77..], " tail");
    for cut in 0..=crossing.len() {
        assert_eq!(
            normalized(&masked(&[&crossing[..cut], &crossing[cut..]])),
            "<masked> tail",
            "cut {cut}"
        );
    }
    let unfinished = format!("https://u:ghp_{}", "A".repeat(36));
    assert_eq!(normalized(&masked(&[&unfinished])), "https://u:<masked>");
}
#[test]
fn block_boundary_ends_token_matches_but_not_pem_blocks() {
    let mut m = Masker::new();
    let mut out = Vec::new();
    m.push("short ghp_", &mut out).unwrap();
    m.block_boundary(&mut out).unwrap();
    m.push(
        "-----BEGIN VENDOR ENCRYPTED PRIVATE KEY-----body-----END VEND",
        &mut out,
    )
    .unwrap();
    m.block_boundary(&mut out).unwrap();
    m.push("OR ENCRYPTED PRIVATE KEY----- visible", &mut out)
        .unwrap();
    m.finish(&mut out).unwrap();
    assert_eq!(
        normalized(std::str::from_utf8(&out).unwrap()),
        "short ghp_<masked> visible"
    );
    for cut in 0..=40 {
        let raw = format!("ghp_{}", "A".repeat(36));
        let mut m = Masker::new();
        let mut out = Vec::new();
        m.push(&raw[..cut], &mut out).unwrap();
        m.block_boundary(&mut out).unwrap();
        m.push(&raw[cut..], &mut out).unwrap();
        m.finish(&mut out).unwrap();
        let expected = if cut == 0 || cut == 40 {
            "<masked>".to_owned()
        } else {
            raw.clone()
        };
        assert_eq!(
            normalized(std::str::from_utf8(&out).unwrap()),
            expected,
            "block cut {cut}"
        );
    }
}
#[test]
fn long_secrets_stream_without_growing_state() {
    for begin in [
        format!("ghp_{}", "A".repeat(36)),
        "Bearer AAAAAAAAAAAAAAAAAAAA".into(),
        "--secret=x".into(),
        "-----BEGIN VENDOR PRIVATE KEY-----".into(),
    ] {
        let mut m = Masker::new();
        let mut out = Vec::new();
        m.push(&begin, &mut out).unwrap();
        let initial = out.len();
        for _ in 0..256 {
            m.push(&"A".repeat(4096), &mut out).unwrap();
            assert!(m.pending.len() <= CARRY);
            assert!(m.candidates.len() <= MAX_CANDIDATES);
            assert!(m.headers.iter().all(|h| h.text.len() <= CARRY));
        }
        m.finish(&mut out).unwrap();
        assert_eq!(out.len(), initial, "{begin}");
    }
    // A URL scheme is never masked, so a long plain word is not held back.
    let ordinary = "Z".repeat(4096);
    assert_eq!(masked(&[&ordinary]), ordinary);
}
#[test]
fn overlong_undecided_text_is_refused_unwritten_and_closes_the_stream() {
    for (prefix, visible) in [
        ("before https://u:", "before https://"),
        ("before eyJA", "before "),
        ("before SG.A", "before "),
        ("before --secret=\"x", "before "),
        ("before -----BEGIN VENDOR ", "before "),
    ] {
        let mut m = Masker::new();
        let mut out = Vec::new();
        m.push(prefix, &mut out).unwrap();
        let error = m.push(&"A".repeat(CARRY), &mut out).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(std::str::from_utf8(&out).unwrap(), visible, "{prefix}");
        let calls = out.len();
        assert!(m.push("later", &mut out).is_err());
        assert!(m.block_boundary(&mut out).is_err());
        assert!(m.finish(&mut out).is_err());
        assert_eq!(out.len(), calls);
    }
}
#[test]
fn too_many_open_matches_are_refused() {
    let mut m = Masker::new();
    let mut out = Vec::new();
    m.push("-----BEGIN PRIVATE KEY-----", &mut out).unwrap();
    let error = m
        .push(&"eyJA-".repeat(MAX_CANDIDATES + 1), &mut out)
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(m.candidates.len() <= MAX_CANDIDATES + 1);
    assert_eq!(normalized(std::str::from_utf8(&out).unwrap()), "<masked>");
}
#[test]
fn sink_errors_are_returned_unchanged_and_close_the_stream() {
    #[derive(Debug)]
    struct Original;
    impl std::fmt::Display for Original {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("original")
        }
    }
    impl std::error::Error for Original {}
    struct Broken {
        calls: usize,
    }
    impl Write for Broken {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            self.calls += 1;
            Err(io::Error::other(Original))
        }
        fn flush(&mut self) -> io::Result<()> {
            panic!("no flush")
        }
    }
    for operation in [0, 1, 2, 3] {
        let mut m = Masker::new();
        let mut sink = Broken { calls: 0 };
        let error = match operation {
            0 => m.push("visible ", &mut sink).unwrap_err(),
            1 => {
                m.push("ghp_", &mut sink).unwrap();
                m.finish(&mut sink).unwrap_err()
            }
            2 => {
                m.push("ghp_", &mut sink).unwrap();
                m.block_boundary(&mut sink).unwrap_err()
            }
            _ => m
                .push(&format!("ghp_{}", "A".repeat(36)), &mut sink)
                .unwrap_err(),
        };
        assert!(error.get_ref().unwrap().is::<Original>());
        assert!(m.push("later", &mut sink).is_err());
        assert!(m.finish(&mut sink).is_err());
        assert!(m.block_boundary(&mut sink).is_err());
        assert_eq!(sink.calls, 1);
    }
}
