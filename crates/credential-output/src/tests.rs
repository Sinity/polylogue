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
            assert!(m.headers.iter().all(|h| h.label.len() <= CARRY));
        }
        m.finish(&mut out).unwrap();
        assert_eq!(out.len(), initial, "{begin}");
    }
    // A URL scheme is never masked, so a long plain word is not held back.
    let ordinary = "Z".repeat(4096);
    assert_eq!(masked(&[&ordinary]), ordinary);
}
#[test]
fn undecided_text_past_the_window_is_masked_and_streaming_continues() {
    let filler = "A".repeat(CARRY);
    for (prefix, expected) in [
        ("before https://u:", "before https://<masked> after"),
        ("before eyJA", "before <masked> after"),
        ("before SG.A", "before <masked> after"),
        // A quoted value runs to its closing quote, so the rest is masked.
        ("before --secret=\"x", "before --secret=<masked>"),
        // Spaces may continue a PEM label; the lowercase `a` ends it.
        ("before -----BEGIN VENDOR ", "before <masked>after"),
    ] {
        let mut m = Masker::new();
        let mut out = Vec::new();
        m.push(prefix, &mut out).unwrap();
        m.push(&filler, &mut out).unwrap();
        m.push(" after", &mut out).unwrap();
        m.finish(&mut out).unwrap();
        let text = String::from_utf8(out).unwrap();
        assert_eq!(normalized(&text), expected, "{prefix}");
        assert!(!text.contains("AAAA"), "{prefix}");
    }
}
#[test]
fn long_jwt_is_masked_whole() {
    let payload = "eyJzdWIiOiJ0ZXN0In0".repeat(40);
    assert!(payload.len() > CARRY);
    let input = format!("token eyJhbGciOiJIUzI1NiJ9.{payload}.c2lnbmF0dXJl rest");
    assert_eq!(normalized(&masked(&[&input])), "token <masked> rest");
    let unfinished = format!("token eyJhbGciOiJIUzI1NiJ9{payload} rest");
    assert_eq!(normalized(&masked(&[&unfinished])), "token <masked> rest");
}
#[test]
fn long_url_password_is_masked_whole() {
    let password = "p4ss".repeat(150);
    let input = format!("see https://user:{password}@example.invalid/path rest");
    let output = masked(&[&input]);
    assert!(!output.contains("p4ss"));
    assert_eq!(normalized(&output), "see https://<masked>/path rest");
    // A short one keeps the host visible.
    assert_eq!(
        normalized(&masked(&["see https://user:p4ss@example.invalid/path"])),
        "see https://<masked>@example.invalid/path"
    );
}
#[test]
fn long_ordinary_base64_is_not_masked() {
    let mut rng = Rng(7);
    let alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let body: String = (0..8192)
        .map(|_| alphabet[rng.below(alphabet.len())] as char)
        .collect();
    for input in [
        format!("data:image/png;base64,{body}== after"),
        format!("{body}\n{body}"),
    ] {
        assert_eq!(masked(&[&input]), input);
    }
}
#[test]
fn long_pem_label_masks_to_the_end_of_the_string() {
    let label = "LONG ".repeat(CARRY / 4);
    let input =
        format!("-----BEGIN {label}PRIVATE KEY-----body-----END {label}PRIVATE KEY----- after");
    assert_eq!(normalized(&masked(&[&input])), "<masked>");
    // A long header that turns out not to be a private key stays masked.
    let input = format!("-----BEGIN {label}PUBLIC KEY----- after");
    assert_eq!(normalized(&masked(&[&input])), "<masked>----- after");
}
#[test]
fn forced_matches_stay_bounded() {
    let mut m = Masker::new();
    let mut out = Vec::new();
    m.push("-----BEGIN PRIVATE KEY-----", &mut out).unwrap();
    for _ in 0..8 {
        m.push(&"eyJA-".repeat(MAX_CANDIDATES), &mut out).unwrap();
        assert!(m.candidates.len() <= 64, "{}", m.candidates.len());
        assert!(m.pending.len() <= CARRY + 1);
    }
    m.push(
        &format!("--secret=\"{}", "--token=y ".repeat(10_000)),
        &mut out,
    )
    .unwrap();
    assert!(m.candidates.len() <= 64, "{}", m.candidates.len());
    m.finish(&mut out).unwrap();
    assert_eq!(normalized(std::str::from_utf8(&out).unwrap()), "<masked>");
}
/// Small deterministic generator for the randomized tests.
struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 % n as u64) as usize
    }
}
/// Random text built from credential fragments, separators and long runs.
fn random_text(rng: &mut Rng) -> String {
    const PIECES: &[&str] = &[
        " ",
        "\n",
        "-",
        ".",
        ":",
        "@",
        "/",
        "\"",
        "'",
        "=",
        "A",
        "z",
        "9",
        "é",
        "雪",
        "eyJ",
        ".eyJ",
        "SG.",
        "ghp_",
        "https://",
        "u:",
        "--secret=",
        "--user ",
        "Bearer ",
        "Authorization: Basic ",
        "aws_secret=",
        "AKIA",
        "-----BEGIN PRIVATE KEY-----",
        "-----END PRIVATE KEY-----",
        "-----BEGIN RSA ",
    ];
    let mut text = String::new();
    for _ in 0..rng.below(60) {
        if rng.below(8) == 0 {
            let run = ["A", "b", "0", " ", "-"][rng.below(5)];
            text.push_str(&run.repeat(rng.below(3 * CARRY)));
        } else {
            text.push_str(PIECES[rng.below(PIECES.len())]);
        }
    }
    text
}
#[test]
fn output_does_not_depend_on_chunking() {
    let mut rng = Rng(0x5eed);
    for _ in 0..100 {
        let text = random_text(&mut rng);
        let whole = masked(&[&text]);
        for _ in 0..3 {
            let mut chunks = Vec::new();
            let mut left = text.len();
            while left > 0 {
                let size = 1 + rng.below(left.min(700));
                chunks.push(size);
                left -= size;
            }
            assert_eq!(byte_chunks(&text, &chunks), whole, "{text:?} {chunks:?}");
        }
    }
}
#[test]
fn embedded_credentials_never_leak() {
    let mut rng = Rng(0xc0ffee);
    for _ in 0..150 {
        let secret = "Zk9PQkFS".repeat(5 + rng.below(200));
        let credential = match rng.below(5) {
            0 => format!("eyJhbGciOiJub25lIn0.eyJ{secret}.{secret}"),
            1 => format!("https://user:{secret}@example.invalid/"),
            2 => format!("ghp_{secret}"),
            3 => format!("--secret=\"{secret} with spaces\""),
            _ => format!("Bearer {secret}"),
        };
        // An unclosed quote before the credential can end at its opening
        // quote, as in the catalog's reference matcher, so quotes in the
        // surrounding text are only allowed across a line break.
        let (before, after) = (random_text(&mut rng), random_text(&mut rng));
        let text = if rng.below(2) == 0 {
            let unquoted = |s: &str| s.replace(['"', '\''], "");
            format!("{} {credential} {}", unquoted(&before), unquoted(&after))
        } else {
            format!("{before}\n{credential}\n{after}")
        };
        let output = masked(&[&text]);
        assert!(!output.contains("Zk9PQkFS"), "{text:?} -> {output:?}");
    }
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
