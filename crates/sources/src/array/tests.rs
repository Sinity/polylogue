use super::*;

struct Fragmented<'a> {
    bytes: &'a [u8],
    fragment: usize,
}
impl Read for Fragmented<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let n = out.len().min(self.fragment).min(self.bytes.len());
        out[..n].copy_from_slice(&self.bytes[..n]);
        self.bytes = &self.bytes[n..];
        Ok(n)
    }
}

#[test]
fn serde_raw_elements_have_exact_spans_on_fragmented_input() {
    let values = [
        r#"{"text":"é 🦀 [,] \\\"","nested":[{},[1,2]]}"#,
        "-12.3e+4",
        "true",
        "false",
        "null",
        r#""[,{}]""#,
        "[[[]]]",
        "{}",
        "[]",
        "0",
    ];
    let mut input = String::from(" \r\n[  ");
    let mut expected = Vec::new();
    for (i, value) in values.iter().enumerate() {
        if i != 0 {
            input.push_str(" ,\n \t");
        }
        let start = input.len() as u64;
        input.push_str(value);
        expected.push((start..input.len() as u64, value.to_string()));
    }
    input.push_str("\n ] \r\n");
    for fragment in 1..=17 {
        let mut actual = Vec::new();
        scan(
            Fragmented {
                bytes: input.as_bytes(),
                fragment,
            },
            |range, raw| {
                assert_eq!(
                    &input.as_bytes()[range.start as usize..range.end as usize],
                    raw.get().as_bytes()
                );
                actual.push((range, raw.get().to_owned()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(actual, expected);
    }
}

#[test]
fn malformed_outer_suffix_is_an_error_after_complete_elements() {
    for input in ["[{},", "[{}]garbage", "[{},]", "{}", "[", "[{}"] {
        assert!(scan(input.as_bytes(), |_, _| Ok(())).is_err(), "{input}");
    }
}

struct Repeated {
    left: usize,
    at: usize,
}
impl Read for Repeated {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let pattern = b",{\"value\":\"escaped \\\" [,] \"}";
        let mut n = 0;
        while n < out.len() && self.left > 0 {
            out[n] = pattern[self.at];
            n += 1;
            self.at += 1;
            if self.at == pattern.len() {
                self.at = 0;
                self.left -= 1;
            }
        }
        Ok(n)
    }
}

#[test]
fn scan_buffers_follow_element_size_rather_than_array_size_or_whitespace() {
    let mut peaks = Vec::new();
    for n in [100, 10_000] {
        let reader = io::Cursor::new(b"[{}".as_slice())
            .chain(Repeated { left: n, at: 0 })
            .chain(io::Cursor::new(b"]"));
        let stats = scan(reader, |_, _| Ok(())).unwrap();
        assert_eq!(stats.elements, n as u64 + 1);
        peaks.push(stats.peak_element_bytes);
    }
    assert_eq!(peaks[0], peaks[1]);
    assert!(peaks[0] < 128);
    let reader = io::repeat(b' ')
        .take(1 << 20)
        .chain(io::Cursor::new(b"[{}]"));
    assert!(scan(reader, |_, _| Ok(())).unwrap().peak_element_bytes <= 5);
}

#[test]
fn scan_failures_distinguish_provider_syntax_input_callback_and_span() {
    struct Broken;
    impl Read for Broken {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("source unavailable"))
        }
    }
    assert!(matches!(
        scan(Broken, |_, _| Ok(())),
        Err(ScanError::Input(_))
    ));
    assert!(matches!(
        scan(b"[{},]".as_slice(), |_, _| Ok(())),
        Err(ScanError::Malformed(_))
    ));
    assert!(matches!(
        scan(b"[{}]".as_slice(), |_, _| Err(
            "Serde element bytes differ".into()
        )),
        Err(ScanError::Callback(_))
    ));
    let tap = Rc::new(RefCell::new(Tap {
        last: Some((0, b'!')),
        ..Default::default()
    }));
    let mut callback = |_: Range<u64>, _: &RawValue| Ok(());
    assert!(
        ItemSeed {
            tap: &tap,
            item: &mut callback
        }
        .deserialize(&mut serde_json::Deserializer::from_str("{}"))
        .is_err()
    );
    assert!(matches!(tap.borrow().failure, Some(ScanError::Span)));
}

#[test]
fn nested_chunks_keep_exact_spans_and_metadata_on_either_side() {
    let input = br#"{"before":17,"chunkedPrompt":{"pendingInputs":[{"text":"draft"}],"chunks":[{"role":"user","text":"[,{}]"},{"role":"model","text":"answer"}],"after":false},"runSettings":{"model":"models/example"}}"#;
    for fragment in 1..=17 {
        let mut chunks = Vec::new();
        let (metadata, stats) = scan_nested(
            Fragmented {
                bytes: input,
                fragment,
            },
            &["chunkedPrompt", "chunks"],
            |range, raw| {
                assert_eq!(
                    &input[range.start as usize..range.end as usize],
                    raw.get().as_bytes()
                );
                chunks.push(serde_json::from_str::<polylogue_model::Json>(raw.get()).unwrap());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(stats.elements, 2);
        assert_eq!(chunks[1]["text"], "answer");
        let mut expected: polylogue_model::Json = serde_json::from_slice(input).unwrap();
        expected["chunkedPrompt"]["chunks"] = serde_json::json!([]);
        assert_eq!(metadata, expected);
    }
}

#[test]
fn nested_scan_validates_suffix_refuses_duplicate_paths_and_bounds_element_buffers() {
    for input in [
        r#"{"chunkedPrompt":{"chunks":[{}]}} false"#,
        r#"{"chunkedPrompt":{"chunks":[{}],"chunks":[]}}"#,
        r#"{"chunkedPrompt":{"chunks":[]},"chunkedPrompt":{"chunks":[]}}"#,
    ] {
        assert!(matches!(
            scan_nested(
                input.as_bytes(),
                &["chunkedPrompt", "chunks"],
                |_, _| Ok(())
            ),
            Err(ScanError::Malformed(_))
        ));
    }
    let mut peaks = Vec::new();
    for n in [100, 10_000] {
        let reader = io::Cursor::new(b"{\"chunkedPrompt\":{\"chunks\":[{}".as_slice())
            .chain(Repeated { left: n, at: 0 })
            .chain(io::Cursor::new(b"]},\"runSettings\":{}}"));
        let (metadata, stats) =
            scan_nested(reader, &["chunkedPrompt", "chunks"], |_, _| Ok(())).unwrap();
        assert_eq!(stats.elements, n as u64 + 1);
        assert_eq!(metadata["chunkedPrompt"]["chunks"], serde_json::json!([]));
        peaks.push(stats.peak_element_bytes);
    }
    assert_eq!(peaks[0], peaks[1]);
}
