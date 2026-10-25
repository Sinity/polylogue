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
    let scan_state = Rc::new(RefCell::new(ScanState {
        last: Some((0, b'!')),
        ..Default::default()
    }));
    let mut callback = |_: Range<u64>, _: &RawValue| Ok(());
    assert!(
        ItemSeed {
            scan_state: &scan_state,
            item: &mut callback
        }
        .deserialize(&mut serde_json::Deserializer::from_str("{}"))
        .is_err()
    );
    assert!(matches!(scan_state.borrow().failure, Some(ScanError::Span)));
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

#[test]
fn optional_wrapper_frames_only_the_root_or_named_message_array() {
    for input in [
        r#" [ {"role":"user","content":"hi"}, {"role":"assistant","content":"hello"} ] "#,
        r#"{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}],"model":"after","other":[1,2]}"#,
    ] {
        for fragment in 1..=17 {
            let mut messages = Vec::new();
            let (metadata, stats) = scan_array_or_nested(
                Fragmented {
                    bytes: input.as_bytes(),
                    fragment,
                },
                &["messages"],
                |range, raw| {
                    assert_eq!(
                        &input.as_bytes()[range.start as usize..range.end as usize],
                        raw.get().as_bytes()
                    );
                    messages
                        .push(serde_json::from_str::<polylogue_model::Json>(raw.get()).unwrap());
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(stats.elements, 2);
            assert_eq!(messages[1]["content"], "hello");
            if metadata.is_object() {
                assert_eq!(metadata["messages"], serde_json::json!([]));
                assert_eq!(metadata["other"], serde_json::json!([1, 2]));
            } else {
                assert_eq!(metadata, serde_json::json!([]));
            }
        }
    }
    // AI Studio's object-only selection does not opt into bare-array records.
    let mut records = 0;
    let (_, stats) = scan_nested(
        b"[{},{}]".as_slice(),
        &["chunkedPrompt", "chunks"],
        |_, _| {
            records += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(records, 0);
    assert_eq!(stats.elements, 0);
}

#[test]
fn typed_wrapper_headers_refuse_duplicates_without_changing_value_wrapper_rules() {
    let input = br#"{"version":99,"items":[{}],"version":1}"#;
    assert!(matches!(
        scan_nested_unique(input.as_slice(), &["items"], |_, _| Ok(())),
        Err(ScanError::Malformed(_))
    ));
    let (metadata, stats) = scan_nested(input.as_slice(), &["items"], |_, _| Ok(())).unwrap();
    assert_eq!(metadata["version"], 1);
    assert_eq!(stats.elements, 1);
    let input = br#"{"version":1,"items":[{}]}"#;
    assert!(scan_nested_unique(input.as_slice(), &["items"], |_, _| Ok(())).is_ok());
}

#[test]
fn number_keys_keep_total_float_order_and_unknowns_last() {
    let mut values = vec![
        None,
        Some(-0.0),
        Some(0.0),
        Some(-1.0),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
    ];
    values.extend(
        (0..200).map(|n| Some(f64::from_bits((n as u64).wrapping_mul(0x9e3779b97f4a7c15)))),
    );
    for a in &values {
        for b in &values {
            let expected = match (a, b) {
                (Some(a), Some(b)) => a.total_cmp(b),
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (Some(_), None) => std::cmp::Ordering::Less,
                _ => std::cmp::Ordering::Equal,
            };
            assert_eq!(number_order(*a).cmp(&number_order(*b)), expected);
        }
    }
}

#[test]
fn keyed_maps_keep_exact_spans_across_short_reads_and_ignore_large_separators() {
    struct Short<'a> {
        bytes: &'a [u8],
        chunk: usize,
    }
    impl Read for Short<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let count = out.len().min(self.chunk).min(self.bytes.len());
            out[..count].copy_from_slice(&self.bytes[..count]);
            self.bytes = &self.bytes[count..];
            Ok(count)
        }
    }
    let text = [
        r#"{"before":1,"messages":{"a\u0000":"#,
        &" ".repeat(256 * 1024),
        r#"{"text":"é}}]"},"z":null,"z":false},"after":2}"#,
    ]
    .concat();
    for chunk in [1, 2, 7, 64, 4096] {
        let mut records = Vec::new();
        let (metadata, stats) = scan_map(
            Short {
                bytes: text.as_bytes(),
                chunk,
            },
            "messages",
            |range, key, raw| {
                assert_eq!(
                    &text.as_bytes()[range.start as usize..range.end as usize],
                    raw.get().as_bytes()
                );
                records.push((
                    key.to_owned(),
                    serde_json::from_str::<serde_json::Value>(raw.get()).unwrap(),
                ));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            records,
            vec![
                ("a\0".into(), serde_json::json!({"text":"é}}]"})),
                ("z".into(), serde_json::Value::Null),
                ("z".into(), false.into())
            ]
        );
        assert_eq!(
            metadata,
            serde_json::json!({"before":1,"messages":{},"after":2})
        );
        assert_eq!(stats.elements, 3);
        assert!(stats.peak_element_bytes < 100, "{stats:?}");
    }
}

#[test]
fn keyed_maps_check_the_complete_wrapper_and_preserve_callback_failures() {
    for text in [
        r#"{"messages":{"a":1}} false"#,
        r#"{"messages":{"a":1},"messages":{"b":2}}"#,
        r#"{"messages":{"a":1,}"#,
    ] {
        assert!(matches!(
            scan_map(text.as_bytes(), "messages", |_, _, _| Ok(())),
            Err(ScanError::Malformed(_))
        ));
    }
    assert!(
        matches!(scan_map(br#"{"messages":{"a":1}}"#.as_slice(),"messages",|_,_,_|Err("stop".into())),Err(ScanError::Callback(value)) if value=="stop")
    );
}
