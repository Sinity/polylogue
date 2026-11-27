use super::*;
use std::io::{BufReader, Read};

struct Chunks<'a> {
    bytes: &'a [u8],
    size: usize,
}
impl io::Read for Chunks<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let length = out.len().min(self.size);
        io::Read::read(&mut self.bytes, &mut out[..length])
    }
}

#[test]
fn optional_fields_and_dynamic_names_keep_bounded_state_and_presence() {
    let mut observer = Observer::default();
    for mask in 0..4096u32 {
        let fields = (0..12)
            .filter(|bit| mask & (1 << bit) != 0)
            .map(|bit| format!("\"field{bit}\":0"))
            .collect::<Vec<_>>()
            .join(",");
        observer.json(format!("{{{fields}}}").as_bytes()).unwrap();
    }
    assert_eq!(observer.root.nodes.len(), 13);
    let schema = observer.finish("synthetic".into(), vec![]);
    for bit in 0..12 {
        let field = &schema.fields[&format!("/field{bit}")];
        assert_eq!(
            (field.parents, field.present, field.missing),
            (4096, 2048, 2048)
        );
    }

    let mut observer = Observer::new(["fixed".into()].into());
    for i in 0..10000 {
        observer
            .json(format!(r#"{{"entry{i}":{{"text":0}},"fixed":true}}"#).as_bytes())
            .unwrap();
    }
    assert!(observer.root.nodes.len() < 160);
    let schema = observer.finish("synthetic".into(), vec![]);
    assert_eq!(schema.fields.len(), 4);
    assert_eq!(schema.fields["/*"].present, 10000);
    assert_eq!(schema.fields["/*/text"].present, 10000);
    assert!(schema.fields["/fixed"].required);
    let diff = compare(&schema, &schema).unwrap();
    assert_eq!(
        (diff.comparable, diff.reasons, diff.changes, diff.tables),
        (true, vec![], vec![], vec![])
    );
}

#[test]
fn grouping_during_one_object_counts_each_parent_once() {
    let mut parts = (0..80)
        .map(|i| format!("\"entry{i}\":{{\"text\":0}}"))
        .collect::<Vec<_>>();
    parts.push("\"0123456789abcdef\":null".into());
    let bytes = format!("{{{}}}\n{{}}", parts.join(","));
    let schema = infer(bytes.as_bytes(), Framing::Lines);
    assert_eq!(schema.fields["/*"].present, 1);
    assert_eq!(schema.fields["/*"].occurrences, 81);
    assert_eq!(schema.fields["/*/text"].present, 80);
    let diff = compare(&schema, &schema).unwrap();
    assert_eq!(
        (diff.comparable, diff.reasons, diff.changes, diff.tables),
        (true, vec![], vec![], vec![])
    );
}

#[test]
fn incremental_grouping_matches_whole_sample_across_merge_orders() {
    let mut bytes = String::new();
    let mut observers = Vec::new();
    for i in 0..90 {
        let record = format!(
            r#"{{"mapping":{{"entry{i}":{{"field{}":0}},"fixed":{{"text":true}},"0123456789abcdef":null}}}}"#,
            i % 7
        );
        bytes.push_str(&record);
        bytes.push('\n');
        let mut observer = Observer::default();
        observer.json(record.as_bytes()).unwrap();
        observers.push(observer);
    }
    let expected = infer(bytes.as_bytes(), Framing::Lines);
    let mut combined = Observer::default();
    for observer in observers.into_iter().rev() {
        combined.merge(observer).unwrap();
    }
    let merged = combined.finish("synthetic".into(), vec![]);
    assert_eq!(merged, expected);
    // Hand count: the 92 distinct sibling names exceed fifty, so all of them
    // group under /mapping/*, three members in each of the 90 records. Of
    // those, 180 are objects (entryN and fixed); only fixed has "text", and
    // entryN has field(N % 7), which is 13 times for 0..=5 and 12 for 6.
    let group = &merged.fields["/mapping/*"];
    assert_eq!(
        (group.present, group.parents, group.occurrences),
        (90, 90, 270)
    );
    let text = &merged.fields["/mapping/*/text"];
    assert_eq!((text.present, text.parents, text.missing), (90, 180, 90));
    let field0 = &merged.fields["/mapping/*/field0"];
    assert_eq!((field0.present, field0.parents), (13, 180));
    let field6 = &merged.fields["/mapping/*/field6"];
    assert_eq!((field6.present, field6.parents), (12, 180));
}

#[test]
fn streaming_json_matches_serde_validation_and_scalar_kinds() {
    use serde_json::{Value, error::Category, json};

    let valid = [
        (
            br#"{"text":"a\u0000\uD834\uDD1E\"\\\/\b\f\n\r\t","n":-0.1e+400}"#.as_slice(),
            json!({"text": "a\0𝄞\"\\/\u{8}\u{c}\n\r\t", "n": "-0.1e+400".parse::<serde_json::Number>().unwrap()}),
        ),
        (
            "{\"é\":\"🦀\",\"x\":[[],{},null,true,false,-123,0,1E-2]}".as_bytes(),
            json!({"é": "🦀", "x": [[], {}, null, true, false, -123, 0, "1E-2".parse::<serde_json::Number>().unwrap()]}),
        ),
        (br#"{"same":1,"same":2}"#.as_slice(), json!({"same": 2})),
    ];
    for (bytes, value) in valid {
        assert_eq!(serde_json::from_slice::<Value>(bytes).unwrap(), value);
        for size in [1, 2, 7, 8192] {
            let mut observer = Observer::default();
            observer
                .read(BufReader::new(Chunks { bytes, size }), Framing::Json)
                .unwrap();
            assert_eq!(
                observer.finish("synthetic".into(), vec![]),
                infer(bytes, Framing::Json)
            );
        }
    }
    let invalid: &[(&[u8], &[u8], Category)] = &[
        (b"01", b"1", Category::Syntax),
        (b"+1", b"1", Category::Syntax),
        (b"1.", b"1.0", Category::Eof),
        (b"1e", b"1e0", Category::Eof),
        (b"--1", b"-1", Category::Syntax),
        (b"[1,]", b"[1]", Category::Syntax),
        (b"{\"x\":0,}", b"{\"x\":0}", Category::Syntax),
        (b"{\"x\":}", b"{\"x\":0}", Category::Syntax),
        (b"[true false]", b"[true,false]", Category::Syntax),
        (b"{}{}", b"{}", Category::Syntax),
        (b"\x0b0", b" 0", Category::Syntax),
        (b"[", b"[]", Category::Eof),
        (b"{", b"{}", Category::Eof),
        (br#""\uD800""#, br#""\uD800\uDC00""#, Category::Syntax),
        (br#""\uDC00""#, br#""\u0000""#, Category::Syntax),
        (br#""\uD800\u0000""#, br#""\uD800\uDC00""#, Category::Syntax),
        (br#""\x00""#, br#""\u0000""#, Category::Syntax),
        (b"\"\xc0\x80\"", b"\"\xc2\x80\"", Category::Syntax),
        (
            b"\"\xed\xa0\x80\"",
            "\"\u{d000}\"".as_bytes(),
            Category::Syntax,
        ),
        (
            b"\"\xf4\x90\x80\x80\"",
            "\"\u{100000}\"".as_bytes(),
            Category::Syntax,
        ),
        (b"\"\n\"", br#""\n""#, Category::Syntax),
    ];
    for &(bytes, valid, category) in invalid {
        assert_eq!(
            serde_json::from_slice::<Value>(bytes)
                .unwrap_err()
                .classify(),
            category,
            "{bytes:?}"
        );
        let mut observer = Observer::default();
        let error = observer.json(bytes).unwrap_err();
        assert_eq!(error.classify(), Category::Data, "{bytes:?}");
        assert_eq!(error.to_string(), "invalid JSON record", "{bytes:?}");
        assert_eq!(observer.counts(), (0, 0));
        observer.json(valid).unwrap();
        assert_eq!(observer.counts(), (1, 0));
    }
}

#[test]
fn invalid_capture_digest_and_array_tail_preserve_existing_structure() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let mut observer = Observer::default();
    observer.json(b"{\"existing\":0}".as_slice()).unwrap();
    let mut bytes = MAGIC.to_vec();
    bytes.extend(Header::of(b"{\"secret\":1}").encode());
    bytes.extend(b"{\"secret\":1}");
    let second_header = bytes.len();
    bytes.extend(Header::of(b"{\"secret\":1}").encode());
    bytes.extend(b"{\"secret\":2}");

    let digest = observer
        .read(io::Cursor::new(&bytes), Framing::CaptureLog)
        .unwrap_err();
    assert_eq!(digest.kind(), io::ErrorKind::Other);
    assert_eq!(digest.to_string(), "invalid_capture");

    let tail = observer
        .read(io::Cursor::new(b"[{\"secret\":2}],"), Framing::Array)
        .unwrap_err();
    assert_eq!(tail.kind(), io::ErrorKind::Other);
    assert_eq!(tail.to_string(), "invalid JSON record");

    let schema = observer.finish("synthetic".into(), vec![]);
    assert_eq!(schema.records, 1);
    assert!(!schema.fields.contains_key("/secret"));

    let mut observer = Observer::default();
    let matching = Header::of(b"{\"secret\":2}").encode();
    bytes[second_header..second_header + matching.len()].copy_from_slice(&matching);
    observer
        .read(io::Cursor::new(bytes), Framing::CaptureLog)
        .unwrap();
    assert_eq!(observer.counts(), (2, 0));
    observer
        .read(io::Cursor::new(b"[{\"secret\":2}]"), Framing::Array)
        .unwrap();
    let schema = observer.finish("synthetic".into(), vec![]);
    assert_eq!((schema.records, schema.fields["/secret"].present), (3, 3));
}

#[test]
fn deep_json_parsing_merging_finishing_and_cleanup_survive_in_a_subprocess() {
    const CHILD: &str = "POLYLOGUE_SCHEMA_DEPTH_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "schema::tests::streaming::deep_json_parsing_merging_finishing_and_cleanup_survive_in_a_subprocess", "--nocapture"])
            .env(CHILD, "1").output().unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            for (open, close) in [("[", "]"), ("{\"x\":", "}")] {
                let bytes = format!("{}0{}", open.repeat(4096), close.repeat(4096));
                let mut observer = Observer::default();
                observer.json(bytes.as_bytes()).unwrap();
                let mut other = Observer::default();
                other.json(bytes.as_bytes()).unwrap();
                observer.merge(other).unwrap();
                let schema = observer.finish("synthetic".into(), vec![]);
                assert_eq!(schema.fields.len(), 4097);
                assert_eq!(schema.records, 2);
                assert!(compare(&schema, &schema).unwrap().changes.is_empty());
                let encoded = serde_json::to_vec(&schema).unwrap();
                assert_eq!(serde_json::from_slice::<Schema>(&encoded).unwrap(), schema);
                let mut observer = Observer::default();
                let prefix = open.repeat(20000);
                let error = observer.json(prefix.as_bytes()).unwrap_err();
                assert_eq!(error.classify(), serde_json::error::Category::Data);
                assert_eq!(error.to_string(), "invalid JSON record");
                let complete = format!("{prefix}0{}", close.repeat(20000));
                observer.json(complete.as_bytes()).unwrap();
                assert_eq!(observer.counts(), (1, 0));
                observer.json(bytes.as_bytes()).unwrap();
                drop(observer);
                let array = format!("[{bytes}]");
                assert_eq!(infer(array.as_bytes(), Framing::Array).records, 1);
                assert_eq!(infer(bytes.as_bytes(), Framing::Lines).records, 1);
                let mut capture = polylogue_model::capture_log::MAGIC.to_vec();
                capture.extend(polylogue_model::capture_log::Header::of(bytes.as_bytes()).encode());
                capture.extend(bytes.as_bytes());
                assert_eq!(infer(&capture, Framing::CaptureLog).records, 1);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn blank_lines_and_invalid_leading_spaces_keep_record_counts() {
    let schema = infer(b"\x0b\x0c\n\n\t \r\n\x0b{}\n{}", Framing::Lines);
    // Vertical tabs make both lines unreadable.
    assert_eq!((schema.records, schema.unreadable_records), (1, 2));
}

#[test]
fn grouping_joins_deep_subtrees_without_recursing() {
    std::thread::Builder::new()
        .stack_size(128 * 1024)
        .spawn(|| {
            let value = format!("{}0{}", "[".repeat(1024), "]".repeat(1024));
            let mut combined = Observer::default();
            for i in 0..51 {
                combined
                    .json(format!("{{\"entry{i}\":{value}}}").as_bytes())
                    .unwrap();
            }
            let schema = combined.finish("synthetic".into(), vec![]);
            assert_eq!(schema.fields.len(), 1026);
            assert_eq!(schema.fields["/*"].occurrences, 51);
            let diff = compare(&schema, &schema).unwrap();
            assert_eq!(
                (diff.comparable, diff.reasons, diff.changes, diff.tables),
                (true, vec![], vec![], vec![])
            );
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
fn buffered_runs_validate_unicode_numbers_and_whitespace_across_reads() {
    let text = "é界🦀".repeat(4096);
    let bytes = format!(
        "{}{{\"{text}\":\"{text}\",\"number\":{}e+{}}}{}",
        " \t\r\n".repeat(4096),
        "9".repeat(16384),
        "1".repeat(16384),
        " \n".repeat(4096)
    );
    for size in [1, 7, 8192] {
        let mut observer = Observer::default();
        observer
            .read(
                BufReader::new(Chunks {
                    bytes: bytes.as_bytes(),
                    size,
                }),
                Framing::Json,
            )
            .unwrap();
        let schema = observer.finish("synthetic".into(), vec![]);
        assert_eq!(schema.records, 1);
        assert_eq!(schema.fields[&format!("/{text}")].types["string"], 1);
        assert_eq!(schema.fields["/number"].types["number"], 1);
    }
    let mut invalid = b"\"".to_vec();
    invalid.extend("é".repeat(8192).as_bytes());
    invalid.extend(b"\xff\"");
    let mut valid = invalid.clone();
    let last_character = valid.len() - 2;
    valid[last_character] = b'a';
    for size in [1, 7, 8192] {
        let mut observer = Observer::default();
        let error = observer
            .json(Chunks {
                bytes: &invalid,
                size,
            })
            .unwrap_err();
        assert_eq!(error.classify(), serde_json::error::Category::Data);
        assert_eq!(error.to_string(), "invalid JSON record");
        assert_eq!(observer.counts(), (0, 0));
        observer
            .json(Chunks {
                bytes: &valid,
                size,
            })
            .unwrap();
        assert_eq!(observer.counts(), (1, 0));
    }
}

#[test]
fn capture_reader_stops_at_the_frame_end_and_preserves_input_errors() {
    use polylogue_model::capture_log::Header;
    let body = b"synthetic capture";
    let header = Header::of(body);
    let mut reader = io::Cursor::new([body.as_slice(), b"next frame"].concat());
    {
        let mut frame = CaptureReader {
            reader: reader.by_ref().take(header.length),
            verifier: header.verifier(),
        };
        let mut bytes = Vec::new();
        frame.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, body);
        assert!(frame.verifier.verifies());
    }
    let mut tail = Vec::new();
    reader.read_to_end(&mut tail).unwrap();
    assert_eq!(tail, b"next frame");

    let mut frame = CaptureReader {
        reader: io::repeat(0).take(1).chain(Fails).take(header.length),
        verifier: header.verifier(),
    };
    let mut bytes = Vec::new();
    assert_eq!(
        frame.read_to_end(&mut bytes).unwrap_err().kind(),
        io::ErrorKind::PermissionDenied
    );
    assert!(!frame.verifier.verifies());
}

struct Fails;
impl Read for Fails {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        Err(io::Error::from(io::ErrorKind::PermissionDenied))
    }
}
