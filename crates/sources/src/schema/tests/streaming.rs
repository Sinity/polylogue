use super::*;
use std::io::BufReader;

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
fn optional_fields_and_dynamic_names_keep_bounded_state_and_exact_presence() {
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
    assert!(compare(&schema, &schema).is_ok());
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
    assert!(compare(&schema, &schema).is_ok());
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
    assert_eq!(combined.finish("synthetic".into(), vec![]), expected);
}

#[test]
fn streaming_json_matches_serde_validation_and_scalar_kinds() {
    let valid: &[&[u8]] = &[
        br#"{"text":"a\u0000\uD834\uDD1E\"\\\/\b\f\n\r\t","n":-0.1e+400}"#,
        "{\"é\":\"🦀\",\"x\":[[],{},null,true,false,-123,0,1E-2]}".as_bytes(),
        br#"{"same":1,"same":2}"#,
    ];
    for bytes in valid {
        assert!(serde_json::from_slice::<serde_json::Value>(bytes).is_ok());
        for size in [1, 2, 7, 8192] {
            let mut observer = Observer::default();
            observer
                .read(BufReader::new(Chunks { bytes, size }), Framing::Json)
                .unwrap();
            assert!(
                compare(
                    &observer.finish("synthetic".into(), vec![]),
                    &infer(bytes, Framing::Json)
                )
                .unwrap()
                .changes
                .is_empty()
            );
        }
    }
    let invalid: &[&[u8]] = &[
        b"01",
        b"+1",
        b"1.",
        b"1e",
        b"--1",
        b"[1,]",
        b"{\"x\":0,}",
        b"{\"x\":}",
        b"[true false]",
        b"{}{}",
        b"\x0b0",
        b"[",
        b"{",
        br#""\uD800""#,
        br#""\uDC00""#,
        br#""\uD800\u0000""#,
        br#""\x00""#,
        b"\"\xc0\x80\"",
        b"\"\xed\xa0\x80\"",
        b"\"\xf4\x90\x80\x80\"",
        b"\"\n\"",
    ];
    for bytes in invalid {
        assert!(serde_json::from_slice::<serde_json::Value>(bytes).is_err());
        let mut observer = Observer::default();
        assert!(observer.json(*bytes).is_err(), "{bytes:?}");
        assert_eq!(observer.counts(), (0, 0));
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
    bytes.extend(Header::of(b"{\"secret\":1}").encode());
    bytes.extend(b"{\"secret\":2}");
    assert!(
        observer
            .read(io::Cursor::new(bytes), Framing::CaptureLog)
            .is_err()
    );
    assert!(
        observer
            .read(io::Cursor::new(b"[{\"secret\":2}],"), Framing::Array)
            .is_err()
    );
    let schema = observer.finish("synthetic".into(), vec![]);
    assert_eq!(schema.records, 1);
    assert!(!schema.fields.contains_key("/secret"));
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
                assert!(serde_json::from_slice::<Schema>(&encoded).is_ok());
                let mut observer = Observer::default();
                assert!(observer.json(open.repeat(20000).as_bytes()).is_err());
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
            assert!(compare(&schema, &schema).is_ok());
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
    for size in [1, 7, 8192] {
        let mut observer = Observer::default();
        assert!(
            observer
                .json(Chunks {
                    bytes: &invalid,
                    size
                })
                .is_err()
        );
        assert_eq!(observer.counts(), (0, 0));
    }
}
