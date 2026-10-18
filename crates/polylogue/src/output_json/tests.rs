use super::*;
use serde_json::json;
use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
fn until() -> Instant {
    Instant::now() + Duration::from_secs(30)
}
#[test]
fn body_byte_ranges_preserve_utf8_masking_and_message_scope() {
    let raw = format!("pré 🧪 ghp_{} fin", "A".repeat(36));
    for width in 1..=raw.len() {
        let mut ranges = raw.as_bytes().chunks(width).map(|x| x.to_vec());
        let mut encoder = Encoder::new(Vec::new(), until());
        encoder.body_ranges(|| Ok(ranges.next())).unwrap();
        let bytes = encoder.finish().unwrap();
        assert_eq!(
            serde_json::from_slice::<String>(&bytes).unwrap(),
            "pré 🧪 [credential: GitHub token, masked] fin"
        );
    }
    let mut encoder = Encoder::new(Vec::new(), until());
    encoder.syntax(b"[").unwrap();
    encoder
        .body_text("-----BEGIN PRIVATE KEY-----\nsecret")
        .unwrap();
    encoder.message_boundary().unwrap();
    encoder.syntax(b",").unwrap();
    encoder.body_text("next independent message").unwrap();
    encoder.syntax(b"]").unwrap();
    let output: Value = serde_json::from_slice(&encoder.finish().unwrap()).unwrap();
    assert_eq!(output[1], "next independent message");
    assert!(!output[0].as_str().unwrap().contains("secret"));
}
#[test]
fn preserved_values_are_written_unchanged_or_refused() {
    let value = json!({"path":["content",0,"text"],"input":"opaque:id","version":1});
    let mut encoder = Encoder::new(Vec::new(), until());
    encoder
        .raw_json(
            &polylogue_model::RawJson::parse(&serde_json::to_vec(&value).unwrap()).unwrap(),
            true,
        )
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&encoder.finish().unwrap()).unwrap(),
        value
    );
    let value = json!({"path":[format!("ghp_{}","A".repeat(36))]});
    let mut encoder = Encoder::new(Vec::new(), until());
    assert!(
        encoder
            .raw_json(
                &polylogue_model::RawJson::parse(&serde_json::to_vec(&value).unwrap()).unwrap(),
                true
            )
            .is_err()
    );
    assert!(!String::from_utf8_lossy(&encoder.out).contains("ghp_"));
    assert!(encoder.syntax(b"null").is_err());
}
#[test]
fn malformed_or_failed_body_ranges_stop_before_future_reads() {
    let mut calls = 0;
    let mut encoder = Encoder::new(Vec::new(), until());
    let error = encoder
        .body_ranges(|| {
            calls += 1;
            Ok(Some(vec![0xff]))
        })
        .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert_eq!(calls, 1);
    assert!(encoder.syntax(b"null").is_err());
    let mut encoder = Encoder::new(Vec::new(), until());
    assert!(
        encoder
            .body_ranges(|| Ok(Some(vec![b'a'; PIECE + 1])))
            .is_err()
    );
    let mut encoder = Encoder::new(Vec::new(), until());
    assert!(encoder.body_ranges(|| Ok(Some(Vec::new()))).is_err());
}
#[test]
fn values_are_masked_once_and_json_escaping_preserves_surrounding_text() {
    let token = format!("ghp_{}", "A".repeat(36));
    let original =
        json!({"data":[format!("before \"{}\" after\n🧪",token),false,0,null],"empty":""});
    let mut encoder = Encoder::new(Vec::new(), until());
    encoder.value(&original).unwrap();
    let bytes = encoder.finish().unwrap();
    let masked: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        masked["data"][0],
        "before \"[credential: GitHub token, masked]\" after\n🧪"
    );
    assert_eq!(masked["data"][1], false);
    assert_eq!(masked["data"][2], 0);
    assert!(masked["data"][3].is_null());
    assert_eq!(masked["empty"], "");
    assert!(original["data"][0].as_str().unwrap().contains(&token));
}
#[test]
fn keys_needing_masking_are_refused_unwritten_and_stop_the_encoder() {
    let key = format!("ghp_{}", "A".repeat(36));
    let mut encoder = Encoder::new(Vec::new(), until());
    let error = encoder.value(&json!({key.clone():"private"})).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
    assert!(!String::from_utf8_lossy(&encoder.out).contains(&key));
    assert!(!String::from_utf8_lossy(&encoder.out).contains("private"));
    let before = encoder.out.len();
    assert!(encoder.syntax(b"null").is_err());
    assert_eq!(encoder.out.len(), before);
    let mut identity = Encoder::new(Vec::new(), until());
    assert!(identity.identity(&key).is_err());
    assert!(identity.out.is_empty());
    let mut ordinary = Encoder::new(Vec::new(), until());
    ordinary.syntax(b"{").unwrap();
    ordinary.key("id").unwrap();
    ordinary.identity("opaque:session:message").unwrap();
    ordinary.syntax(b"}").unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&ordinary.finish().unwrap()).unwrap(),
        json!({"id":"opaque:session:message"})
    );
}
#[test]
fn dense_credentials_expand_without_an_aggregate_piece_limit() {
    let input = "-u=a:b ".repeat(585);
    assert_eq!(input.len(), 4095);
    let mut encoder = Encoder::new(Vec::new(), until());
    encoder.value(&Value::String(input)).unwrap();
    let bytes = encoder.finish().unwrap();
    assert!(bytes.len() > 16384);
    let text: String = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        text,
        "-u=[credential: CLI credentials, masked] ".repeat(585)
    );
}
#[test]
fn body_pem_suppression_crosses_blocks_and_multibyte_piece_edges() {
    let mut encoder = Encoder::new(Vec::new(), until());
    encoder.syntax(b"[").unwrap();
    encoder
        .body_text("before -----BEGIN PRIVATE KEY-----\nQUFBQUFB")
        .unwrap();
    encoder.syntax(b",").unwrap();
    encoder
        .body_text("QkJCQkJC\n-----END PRIVATE KEY----- after🧪")
        .unwrap();
    encoder.syntax(b"]").unwrap();
    let parsed: Value = serde_json::from_slice(&encoder.finish().unwrap()).unwrap();
    let combined = parsed
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect::<String>();
    assert!(!combined.contains("QUFB"));
    assert!(!combined.contains("QkJC"));
    assert!(combined.starts_with("before "));
    assert!(combined.ends_with(" after🧪"));
    assert_eq!(combined.matches("[credential:").count(), 1);
    let text = format!("{} é🧪 ", "é".repeat(4095));
    let mut encoder = Encoder::new(Vec::new(), until());
    encoder.value(&Value::String(text.clone())).unwrap();
    assert_eq!(
        serde_json::from_slice::<String>(&encoder.finish().unwrap()).unwrap(),
        text
    );
}
#[derive(Debug)]
struct Marker(Arc<()>);
impl std::fmt::Display for Marker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("original sink failure")
    }
}
impl std::error::Error for Marker {}
#[test]
fn sink_failure_and_deadline_stop_the_encoder() {
    struct Fail {
        calls: Arc<AtomicUsize>,
        marker: Arc<()>,
    }
    impl Write for Fail {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 1 {
                return Err(io::Error::new(
                    io::ErrorKind::WouldBlock,
                    Marker(Arc::clone(&self.marker)),
                ));
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let marker = Arc::new(());
    let calls = Arc::new(AtomicUsize::new(0));
    let mut encoder = Encoder::new(
        Fail {
            calls: Arc::clone(&calls),
            marker: Arc::clone(&marker),
        },
        until(),
    );
    let error = encoder.value(&json!("ordinary text")).unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::WouldBlock);
    assert!(Arc::ptr_eq(
        &error.get_ref().unwrap().downcast_ref::<Marker>().unwrap().0,
        &marker
    ));
    let count = calls.load(Ordering::SeqCst);
    assert!(encoder.body_text("later").is_err());
    assert_eq!(calls.load(Ordering::SeqCst), count);
    let mut expired = Encoder::new(Vec::new(), Instant::now());
    assert_eq!(
        expired.value(&json!("unwritten")).unwrap_err().kind(),
        io::ErrorKind::TimedOut
    );
    assert!(expired.out.is_empty());
}
#[test]
fn escaped_output_accepts_multibyte_characters_one_byte_at_a_time() {
    let mut out = Vec::new();
    let text = "é🧪\n\"\\".repeat(1000);
    {
        let mut escaped = Escaped {
            out: &mut out,
            bytes: Vec::new(),
            until: until(),
        };
        for byte in text.as_bytes() {
            escaped.write_all(&[*byte]).unwrap();
        }
        escaped.emit(true).unwrap();
    }
    let mut quoted = vec![b'"'];
    quoted.extend_from_slice(&out);
    quoted.push(b'"');
    assert_eq!(serde_json::from_slice::<String>(&quoted).unwrap(), text);
}
