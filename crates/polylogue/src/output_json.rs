//! One response-boundary JSON encoder for normalized read/export documents.
//! It masks logical string values, refuses credential-bearing structural keys,
//! and keeps settled UTF-8 output bounded independently of masking expansion.
use credential_output::Masker;
mod native_identity;
use serde_json::Value;
use std::io::{self, Write};
use std::time::Instant;

const PIECE: usize = 4096;

pub(crate) struct Encoder<W> {
    out: W,
    body: Masker,
    deadline: Instant,
    failed: bool,
}
impl<W: Write> Encoder<W> {
    pub(crate) fn new(out: W, deadline: Instant) -> Self {
        Self {
            out,
            body: Masker::new(),
            deadline,
            failed: false,
        }
    }
    fn check(&self) -> io::Result<()> {
        if self.failed {
            return Err(io::Error::other("JSON response encoder already failed"));
        }
        if Instant::now() >= self.deadline {
            return Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "JSON response deadline exceeded",
            ));
        }
        Ok(())
    }
    fn result(&mut self, result: io::Result<()>) -> io::Result<()> {
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    pub(crate) fn syntax(&mut self, bytes: &'static [u8]) -> io::Result<()> {
        self.check()?;
        let result = self.out.write_all(bytes);
        self.result(result)
    }
    pub(crate) fn key(&mut self, key: &str) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            unchanged_key(key, self.deadline)?;
            serde_json::to_writer(&mut self.out, key)?;
            self.out.write_all(b":")
        })();
        self.result(result)
    }
    /// Stable identities cannot be rewritten into credential markers without
    /// changing references. Refuse such a value before writing its bytes.
    pub(crate) fn identity(&mut self, value: &str) -> io::Result<()> {
        self.exact_identity(value, unchanged_key)
    }
    pub(crate) fn reference(&mut self, value: &str) -> io::Result<()> {
        self.exact_identity(value, check_reference)
    }
    fn exact_identity(
        &mut self,
        value: &str,
        assess: fn(&str, Instant) -> io::Result<()>,
    ) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            assess(value, self.deadline)?;
            serde_json::to_writer(&mut self.out, value).map_err(Into::into)
        })();
        self.result(result)
    }
    pub(crate) fn native_id(&mut self, value: &polylogue_model::NativeId) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            native_identity::check_native(value, self.deadline)?;
            serde_json::to_writer(&mut self.out, value).map_err(Into::into)
        })();
        self.result(result)
    }
    pub(crate) fn origin(&mut self, value: &polylogue_model::RetainedJson) -> io::Result<()> {
        self.check()?;
        let result = native_identity::check_origin(value, self.deadline);
        self.result(result)?;
        self.retained(value, true)
    }
    pub(crate) fn value(&mut self, value: &Value) -> io::Result<()> {
        self.check()?;
        let result = write_value(&mut self.out, value, self.deadline, 0);
        self.result(result)
    }
    /// Retained provider values never pass through serde_json::Value. Privacy
    /// examines logical strings before the canonical writer emits JSON.
    pub(crate) fn retained(
        &mut self,
        value: &polylogue_model::RetainedJson,
        preserve: bool,
    ) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            let (value, _) = retained::present(value, self.deadline, preserve)?;
            self.out.write_all(&value.canonical())
        })();
        self.result(result)
    }
    pub(crate) fn opaque(&mut self, value: &polylogue_model::RetainedJson) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            let envelope = retained::envelope(value, self.deadline, false)?;
            serde_json::to_writer(&mut self.out, &envelope).map_err(Into::into)
        })();
        self.result(result)
    }
    /// A Message is one logical body. PEM suppression crosses its blocks, never
    /// silently acquiring authority over the next independent Message.
    pub(crate) fn message_boundary(&mut self) -> io::Result<()> {
        self.check()?;
        let result = self.body.finish(&mut Empty);
        self.result(result)?;
        self.body = Masker::new();
        Ok(())
    }
    /// Text/Thinking blocks share the existing logical body masker. A block
    /// boundary flushes settled ordinary text while retaining PEM suppression.
    pub(crate) fn body_text(&mut self, text: &str) -> io::Result<()> {
        self.check()?;
        let result = write_string(&mut self.out, text, &mut self.body, self.deadline, true);
        self.result(result)
    }
    /// One logical body string from admitted raw UTF-8 byte ranges. Codepoints
    /// and credentials may straddle every range; only a partial codepoint is
    /// retained in addition to the existing fixed masking state.
    pub(crate) fn body_ranges(
        &mut self,
        mut next: impl FnMut() -> io::Result<Option<Vec<u8>>>,
    ) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(PIECE).map_err(io::Error::other)?;
            self.out.write_all(b"\"")?;
            let mut escaped = Escaped {
                out: &mut self.out,
                bytes,
                until: self.deadline,
            };
            let mut carry = Vec::with_capacity(PIECE + 3);
            while let Some(piece) = next()? {
                deadline(self.deadline)?;
                if piece.is_empty() || piece.len() > PIECE {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "body range exceeds admission or makes no progress",
                    ));
                }
                carry.extend_from_slice(&piece);
                let valid = match std::str::from_utf8(&carry) {
                    Ok(_) => carry.len(),
                    Err(e) if e.error_len().is_none() => e.valid_up_to(),
                    Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
                };
                self.body.push(
                    std::str::from_utf8(&carry[..valid]).map_err(io::Error::other)?,
                    &mut escaped,
                )?;
                carry.drain(..valid);
            }
            if !carry.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "incomplete body UTF-8",
                ));
            }
            self.body.block_boundary(&mut escaped)?;
            escaped.emit(true)?;
            escaped.out.write_all(b"\"")
        })();
        self.result(result)
    }
    pub(crate) fn finish(mut self) -> io::Result<W> {
        self.check()?;
        self.body.finish(&mut Empty)?;
        self.out.flush()?;
        Ok(self.out)
    }
}
struct Empty;
impl Write for Empty {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.is_empty() {
            Ok(0)
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsettled JSON body text",
            ))
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn deadline(until: Instant) -> io::Result<()> {
    if Instant::now() >= until {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "JSON response deadline exceeded",
        ))
    } else {
        Ok(())
    }
}
fn pieces(
    text: &str,
    until: Instant,
    mut push: impl FnMut(&str) -> io::Result<()>,
) -> io::Result<()> {
    let mut start = 0;
    while start < text.len() {
        deadline(until)?;
        let mut end = text.len().min(start.saturating_add(PIECE));
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        push(&text[start..end])?;
        start = end;
    }
    Ok(())
}
pub(crate) fn check_native_identity(
    value: &polylogue_model::NativeId,
    until: Instant,
) -> io::Result<()> {
    native_identity::check_native(value, until)
}
pub(crate) fn check_source_origin(
    value: &polylogue_model::RetainedJson,
    until: Instant,
) -> io::Result<()> {
    native_identity::check_origin(value, until)
}
pub(crate) fn check_reference(key: &str, until: Instant) -> io::Result<()> {
    native_identity::check_public(key, until)
}
pub(crate) fn check_identity(key: &str, until: Instant) -> io::Result<()> {
    unchanged_key(key, until)
}
fn unchanged_key(key: &str, until: Instant) -> io::Result<()> {
    struct Compare<'a> {
        original: &'a [u8],
        offset: usize,
        changed: bool,
    }
    impl Write for Compare<'_> {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let end = self
                .offset
                .checked_add(bytes.len())
                .ok_or_else(|| io::Error::other("key length overflow"))?;
            self.changed |= self.original.get(self.offset..end) != Some(bytes);
            self.offset = end;
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut compare = Compare {
        original: key.as_bytes(),
        offset: 0,
        changed: false,
    };
    let mut masker = Masker::new();
    pieces(key, until, |piece| masker.push(piece, &mut compare))?;
    masker.finish(&mut compare)?;
    if compare.changed || compare.offset != key.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "structural JSON key requires credential masking",
        ));
    }
    Ok(())
}
fn write_value(
    out: &mut impl Write,
    value: &Value,
    until: Instant,
    depth: usize,
) -> io::Result<()> {
    deadline(until)?;
    if depth > 128 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "JSON response nesting exceeds admission",
        ));
    }
    match value {
        Value::String(text) => write_string(out, text, &mut Masker::new(), until, false),
        Value::Array(items) => {
            out.write_all(b"[")?;
            for (i, item) in items.iter().enumerate() {
                if i != 0 {
                    out.write_all(b",")?;
                }
                write_value(out, item, until, depth + 1)?;
            }
            out.write_all(b"]")
        }
        Value::Object(fields) => {
            out.write_all(b"{")?;
            for (i, (key, value)) in fields.iter().enumerate() {
                unchanged_key(key, until)?;
                if i != 0 {
                    out.write_all(b",")?;
                }
                serde_json::to_writer(&mut *out, key)?;
                out.write_all(b":")?;
                write_value(out, value, until, depth + 1)?;
            }
            out.write_all(b"}")
        }
        _ => serde_json::to_writer(out, value).map_err(Into::into),
    }
}
fn write_string(
    out: &mut impl Write,
    text: &str,
    masker: &mut Masker,
    until: Instant,
    boundary: bool,
) -> io::Result<()> {
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(PIECE).map_err(io::Error::other)?;
    out.write_all(b"\"")?;
    let mut escaped = Escaped { out, bytes, until };
    pieces(text, until, |piece| masker.push(piece, &mut escaped))?;
    if boundary {
        masker.block_boundary(&mut escaped)?;
    } else {
        masker.finish(&mut escaped)?;
    }
    escaped.emit(true)?;
    escaped.out.write_all(b"\"")
}
struct Escaped<'a, W> {
    out: &'a mut W,
    bytes: Vec<u8>,
    until: Instant,
}
impl<W: Write> Escaped<'_, W> {
    fn emit(&mut self, complete: bool) -> io::Result<()> {
        deadline(self.until)?;
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(_) => self.bytes.len(),
            Err(error) if error.error_len().is_none() && !complete => error.valid_up_to(),
            Err(error) => return Err(io::Error::new(io::ErrorKind::InvalidData, error)),
        };
        if valid != 0 {
            let text = std::str::from_utf8(&self.bytes[..valid]).map_err(io::Error::other)?;
            // serde owns JSON escaping. At most4096 source bytes become at most
            // 24578 quoted bytes; this allocation never grows with a whole leaf.
            let encoded = serde_json::to_string(text)?;
            self.out
                .write_all(&encoded.as_bytes()[1..encoded.len() - 1])?;
            self.bytes.drain(..valid);
        }
        Ok(())
    }
}
impl<W: Write> Write for Escaped<'_, W> {
    fn write(&mut self, mut bytes: &[u8]) -> io::Result<usize> {
        let length = bytes.len();
        while !bytes.is_empty() {
            let take = bytes.len().min(PIECE - self.bytes.len());
            self.bytes
                .try_reserve_exact(take)
                .map_err(io::Error::other)?;
            self.bytes.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.bytes.len() == PIECE {
                self.emit(false)?;
            }
        }
        Ok(length)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.emit(false)
    }
}

pub(crate) mod retained;

#[cfg(test)]
mod tests {
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
    fn interpretation_descriptors_are_preserved_or_refused_never_rewritten() {
        let value = json!({"path":["content",0,"text"],"input":"opaque:id","version":1});
        let mut encoder = Encoder::new(Vec::new(), until());
        encoder
            .retained(
                &polylogue_model::RetainedJson::parse(&serde_json::to_vec(&value).unwrap())
                    .unwrap(),
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
                .retained(
                    &polylogue_model::RetainedJson::parse(&serde_json::to_vec(&value).unwrap())
                        .unwrap(),
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
    fn structural_keys_refuse_without_disclosure_or_renaming_and_then_fuse() {
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
    fn original_sink_failure_and_deadline_are_explicit_and_fused() {
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
    fn settled_output_handles_each_byte_of_a_multibyte_scalar() {
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
}

#[cfg(test)]
#[path = "output_json/native_identity_tests.rs"]
mod native_identity_tests;
