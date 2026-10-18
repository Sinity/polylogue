//! JSON encoder for read and export responses. It masks credentials in string
//! values, refuses keys and identities that would need masking, and writes in
//! fixed-size pieces however much masking expands the text.
use credential_masking::Masker;
mod native_identity;
pub(crate) mod raw_value;
mod string_ranges;
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
    /// One stored JSON string read in ranges from the same field. `range`
    /// returns at most `PIECE` bytes from an offset and whether it reached the end.
    pub(crate) fn string_ranges(
        &mut self,
        range: impl Fn(u64) -> io::Result<(Vec<u8>, bool)>,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> io::Result<()> {
        self.check()?;
        let result = string_ranges::write(&mut self.out, self.deadline, &range, cancelled);
        self.result(result)
    }
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
    /// Masking an identity would break references to it, so an identity that
    /// needs masking is refused before any byte is written.
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
    pub(crate) fn origin(&mut self, value: &polylogue_model::RawJson) -> io::Result<()> {
        self.check()?;
        let result = native_identity::check_origin(value, self.deadline);
        self.result(result)?;
        self.raw_json(value, true)
    }
    pub(crate) fn value(&mut self, value: &Value) -> io::Result<()> {
        self.check()?;
        let result = write_value(&mut self.out, value, self.deadline, 0);
        self.result(result)
    }
    /// Provider values stay `RawJson` so lone surrogates and number spelling
    /// survive; strings are masked before the canonical writer emits them.
    /// With `preserve`, a value that needs masking is refused instead.
    pub(crate) fn raw_json(
        &mut self,
        value: &polylogue_model::RawJson,
        preserve: bool,
    ) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            let (value, _) = raw_value::mask(value, self.deadline, preserve)?;
            self.out.write_all(&value.canonical())
        })();
        self.result(result)
    }
    pub(crate) fn opaque(&mut self, value: &polylogue_model::RawJson) -> io::Result<()> {
        self.check()?;
        let result = (|| {
            let envelope = raw_value::envelope(value, self.deadline, false)?;
            serde_json::to_writer(&mut self.out, &envelope).map_err(Into::into)
        })();
        self.result(result)
    }
    /// A message is one masking scope: a PEM block may span its text blocks but
    /// never continues into the next message.
    pub(crate) fn message_boundary(&mut self) -> io::Result<()> {
        self.check()?;
        let result = self.body.finish(&mut Empty);
        self.result(result)?;
        self.body = Masker::new();
        Ok(())
    }
    /// Text and thinking blocks share the message's masker. The block boundary
    /// flushes finished text but keeps an open PEM block suppressed.
    pub(crate) fn body_text(&mut self, text: &str) -> io::Result<()> {
        self.check()?;
        let result = write_string(&mut self.out, text, &mut self.body, self.deadline, true);
        self.result(result)
    }
    /// One body string from UTF-8 byte ranges of at most `PIECE` bytes.
    /// Codepoints and credentials may cross range edges; only a partial
    /// codepoint is carried between ranges.
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
                        "body range is empty or too large",
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
                "unflushed JSON body text",
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
    value: &polylogue_model::RawJson,
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
            "JSON response nested too deeply",
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
            // serde does the escaping. At most 4096 source bytes become at most
            // 24578 quoted bytes, so this allocation never grows with the string.
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

#[cfg(test)]
mod tests;
