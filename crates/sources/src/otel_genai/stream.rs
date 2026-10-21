//! Validate a whole OTLP document, otherwise account independent JSONL lines.
use super::*;
use crate::export::{Input, ItemSink, ScanError, ScanStats};
use std::io::{self, Read};

const PATH: &[&str] = &["resourceSpans", "scopeSpans", "spans"];

fn input_error(error: io::Error) -> ScanError {
    ScanError::Input(serde_json::Error::io(error))
}
fn callback(error: String) -> ScanError {
    ScanError::Callback(error)
}

fn document(
    reader: impl Read,
    offset: u64,
    sink: &mut dyn ItemSink,
) -> Result<ScanStats, ScanError> {
    let mut invalid = None;
    let result = crate::export::scan_path(reader, PATH, |range, raw| {
        let value: Json = match serde_json::from_str(raw.get()) {
            Ok(value) => value,
            Err(error) => {
                invalid = Some(error);
                return Err("span cannot be decoded".into());
            }
        };
        let key = order_key(&value);
        sink.push(range.start + offset..range.end + offset, key, None, raw)
    });
    if let Some(error) = invalid {
        Err(ScanError::Malformed(error))
    } else {
        result
    }
}

fn order_key(span: &Json) -> Vec<u8> {
    let at = nanos(span, "startTimeUnixNano");
    let mut key = vec![u8::from(at.is_some())];
    if let Some(at) = at {
        key.extend(at.to_be_bytes());
    }
    key.extend(
        span.get("spanId")
            .and_then(Json::as_str)
            .unwrap_or("")
            .as_bytes(),
    );
    key
}

pub(super) fn scan(
    reader: &mut dyn Input,
    sink: &mut dyn ItemSink,
) -> Result<(ScanStats, u64), ScanError> {
    sink.begin().map_err(callback)?;
    match document(&mut *reader, 0, sink) {
        Ok(stats) => {
            sink.commit().map_err(callback)?;
            return Ok((stats, 0));
        }
        Err(ScanError::Malformed(_)) => sink.rollback().map_err(callback)?,
        Err(error) => {
            sink.rollback().map_err(callback)?;
            return Err(error);
        }
    }
    reader.rewind().map_err(input_error)?;
    let mut offset = 0;
    let mut refused = 0;
    let mut stats = ScanStats::default();
    loop {
        let next = reader.fill_buf().map_err(input_error)?.first().copied();
        match next {
            None => break,
            Some(b'\n') => {
                reader.consume(1);
                offset += 1;
                continue;
            }
            _ => {}
        }
        sink.begin().map_err(callback)?;
        let mut line = Line {
            reader,
            ended: false,
            bytes: 0,
        };
        let result = document(&mut line, offset, sink);
        match result {
            Ok(part) => {
                sink.commit().map_err(callback)?;
                stats.elements += part.elements;
                stats.peak_element_bytes = stats.peak_element_bytes.max(part.peak_element_bytes);
            }
            Err(ScanError::Malformed(_)) => {
                sink.rollback().map_err(callback)?;
                refused += 1;
            }
            Err(error) => {
                sink.rollback().map_err(callback)?;
                return Err(error);
            }
        }
        io::copy(&mut line, &mut io::sink()).map_err(input_error)?;
        offset += line.bytes;
    }
    Ok((stats, refused))
}

struct Line<'a> {
    reader: &'a mut dyn Input,
    ended: bool,
    bytes: u64,
}
impl Read for Line<'_> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.ended || out.is_empty() {
            return Ok(0);
        }
        let buffer = self.reader.fill_buf()?;
        if buffer.is_empty() {
            self.ended = true;
            return Ok(0);
        }
        let available = buffer
            .iter()
            .position(|&b| b == b'\n')
            .map_or(buffer.len(), |at| at + 1);
        let n = available.min(out.len());
        self.ended = buffer[n - 1] == b'\n';
        out[..n].copy_from_slice(&buffer[..n]);
        self.reader.consume(n);
        self.bytes += n as u64;
        Ok(n)
    }
}

#[cfg(test)]
mod tests;
