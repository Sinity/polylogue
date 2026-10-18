//! Only one normalized JSON string, not a general retained-value parser.
//! Assessment compares the masking output with a second bounded logical read.
//! Native non-scalar UTF-16 can then be copied exactly only if unchanged.
use super::{Escaped, PIECE, deadline};
use credential_output::Masker;
use std::io::{self, Read, Write};
use std::time::Instant;

fn invalid() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        "invalid normalized JSON string range",
    )
}

struct Bytes<'a, F> {
    range: &'a F,
    until: Instant,
    bytes: Vec<u8>,
    at: usize,
    offset: u64,
    eof: bool,
}
impl<'a, F: Fn(u64) -> io::Result<(Vec<u8>, bool)>> Bytes<'a, F> {
    fn new(range: &'a F, until: Instant) -> Self {
        Self {
            range,
            until,
            bytes: Vec::new(),
            at: 0,
            offset: 0,
            eof: false,
        }
    }
    fn next(&mut self) -> io::Result<Option<u8>> {
        if self.at == self.bytes.len() {
            if self.eof {
                return Ok(None);
            }
            deadline(self.until)?;
            let (bytes, eof) = (self.range)(self.offset)?;
            if bytes.len() > PIECE || (bytes.is_empty() && !eof) {
                return Err(invalid());
            }
            self.offset = self
                .offset
                .checked_add(bytes.len() as u64)
                .ok_or_else(invalid)?;
            self.bytes = bytes;
            self.at = 0;
            self.eof = eof;
            if self.bytes.is_empty() {
                return Ok(None);
            }
        }
        let byte = self.bytes[self.at];
        self.at += 1;
        Ok(Some(byte))
    }
    fn byte(&mut self) -> io::Result<u8> {
        self.next()?.ok_or_else(invalid)
    }
}

struct Logical<'a, F> {
    raw: Bytes<'a, F>,
    started: bool,
    done: bool,
    unit: Option<u16>,
    high: Option<u16>,
    text: [u8; 4],
    at: usize,
    end: usize,
    non_scalar: bool,
}
impl<'a, F: Fn(u64) -> io::Result<(Vec<u8>, bool)>> Logical<'a, F> {
    fn new(range: &'a F, until: Instant) -> Self {
        Self {
            raw: Bytes::new(range, until),
            started: false,
            done: false,
            unit: None,
            high: None,
            text: [0; 4],
            at: 0,
            end: 0,
            non_scalar: false,
        }
    }
    fn unit(&mut self) -> io::Result<Option<u16>> {
        if let Some(unit) = self.unit.take() {
            return Ok(Some(unit));
        }
        if self.done {
            return Ok(None);
        }
        if !self.started {
            if self.raw.byte()? != b'"' {
                return Err(invalid());
            }
            self.started = true;
        }
        let first = self.raw.byte()?;
        let unit = match first {
            b'"' => {
                self.done = true;
                if self.raw.next()?.is_some() {
                    return Err(invalid());
                }
                return Ok(None);
            }
            b'\\' => match self.raw.byte()? {
                b'"' => b'"' as u16,
                b'\\' => b'\\' as u16,
                b'/' => b'/' as u16,
                b'b' => 8,
                b'f' => 12,
                b'n' => 10,
                b'r' => 13,
                b't' => 9,
                b'u' => {
                    let mut unit = 0;
                    for _ in 0..4 {
                        let digit = (self.raw.byte()? as char)
                            .to_digit(16)
                            .ok_or_else(invalid)?;
                        unit = unit * 16 + digit as u16;
                    }
                    unit
                }
                _ => return Err(invalid()),
            },
            0..=31 => return Err(invalid()),
            32..=127 => first as u16,
            _ => {
                let length = match first {
                    0xc2..=0xdf => 2,
                    0xe0..=0xef => 3,
                    0xf0..=0xf4 => 4,
                    _ => return Err(invalid()),
                };
                let mut bytes = [0; 4];
                bytes[0] = first;
                for byte in &mut bytes[1..length] {
                    *byte = self.raw.byte()?;
                }
                let scalar = std::str::from_utf8(&bytes[..length])
                    .map_err(|_| invalid())?
                    .chars()
                    .next()
                    .ok_or_else(invalid)?;
                let mut units = [0; 2];
                let units = scalar.encode_utf16(&mut units);
                if units.len() == 2 {
                    self.unit = Some(units[1]);
                }
                units[0]
            }
        };
        Ok(Some(unit))
    }
    fn scalar(&mut self) -> io::Result<Option<char>> {
        let unit = match self.high.take() {
            Some(unit) => Some(unit),
            None => self.unit()?,
        };
        let Some(unit) = unit else {
            return Ok(None);
        };
        if (0xd800..=0xdbff).contains(&unit) {
            if let Some(next) = self.unit()? {
                if (0xdc00..=0xdfff).contains(&next) {
                    return Ok(char::from_u32(
                        0x10000 + (((unit as u32) - 0xd800) << 10) + next as u32 - 0xdc00,
                    ));
                }
                self.high = Some(next);
            }
        } else if !(0xdc00..=0xdfff).contains(&unit) {
            return Ok(char::from_u32(unit as u32));
        }
        self.non_scalar = true;
        Ok(Some(char::REPLACEMENT_CHARACTER))
    }
}
impl<F: Fn(u64) -> io::Result<(Vec<u8>, bool)>> Read for Logical<'_, F> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        let mut n = 0;
        while n < bytes.len() {
            if self.at == self.end {
                let Some(scalar) = self.scalar()? else {
                    break;
                };
                self.end = scalar.encode_utf8(&mut self.text).len();
                self.at = 0;
            }
            let take = (bytes.len() - n).min(self.end - self.at);
            bytes[n..n + take].copy_from_slice(&self.text[self.at..self.at + take]);
            n += take;
            self.at += take;
        }
        Ok(n)
    }
}

struct Compare<R> {
    expected: R,
    changed: bool,
}
impl<R: Read> Write for Compare<R> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut expected = [0; PIECE];
        for piece in bytes.chunks(PIECE) {
            let mut n = 0;
            while n < piece.len() {
                let got = self.expected.read(&mut expected[n..piece.len()])?;
                if got == 0 {
                    self.changed = true;
                    break;
                }
                n += got;
            }
            self.changed |= expected[..n] != piece[..n];
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn chunks<R: Read>(
    reader: &mut R,
    mut consume: impl FnMut(&str) -> io::Result<()>,
) -> io::Result<()> {
    let mut bytes = [0; PIECE];
    let mut carry = Vec::with_capacity(PIECE + 3);
    loop {
        let n = reader.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        carry.extend_from_slice(&bytes[..n]);
        let valid = match std::str::from_utf8(&carry) {
            Ok(_) => carry.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => return Err(invalid()),
        };
        consume(std::str::from_utf8(&carry[..valid]).map_err(|_| invalid())?)?;
        carry.drain(..valid);
    }
    if !carry.is_empty() {
        return Err(invalid());
    }
    Ok(())
}

pub(super) fn write<W: Write>(
    out: &mut W,
    until: Instant,
    range: &impl Fn(u64) -> io::Result<(Vec<u8>, bool)>,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    let check = |cancelled: &mut dyn FnMut() -> bool| {
        deadline(until)?;
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "Session reader cancelled",
            ));
        }
        Ok(())
    };
    let mut logical = Logical::new(range, until);
    let mut comparison = Compare {
        expected: Logical::new(range, until),
        changed: false,
    };
    let mut masker = Masker::new();
    chunks(&mut logical, |text| {
        check(cancelled)?;
        masker.push(text, &mut comparison)
    })?;
    masker.finish(&mut comparison)?;
    comparison.changed |= comparison.expected.read(&mut [0; 1])? != 0;
    if logical.non_scalar {
        if comparison.changed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "retained string requires an unavailable privacy transformation",
            ));
        }
        // Validation and privacy assessment completed before this exact replay.
        let mut offset = 0;
        loop {
            check(cancelled)?;
            let (bytes, eof) = range(offset)?;
            if bytes.len() > PIECE || (bytes.is_empty() && !eof) {
                return Err(invalid());
            }
            offset = offset.checked_add(bytes.len() as u64).ok_or_else(invalid)?;
            out.write_all(&bytes)?;
            if eof {
                break;
            }
        }
        return Ok(());
    }
    out.write_all(b"\"")?;
    let mut escaped = Escaped {
        out,
        bytes: Vec::with_capacity(PIECE),
        until,
    };
    let mut masker = Masker::new();
    chunks(&mut Logical::new(range, until), |text| {
        check(cancelled)?;
        masker.push(text, &mut escaped)
    })?;
    masker.finish(&mut escaped)?;
    escaped.emit(true)?;
    escaped.out.write_all(b"\"")
}

#[cfg(test)]
mod tests;
