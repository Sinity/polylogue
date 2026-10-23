//! Matching decoded JSON string values without buffering their enclosing record.

use crate::{Result, StoreError};
use std::collections::VecDeque;
use std::sync::Arc;

struct Pattern {
    needle: Vec<u8>,
    prefix: Vec<usize>,
}

impl Pattern {
    fn new(needle: &str) -> Arc<Self> {
        let needle = needle.as_bytes().to_vec();
        let mut prefix = vec![0; needle.len()];
        for i in 1..needle.len() {
            let mut n = prefix[i - 1];
            while n > 0 && needle[i] != needle[n] {
                n = prefix[n - 1];
            }
            if needle[i] == needle[n] {
                n += 1;
            }
            prefix[i] = n;
        }
        Arc::new(Self { needle, prefix })
    }
}

struct Matcher {
    pattern: Arc<Pattern>,
    matched: usize,
    positions: VecDeque<(u64, u64)>,
}

impl Matcher {
    fn new(pattern: Arc<Pattern>) -> Self {
        Self {
            pattern,
            matched: 0,
            positions: VecDeque::new(),
        }
    }

    fn reset(&mut self) {
        self.matched = 0;
        self.positions.clear();
    }

    fn push(&mut self, byte: u8, start: u64, end: u64) -> Option<(u64, u64)> {
        self.positions.push_back((start, end));
        if self.positions.len() > self.pattern.needle.len() {
            self.positions.pop_front();
        }
        while self.matched > 0 && byte != self.pattern.needle[self.matched] {
            self.matched = self.pattern.prefix[self.matched - 1];
        }
        if byte == self.pattern.needle[self.matched] {
            self.matched += 1;
        }
        if self.matched != self.pattern.needle.len() {
            return None;
        }
        self.matched = self.pattern.prefix[self.matched - 1];
        Some((self.positions.front().unwrap().0, end))
    }
}

pub(super) struct TextMatches {
    raw: Matcher,
    decoded: Matcher,
    raw_ranges: Vec<(u64, u64)>,
    decoded_ranges: Vec<(u64, u64)>,
    json: Option<bool>,
    quoted: bool,
    string_ranges_start: usize,
    pending_key: Option<usize>,
    escape: Vec<u8>,
    escape_start: u64,
    offset: u64,
    utf8_tail: Vec<u8>,
    nested: Option<Box<TextMatches>>,
    nested_mode: Option<bool>,
    depth: usize,
    depth_limited: bool,
    unhandled_key: bool,
}

/// Raw hits before the first non-whitespace byte remain tentative: later JSON
/// selection discards them. JSON batches contain only completed value hits.
pub(super) enum Matches {
    Undecided(Vec<(u64, u64)>),
    Plain(Vec<(u64, u64)>),
    Json(Vec<(u64, u64)>),
}

impl TextMatches {
    pub(super) fn new(needle: &str) -> Result<Self> {
        if needle.is_empty() {
            return Err(StoreError::Invalid(
                "an excised text must not be empty".into(),
            ));
        }
        replacement(needle)?;
        Ok(Self::at_depth(Pattern::new(needle), 0))
    }

    fn at_depth(pattern: Arc<Pattern>, depth: usize) -> Self {
        Self {
            raw: Matcher::new(pattern.clone()),
            decoded: Matcher::new(pattern),
            raw_ranges: Vec::new(),
            decoded_ranges: Vec::new(),
            json: None,
            quoted: false,
            string_ranges_start: 0,
            pending_key: None,
            escape: Vec::new(),
            escape_start: 0,
            offset: 0,
            utf8_tail: Vec::new(),
            nested: None,
            nested_mode: None,
            depth,
            depth_limited: false,
            unhandled_key: false,
        }
    }

    pub(super) fn plain(needle: &str) -> Result<Self> {
        let mut s = Self::new(needle)?;
        s.json = Some(false);
        Ok(s)
    }

    pub(super) fn feed(&mut self, bytes: &[u8]) -> Result<()> {
        let mut utf8 = std::mem::take(&mut self.utf8_tail);
        utf8.extend_from_slice(bytes);
        if let Err(e) = std::str::from_utf8(&utf8) {
            if e.error_len().is_some() {
                return Err(StoreError::Invalid(
                    "retained bytes are not UTF-8 text".into(),
                ));
            }
            self.utf8_tail.extend_from_slice(&utf8[e.valid_up_to()..]);
        }
        for &byte in bytes {
            let at = self.offset;
            self.offset += 1;
            self.unit(byte, at, at + 1)?;
            self.check_key()?;
        }
        Ok(())
    }

    fn decoded_byte(&mut self, byte: u8, start: u64, end: u64) {
        if let Some(range) = self.decoded.push(byte, start, end) {
            self.decoded_ranges.push(range);
        }
        if self.nested_mode.is_none() && !byte.is_ascii_whitespace() {
            let json = matches!(byte, b'{' | b'[' | b'"');
            self.nested_mode = Some(json);
            if json {
                self.nested = Some(Box::new(Self::at_depth(
                    self.decoded.pattern.clone(),
                    self.depth + 1,
                )));
            }
        }
        if let Some(nested) = &mut self.nested {
            let ok = nested.unit(byte, start, end).is_ok();
            self.depth_limited |= nested.depth_limited;
            self.unhandled_key |= nested.unhandled_key;
            // A prose value may begin with an incomplete JSON example. It is
            // still searched directly; only valid embedded JSON adds matches.
            if !ok {
                self.nested = None;
            }
        }
    }

    fn unit(&mut self, byte: u8, at: u64, end: u64) -> Result<()> {
        if self.json != Some(true)
            && let Some(range) = self.raw.push(byte, at, end)
        {
            self.raw_ranges.push(range);
        }
        if self.json.is_none() && !byte.is_ascii_whitespace() {
            self.json = Some(matches!(byte, b'{' | b'[' | b'"'));
            if self.json == Some(true) {
                self.raw_ranges.clear();
            }
        }
        if self.json != Some(true) {
            return Ok(());
        }
        // Like serde_json, refuse excessive nesting instead of silently
        // missing payloads encoded repeatedly as JSON strings.
        if self.depth > 128 {
            self.depth_limited = true;
            return Ok(());
        }
        if !self.quoted {
            if !byte.is_ascii_whitespace()
                && let Some(start) = self.pending_key.take()
                && byte == b':'
            {
                self.unhandled_key |= self.decoded_ranges.len() > start;
                self.decoded_ranges.truncate(start);
            }
            if byte == b'"' {
                self.quoted = true;
                self.nested = None;
                self.nested_mode = None;
                self.string_ranges_start = self.decoded_ranges.len();
                self.decoded.reset();
            }
            return Ok(());
        }
        if !self.escape.is_empty() {
            self.escape.push(byte);
            let n = self.escape.len();
            let unicode = self.escape.get(1) == Some(&b'u');
            if unicode && n < 6 {
                return Ok(());
            }
            if unicode && n == 6 {
                let high = std::str::from_utf8(&self.escape[2..])
                    .ok()
                    .and_then(|s| u16::from_str_radix(s, 16).ok());
                if high.is_some_and(|u| (0xd800..=0xdbff).contains(&u)) {
                    return Ok(());
                }
            }
            if unicode && n > 6 && n < 12 {
                return Ok(());
            }
            let mut token = vec![b'"'];
            token.extend_from_slice(&self.escape);
            token.push(b'"');
            let value: String = serde_json::from_slice(&token).map_err(|e| {
                StoreError::Invalid(format!(
                    "invalid JSON string escape at {}: {e}",
                    self.escape_start
                ))
            })?;
            for b in value.bytes() {
                self.decoded_byte(b, self.escape_start, end);
            }
            self.escape.clear();
        } else if byte == b'\\' {
            self.escape.push(byte);
            self.escape_start = at;
        } else if byte == b'"' {
            if let Some(nested) = self.nested.take()
                && let Ok(ranges) = nested.finish()
            {
                self.decoded_ranges.extend(ranges);
            }
            self.quoted = false;
            self.pending_key = Some(self.string_ranges_start);
        } else {
            self.decoded_byte(byte, at, end);
        }
        Ok(())
    }

    fn check_key(&self) -> Result<()> {
        if self.unhandled_key {
            return Err(StoreError::Invalid("a JSON object key contains the excised text; it needs the owning format's rewrite to preserve structural meaning".into()));
        }
        Ok(())
    }

    /// Release irrevocable hits without retaining all earlier values. A current
    /// string and its nested JSON stay tentative until value/key classification.
    pub(super) fn drain(&mut self) -> Matches {
        match self.json {
            None => Matches::Undecided(std::mem::take(&mut self.raw_ranges)),
            Some(false) => Matches::Plain(std::mem::take(&mut self.raw_ranges)),
            Some(true) => {
                let ready = if self.quoted {
                    self.string_ranges_start
                } else {
                    self.pending_key.unwrap_or(self.decoded_ranges.len())
                };
                let ranges = self.decoded_ranges.drain(..ready).collect();
                self.string_ranges_start = self.string_ranges_start.saturating_sub(ready);
                if let Some(start) = &mut self.pending_key {
                    *start -= ready;
                }
                Matches::Json(ranges)
            }
        }
    }

    pub(super) fn complete(self) -> Result<Matches> {
        self.check_key()?;
        if self.depth_limited {
            return Err(StoreError::Invalid(
                "embedded JSON string nesting exceeds 128 levels".into(),
            ));
        }
        if !self.utf8_tail.is_empty() {
            return Err(StoreError::Invalid(
                "retained UTF-8 text ends inside a character".into(),
            ));
        }
        if self.json == Some(true) && self.quoted {
            return Err(StoreError::Invalid(
                "retained JSON ends inside a string value".into(),
            ));
        }
        Ok(if self.json == Some(true) {
            Matches::Json(self.decoded_ranges)
        } else {
            Matches::Plain(self.raw_ranges)
        })
    }

    /// Whole-value users already own the value, and need its final hit list.
    pub(super) fn finish(self) -> Result<Vec<(u64, u64)>> {
        let mut ranges = match self.complete()? {
            Matches::Undecided(_) => unreachable!("EOF chooses plain text"),
            Matches::Plain(ranges) | Matches::Json(ranges) => ranges,
        };
        ranges.sort_unstable();
        ranges.dedup();
        Ok(ranges)
    }
}

#[cfg(test)]
pub(super) fn in_bytes(bytes: &[u8], needle: &str) -> Result<Vec<(u64, u64)>> {
    let mut scan = TextMatches::new(needle)?;
    scan.feed(bytes)?;
    scan.finish()
}

#[cfg(test)]
mod tests;

/// An inserted byte absent from the target cannot create a new occurrence at
/// either edge of a replaced span. Quotes and backslashes would break JSON.
pub(super) fn replacement(needle: &str) -> Result<u8> {
    if needle.bytes().all(|b| b == b'\n') {
        return Err(StoreError::Invalid(
            "an excised text must contain a byte other than a line break".into(),
        ));
    }
    (*b" *")
        .into_iter()
        .chain(b'!'..=b'~')
        .find(|b| *b != b'"' && *b != b'\\' && !needle.as_bytes().contains(b))
        .ok_or_else(|| {
            StoreError::Invalid(
                "this text contains every safe in-place replacement byte; choose a narrower text"
                    .into(),
            )
        })
}
