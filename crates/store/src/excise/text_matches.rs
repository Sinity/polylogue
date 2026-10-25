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

mod hits;
pub(super) use hits::{Collected, HitSink, Scope};

#[derive(Debug)]
pub(super) enum Error {
    Syntax(String),
    Sink(StoreError),
}

impl From<StoreError> for Error {
    fn from(error: StoreError) -> Self {
        Self::Sink(error)
    }
}

impl Error {
    fn into_store(self) -> StoreError {
        match self {
            Self::Syntax(message) => StoreError::Invalid(message),
            Self::Sink(error) => error,
        }
    }
}

type MatchResult<T> = std::result::Result<T, Error>;

pub(super) struct HitGroup {
    pub(super) scope: Scope,
    has_hits: bool,
}

impl HitGroup {
    fn new(sink: &mut impl HitSink) -> Result<Self> {
        Ok(Self {
            scope: sink.scope()?,
            has_hits: false,
        })
    }

    fn accept(self, parent: &mut Self, sink: &mut impl HitSink) -> Result<()> {
        sink.accept(self.scope, parent.scope)?;
        parent.has_hits |= self.has_hits;
        Ok(())
    }
}

pub(super) struct TextMatches {
    raw: Matcher,
    decoded: Matcher,
    output: HitGroup,
    raw_group: Option<HitGroup>,
    string: Option<HitGroup>,
    pending_key: Option<HitGroup>,
    json: Option<bool>,
    quoted: bool,
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

impl TextMatches {
    pub(super) fn new(needle: &str, sink: &mut impl HitSink) -> Result<Self> {
        if needle.is_empty() {
            return Err(StoreError::Invalid(
                "an excised text must not be empty".into(),
            ));
        }
        replacement(needle)?;
        Self::at_depth(Pattern::new(needle), 0, sink)
    }

    fn at_depth(pattern: Arc<Pattern>, depth: usize, sink: &mut impl HitSink) -> Result<Self> {
        Ok(Self {
            raw: Matcher::new(pattern.clone()),
            decoded: Matcher::new(pattern),
            output: HitGroup::new(sink)?,
            raw_group: Some(HitGroup::new(sink)?),
            string: None,
            pending_key: None,
            json: None,
            quoted: false,
            escape: Vec::new(),
            escape_start: 0,
            offset: 0,
            utf8_tail: Vec::new(),
            nested: None,
            nested_mode: None,
            depth,
            depth_limited: false,
            unhandled_key: false,
        })
    }

    pub(super) fn feed(&mut self, bytes: &[u8], sink: &mut impl HitSink) -> MatchResult<()> {
        let mut utf8 = std::mem::take(&mut self.utf8_tail);
        utf8.extend_from_slice(bytes);
        if let Err(e) = std::str::from_utf8(&utf8) {
            if e.error_len().is_some() {
                return Err(Error::Syntax("retained bytes are not UTF-8 text".into()));
            }
            self.utf8_tail.extend_from_slice(&utf8[e.valid_up_to()..]);
        }
        for &byte in bytes {
            let at = self.offset;
            self.offset += 1;
            self.unit(byte, at, at + 1, sink)?;
            self.check_key()?;
        }
        Ok(())
    }

    fn decoded_byte(
        &mut self,
        byte: u8,
        start: u64,
        end: u64,
        sink: &mut impl HitSink,
    ) -> MatchResult<()> {
        if let Some((start, end)) = self.decoded.push(byte, start, end) {
            let value = self.string.as_mut().expect("inside a string");
            sink.hit(value.scope, start, end)?;
            value.has_hits = true;
        }
        if self.nested_mode.is_none() && !byte.is_ascii_whitespace() {
            let json = matches!(byte, b'{' | b'[' | b'"');
            self.nested_mode = Some(json);
            if json {
                self.nested = Some(Box::new(Self::at_depth(
                    self.decoded.pattern.clone(),
                    self.depth + 1,
                    sink,
                )?));
            }
        }
        if let Some(nested) = &mut self.nested {
            let result = nested.unit(byte, start, end, sink);
            self.depth_limited |= nested.depth_limited;
            self.unhandled_key |= nested.unhandled_key;
            match result {
                Ok(()) => {}
                Err(Error::Syntax(_)) => {
                    // Invalid embedded JSON still contributes direct outer
                    // string matches, but none of its decoded hits.
                    self.nested.take().unwrap().discard(sink)?;
                }
                Err(error @ Error::Sink(_)) => return Err(error),
            }
        }
        Ok(())
    }

    fn unit(&mut self, byte: u8, at: u64, end: u64, sink: &mut impl HitSink) -> MatchResult<()> {
        if self.json != Some(true)
            && let Some((start, end)) = self.raw.push(byte, at, end)
        {
            let group = self.raw_group.as_mut().unwrap_or(&mut self.output);
            sink.hit(group.scope, start, end)?;
            group.has_hits = true;
        }
        if self.json.is_none() && !byte.is_ascii_whitespace() {
            self.json = Some(matches!(byte, b'{' | b'[' | b'"'));
            let raw = self.raw_group.take().unwrap();
            if self.json == Some(true) {
                sink.discard(raw.scope)?;
            } else {
                raw.accept(&mut self.output, sink)?;
            }
        }
        if self.json != Some(true) {
            return Ok(());
        }
        if self.depth > 128 {
            self.depth_limited = true;
            return Ok(());
        }
        if !self.quoted {
            if !byte.is_ascii_whitespace()
                && let Some(pending) = self.pending_key.take()
            {
                if byte == b':' {
                    self.unhandled_key |= pending.has_hits;
                    sink.discard(pending.scope)?;
                } else {
                    pending.accept(&mut self.output, sink)?;
                }
            }
            if byte == b'"' {
                self.quoted = true;
                self.nested_mode = None;
                self.string = Some(HitGroup::new(sink)?);
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
                Error::Syntax(format!(
                    "invalid JSON string escape at {}: {e}",
                    self.escape_start
                ))
            })?;
            for b in value.bytes() {
                self.decoded_byte(b, self.escape_start, end, sink)?;
            }
            self.escape.clear();
        } else if byte == b'\\' {
            self.escape.push(byte);
            self.escape_start = at;
        } else if byte == b'"' {
            if let Some(nested) = self.nested.take() {
                match nested.finish(sink) {
                    Ok(group) => group.accept(self.string.as_mut().unwrap(), sink)?,
                    Err(Error::Syntax(_)) => {}
                    Err(error @ Error::Sink(_)) => return Err(error),
                }
            }
            self.quoted = false;
            self.pending_key = self.string.take();
        } else {
            self.decoded_byte(byte, at, end, sink)?;
        }
        Ok(())
    }

    fn check_key(&self) -> MatchResult<()> {
        if self.unhandled_key {
            return Err(Error::Syntax("a JSON object key contains the excised text; it needs the owning format's rewrite to preserve structural meaning".into()));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn held_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self.raw.positions.capacity() + self.decoded.positions.capacity())
                * std::mem::size_of::<(u64, u64)>()
            + self.escape.capacity()
            + self.utf8_tail.capacity()
            + self.nested.as_ref().map_or(0, |nested| nested.held_bytes())
    }

    fn validate(&self) -> MatchResult<()> {
        self.check_key()?;
        if self.depth_limited {
            return Err(Error::Syntax(
                "embedded JSON string nesting exceeds 128 levels".into(),
            ));
        }
        if !self.utf8_tail.is_empty() {
            return Err(Error::Syntax(
                "retained UTF-8 text ends inside a character".into(),
            ));
        }
        if self.json == Some(true) && self.quoted {
            return Err(Error::Syntax(
                "retained JSON ends inside a string value".into(),
            ));
        }
        Ok(())
    }

    fn discard(self, sink: &mut impl HitSink) -> Result<()> {
        if let Some(nested) = self.nested {
            nested.discard(sink)?;
        }
        for group in [self.raw_group, self.string, self.pending_key]
            .into_iter()
            .flatten()
        {
            sink.discard(group.scope)?;
        }
        sink.discard(self.output.scope)
    }

    pub(super) fn finish(mut self, sink: &mut impl HitSink) -> MatchResult<HitGroup> {
        if let Err(error) = self.validate() {
            self.discard(sink)?;
            return Err(error);
        }
        if let Some(raw) = self.raw_group.take() {
            raw.accept(&mut self.output, sink)?;
        }
        if let Some(value) = self.pending_key.take() {
            value.accept(&mut self.output, sink)?;
        }
        Ok(self.output)
    }
}

#[cfg(test)]
pub(super) fn in_bytes(bytes: &[u8], needle: &str) -> Result<Vec<(u64, u64)>> {
    let mut scan = Collected::new(needle)?;
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
