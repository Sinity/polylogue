//! JSON record erasure without breaking array separators or decoding a value tree.
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde_json::value::RawValue;

type Range = (usize, usize);
mod file;
pub(super) use file::{Error, edge, plan_file};

#[cfg(test)]
pub(super) use tests::oracle::{Erasure, plan};

pub(in crate::excise) trait Ranges {
    fn covered(&mut self, range: Range) -> Result<bool, String>;
    fn touched(&mut self, range: Range) -> Result<bool, String>;
    fn mark(&mut self, range: Range);
    fn push(&mut self, range: Range) -> Result<(), String>;
}

struct Plan<'a, 'b, R> {
    bytes: &'a [u8],
    base: usize,
    ranges: &'b mut R,
}
impl<R: Ranges> Plan<'_, '_, R> {
    fn range(&self, raw: &RawValue) -> Result<Range, String> {
        let start = (raw.get().as_ptr() as usize)
            .checked_sub(self.bytes.as_ptr() as usize)
            .ok_or("JSON value does not borrow this source")?;
        let end = start + raw.get().len();
        if end > self.bytes.len() {
            return Err("JSON value is outside this source".into());
        }
        Ok((self.base + start, self.base + end))
    }
    fn comma(&self, start: usize, end: usize) -> Result<usize, String> {
        self.bytes[start - self.base..end - self.base]
            .iter()
            .position(|&b| b == b',')
            .map(|at| start + at)
            .ok_or_else(|| "array separator is missing".into())
    }
    fn member_start(&self, value_start: usize) -> Result<usize, String> {
        let mut at = value_start
            .checked_sub(self.base)
            .ok_or("object value precedes source")?;
        while at > 0 && self.bytes[at - 1].is_ascii_whitespace() {
            at -= 1;
        }
        if at == 0 || self.bytes[at - 1] != b':' {
            return Err("object value has no colon".into());
        }
        at -= 1;
        while at > 0 && self.bytes[at - 1].is_ascii_whitespace() {
            at -= 1;
        }
        if at == 0 || self.bytes[at - 1] != b'"' {
            return Err("object member has no quoted key".into());
        }
        at -= 1;
        while at > 0 {
            at -= 1;
            if self.bytes[at] != b'"' {
                continue;
            }
            let mut before = at;
            while before > 0 && self.bytes[before - 1] == b'\\' {
                before -= 1;
            }
            if (at - before) % 2 == 0 {
                return Ok(self.base + at);
            }
        }
        Err("object member key has no opening quote".into())
    }

    fn descend(&mut self, raw: &RawValue) -> Result<(), String> {
        if !matches!(raw.get().as_bytes().first(), Some(b'{' | b'[')) {
            return Err("source citation cuts through a JSON scalar".into());
        }
        let mut de = serde_json::Deserializer::from_str(raw.get());
        (&mut *self)
            .deserialize(&mut de)
            .map_err(|e| e.to_string())?;
        de.end().map_err(|e| e.to_string())
    }
}
impl<'de, R: Ranges> DeserializeSeed<'de> for &mut Plan<'_, '_, R> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_any(self)
    }
}
impl<'de, R: Ranges> Visitor<'de> for &mut Plan<'_, '_, R> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON container")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut previous_end = None;
        let mut run: Option<(usize, usize, Option<usize>)> = None;
        while map.next_key::<IgnoredAny>()?.is_some() {
            let raw = map.next_value::<&RawValue>()?;
            let range = self.range(raw).map_err(A::Error::custom)?;
            let start = self.member_start(range.0).map_err(A::Error::custom)?;
            if self.ranges.covered(range).map_err(A::Error::custom)? {
                self.ranges.mark(range);
                if let Some(run) = &mut run {
                    run.1 = range.1;
                } else {
                    let left = previous_end
                        .map(|end| self.comma(end, start))
                        .transpose()
                        .map_err(A::Error::custom)?;
                    run = Some((start, range.1, left));
                }
            } else {
                if let Some((start, end, _)) = run.take() {
                    let comma = self.comma(end, range.0).map_err(A::Error::custom)?;
                    self.ranges
                        .push((start, comma + 1))
                        .map_err(A::Error::custom)?;
                }
                if self.ranges.touched(range).map_err(A::Error::custom)? {
                    self.descend(raw).map_err(A::Error::custom)?;
                }
            }
            previous_end = Some(range.1);
        }
        if let Some((start, end, left)) = run {
            self.ranges
                .push((left.unwrap_or(start), end))
                .map_err(A::Error::custom)?;
        }
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        let mut previous_end = None;
        let mut run: Option<(usize, usize, Option<usize>)> = None;
        while let Some(raw) = seq.next_element::<&RawValue>()? {
            let range = self.range(raw).map_err(A::Error::custom)?;
            if self.ranges.covered(range).map_err(A::Error::custom)? {
                self.ranges.mark(range);
                if let Some(run) = &mut run {
                    run.1 = range.1;
                } else {
                    let left = previous_end
                        .map(|end| self.comma(end, range.0))
                        .transpose()
                        .map_err(A::Error::custom)?;
                    run = Some((range.0, range.1, left));
                }
            } else {
                if let Some((start, end, _)) = run.take() {
                    let comma = self.comma(end, range.0).map_err(A::Error::custom)?;
                    self.ranges
                        .push((start, comma + 1))
                        .map_err(A::Error::custom)?;
                }
                if self.ranges.touched(range).map_err(A::Error::custom)? {
                    self.descend(raw).map_err(A::Error::custom)?;
                }
            }
            previous_end = Some(range.1);
        }
        if let Some((start, end, left)) = run {
            self.ranges
                .push((left.unwrap_or(start), end))
                .map_err(A::Error::custom)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
