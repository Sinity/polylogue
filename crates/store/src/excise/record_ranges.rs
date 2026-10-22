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
        while map.next_key::<IgnoredAny>()?.is_some() {
            let raw = map.next_value::<&RawValue>()?;
            let range = self.range(raw).map_err(A::Error::custom)?;
            if self.ranges.covered(range).map_err(A::Error::custom)? {
                return Err(A::Error::custom(
                    "object-member removal needs complete key and value boundaries",
                ));
            }
            if self.ranges.touched(range).map_err(A::Error::custom)? {
                self.descend(raw).map_err(A::Error::custom)?;
            }
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
