//! Visit a canonical deletion array without copying the complete hash set.
use crate::{Result, StoreError};
use serde::de::{DeserializeSeed, SeqAccess, Visitor};
use serde::ser::{SerializeSeq, Serializer};
use std::io::{self, Write};

pub(super) fn visit(
    version: i64,
    json: &str,
    mut visit: impl FnMut(&[u8; 32]) -> Result<()>,
) -> Result<()> {
    if version != 1 {
        return Err(super::invalid("unsupported vector deletion policy"));
    }
    let mut error = None;
    let mut de = serde_json::Deserializer::from_str(json);
    let result = Hashes {
        json,
        visit: &mut visit,
        error: &mut error,
    }
    .deserialize(&mut de);
    if let Some(error) = error {
        return Err(error);
    }
    result?;
    de.end()?;
    Ok(())
}

struct Hashes<'a, F> {
    json: &'a str,
    visit: &'a mut F,
    error: &'a mut Option<StoreError>,
}
impl<'de, F: FnMut(&[u8; 32]) -> Result<()>> DeserializeSeed<'de> for Hashes<'_, F> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> std::result::Result<(), D::Error> {
        de.deserialize_seq(self)
    }
}
impl<'de, F: FnMut(&[u8; 32]) -> Result<()>> Visitor<'de> for Hashes<'_, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a canonical sorted array of deletion hashes")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> std::result::Result<(), A::Error> {
        let mut canonical = Canonical(self.json.as_bytes());
        let mut serializer = serde_json::Serializer::new(&mut canonical);
        let mut encoded = serializer
            .serialize_seq(None)
            .map_err(|_| noncanonical(self.error))?;
        let mut previous = None;
        while let Some(name) = sequence.next_element::<String>()? {
            let hash = crate::sinex_rows::unhex(&name)
                .and_then(|bytes| {
                    <[u8; 32]>::try_from(bytes)
                        .map_err(|_| super::invalid("vector deletion hash is not 32 bytes"))
                })
                .map_err(|error| refused(self.error, error))?;
            if crate::sinex_rows::hex(&hash) != name
                || previous.is_some_and(|previous| previous >= hash)
            {
                return Err(refused(
                    self.error,
                    super::invalid("vector deletion hashes are not canonical sorted unique hashes"),
                ));
            }
            encoded
                .serialize_element(&name)
                .map_err(|_| noncanonical(self.error))?;
            (self.visit)(&hash).map_err(|error| refused(self.error, error))?;
            previous = Some(hash);
        }
        encoded.end().map_err(|_| noncanonical(self.error))?;
        if !canonical.0.is_empty() {
            return Err(noncanonical(self.error));
        }
        Ok(())
    }
}

fn refused<E: serde::de::Error>(slot: &mut Option<StoreError>, error: StoreError) -> E {
    *slot = Some(error);
    E::custom("vector deletion hash refused")
}
fn noncanonical<E: serde::de::Error>(slot: &mut Option<StoreError>) -> E {
    refused(
        slot,
        super::invalid("vector deletion policy JSON is not canonical"),
    )
}

/// Compare serde's normal encoding directly with the retained bytes.
struct Canonical<'a>(&'a [u8]);
impl Write for Canonical<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0 = self
            .0
            .strip_prefix(bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "noncanonical JSON"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
