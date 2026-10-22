//! Validated historical manifests, borrowing their arrays from one SQLite value.
use super::super::{Replacement, Unit, batch_hash, digest};
use crate::{Result, StoreError};
use serde::de::{DeserializeOwned, DeserializeSeed, SeqAccess, Visitor};
use serde::ser::{Error as _, SerializeSeq};
use serde::{Deserialize, Serialize, Serializer};
use serde_json::value::RawValue;
use std::marker::PhantomData;

#[derive(Deserialize)]
struct Fields<'a> {
    version: u32,
    #[serde(borrow)]
    replacements: &'a RawValue,
    #[serde(borrow)]
    retired_batches: &'a RawValue,
    digest: String,
}

pub(in crate::sinex_rows) struct View<'a>(Fields<'a>);

impl<'a> View<'a> {
    pub(in crate::sinex_rows) fn decode(json: &'a str) -> Result<Option<Self>> {
        super::raw(json)?
            .map(|raw| {
                let view = Self(serde_json::from_str(raw.get())?);
                if view.0.version != 1 {
                    return Err(StoreError::Invalid(
                        "excision replacement manifest failed its version/hash check".into(),
                    ));
                }
                let mut previous: Option<Unit> = None;
                view.each_replacement(|replacement| {
                    replacement.validate(previous.as_ref())?;
                    previous = Some(replacement.unit.clone());
                    Ok(())
                })?;
                each::<String>(view.0.retired_batches, |name| {
                    batch_hash(&name)?;
                    Ok(())
                })?;
                if view.computed_digest()? != view.0.digest {
                    return Err(StoreError::Invalid(
                        "excision replacement manifest failed its version/hash check".into(),
                    ));
                }
                Ok(view)
            })
            .transpose()
    }

    /// Decode one entry at a time, only after the complete view was validated.
    pub(in crate::sinex_rows) fn each_replacement(
        &self,
        mut visit: impl FnMut(&Replacement) -> Result<()>,
    ) -> Result<()> {
        each(self.0.replacements, |replacement| visit(&replacement))
    }

    fn computed_digest(&self) -> Result<String> {
        // Match the existing digest's sorted field order and typed normalization.
        #[derive(Serialize)]
        struct Manifest<'a> {
            replacements: Replacements<'a>,
            retired_batches: Strings<'a>,
            version: u32,
        }
        digest::hash(&Manifest {
            replacements: Replacements(self.0.replacements),
            retired_batches: Strings(self.0.retired_batches),
            version: self.0.version,
        })
    }
}

struct Replacements<'a>(&'a RawValue);
impl Serialize for Replacements<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        each::<Replacement>(self.0, |replacement| {
            sequence
                .serialize_element(&digest::SortedReplacement::from(&replacement))
                .map_err(|error| StoreError::Invalid(error.to_string()))
        })
        .map_err(S::Error::custom)?;
        sequence.end()
    }
}

struct Strings<'a>(&'a RawValue);
impl Serialize for Strings<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        each::<String>(self.0, |name| {
            sequence
                .serialize_element(&name)
                .map_err(|error| StoreError::Invalid(error.to_string()))
        })
        .map_err(S::Error::custom)?;
        sequence.end()
    }
}

fn each<T: DeserializeOwned>(
    json: &RawValue,
    mut visit: impl FnMut(T) -> Result<()>,
) -> Result<()> {
    let mut error = None;
    let mut de = serde_json::Deserializer::from_str(json.get());
    let result = Entries::<T, _> {
        visit: &mut visit,
        error: &mut error,
        item: PhantomData,
    }
    .deserialize(&mut de);
    if let Some(error) = error {
        return Err(error);
    }
    result?;
    de.end()?;
    Ok(())
}

struct Entries<'a, T, F> {
    visit: &'a mut F,
    error: &'a mut Option<StoreError>,
    item: PhantomData<T>,
}
impl<'de, T: DeserializeOwned, F: FnMut(T) -> Result<()>> DeserializeSeed<'de>
    for Entries<'_, T, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> std::result::Result<(), D::Error> {
        de.deserialize_seq(self)
    }
}
impl<'de, T: DeserializeOwned, F: FnMut(T) -> Result<()>> Visitor<'de> for Entries<'_, T, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an array of manifest entries")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> std::result::Result<(), A::Error> {
        while let Some(value) = sequence.next_element()? {
            if let Err(error) = (self.visit)(value) {
                *self.error = Some(error);
                return Err(serde::de::Error::custom("manifest entry refused"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
