//! Direct donors of the effective record's immediate fields. Opaque nested
//! values replace as units, so a nested leaf extends its original donor path.
//! None is unavailable provenance, never a fabricated effective-overlay span.
use crate::selection::{PathPart, RecordRange, SelectedRaw};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecordOrigins {
    pub envelope: Option<RecordRange>,
    /// Qualified component vectors avoid collisions with literal dotted keys.
    #[serde(with = "field_entries")]
    pub fields: BTreeMap<Vec<String>, SelectedRaw>,
}
impl RecordOrigins {
    pub(super) fn overlay(
        old: Option<&Self>,
        incoming: &Value,
        effective: &Value,
        range: Option<RecordRange>,
    ) -> Self {
        let mut result = Self {
            envelope: range,
            fields: BTreeMap::new(),
        };
        let Some(object) = effective.as_object() else {
            return result;
        };
        for (name, value) in object {
            if name == "message" && value.is_object() {
                for field in value.as_object().into_iter().flat_map(|m| m.keys()) {
                    let key = vec![name.clone(), field.clone()];
                    let supplied = incoming
                        .get("message")
                        .and_then(|m| m.as_object())
                        .is_some_and(|m| m.contains_key(field));
                    if let Some(origin) = donor(old, &key, supplied, range) {
                        result.fields.insert(key, origin);
                    }
                }
            } else {
                let key = vec![name.clone()];
                let supplied = incoming.as_object().is_some_and(|m| m.contains_key(name));
                if let Some(origin) = donor(old, &key, supplied, range) {
                    result.fields.insert(key, origin);
                }
            }
        }
        result
    }
    pub fn select(&self, path: &[PathPart]) -> Option<SelectedRaw> {
        let PathPart::Key(first) = path.first()? else {
            return None;
        };
        let mut key = vec![first.clone()];
        if first == "message" {
            if let Some(PathPart::Key(second)) = path.get(1) {
                let qualified = vec![first.clone(), second.clone()];
                if self.fields.contains_key(&qualified) {
                    key = qualified;
                }
            }
        }
        let mut selected = self.fields.get(&key)?.clone();
        selected.path.extend_from_slice(&path[key.len()..]);
        Some(selected)
    }
    /// Scalar keyset cursor over this one admitted map, not a response scan.
    pub fn next(&self, after: Option<&[String]>) -> Option<(&[String], &SelectedRaw)> {
        use std::ops::Bound::{Excluded, Unbounded};
        let entry = match after {
            None => self.fields.iter().next(),
            Some(key) => self
                .fields
                .range::<[String], _>((Excluded(key), Unbounded))
                .next(),
        };
        entry.map(|(key, value)| (key.as_slice(), value))
    }
}
fn donor(
    old: Option<&RecordOrigins>,
    key: &[String],
    supplied: bool,
    range: Option<RecordRange>,
) -> Option<SelectedRaw> {
    if supplied {
        Some(SelectedRaw {
            record: range?,
            path: key.iter().cloned().map(PathPart::Key).collect(),
        })
    } else {
        old?.fields.get(key).cloned()
    }
}
pub(super) fn typed_path(root: &Value, path: &[String]) -> Option<Vec<PathPart>> {
    let mut current = root;
    let mut result = Vec::new();
    for part in path {
        if let Some(array) = current.as_array() {
            let index = part.parse::<u64>().ok()?;
            current = array.get(usize::try_from(index).ok()?)?;
            result.push(PathPart::Index(index));
        } else {
            current = current.as_object()?.get(part)?;
            result.push(PathPart::Key(part.clone()));
        }
    }
    Some(result)
}

// JSON object keys cannot encode typed paths. A sequence retains the component
// boundaries and rejects repeated keys instead of silently selecting a donor.
mod field_entries {
    use super::*;
    use serde::{
        Deserializer, Serializer,
        de::{Error, SeqAccess, Visitor},
        ser::SerializeSeq,
    };
    pub fn serialize<S: Serializer>(
        fields: &BTreeMap<Vec<String>, SelectedRaw>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(fields.len()))?;
        for entry in fields {
            sequence.serialize_element(&entry)?;
        }
        sequence.end()
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<BTreeMap<Vec<String>, SelectedRaw>, D::Error> {
        struct Entries;
        impl<'de> Visitor<'de> for Entries {
            type Value = BTreeMap<Vec<String>, SelectedRaw>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("unique typed Claude origin path entries")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut fields = BTreeMap::new();
                while let Some((key, value)) =
                    sequence.next_element::<(Vec<String>, SelectedRaw)>()?
                {
                    if fields.insert(key, value).is_some() {
                        return Err(A::Error::custom("duplicate Claude origin path"));
                    }
                }
                Ok(fields)
            }
        }
        deserializer.deserialize_seq(Entries)
    }
}
