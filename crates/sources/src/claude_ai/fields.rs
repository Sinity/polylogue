//! Borrowed JSON fields keep each value's exact byte range.
use serde::de::{DeserializeSeed, Error as _, MapAccess, Visitor};
use serde_json::value::RawValue;
use std::collections::BTreeMap;

pub(super) struct Field<'a> {
    pub raw: &'a RawValue,
}

pub(super) fn object(bytes: &[u8]) -> Result<BTreeMap<String, Field<'_>>, String> {
    struct Fields;
    impl<'de> DeserializeSeed<'de> for Fields {
        type Value = BTreeMap<String, Field<'de>>;
        fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Self::Value, D::Error> {
            de.deserialize_map(self)
        }
    }
    impl<'de> Visitor<'de> for Fields {
        type Value = BTreeMap<String, Field<'de>>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON object with unique fields")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut fields = BTreeMap::new();
            while let Some(key) = map.next_key::<String>()? {
                let raw = map.next_value::<&'de RawValue>()?;
                if fields.insert(key, Field { raw }).is_some() {
                    return Err(A::Error::custom("duplicate sidecar field"));
                }
            }
            Ok(fields)
        }
    }
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let fields = Fields.deserialize(&mut de).map_err(|e| e.to_string())?;
    de.end().map_err(|e| e.to_string())?;
    Ok(fields)
}

pub(super) fn text(fields: &BTreeMap<String, Field<'_>>, name: &str) -> Option<String> {
    serde_json::from_str(fields.get(name)?.raw.get()).ok()
}
