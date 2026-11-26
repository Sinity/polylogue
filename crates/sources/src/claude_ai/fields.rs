//! A sidecar object as borrowed raw JSON values by field name, refusing a repeated field.
/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use serde::de::{DeserializeSeed, Error as _, MapAccess, Visitor};
use serde_json::value::RawValue;
use std::collections::BTreeMap;

pub(super) fn object(bytes: &[u8]) -> Result<BTreeMap<String, &RawValue>, String> {
    struct Fields;
    impl<'de> DeserializeSeed<'de> for Fields {
        type Value = BTreeMap<String, &'de RawValue>;
        fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Self::Value, D::Error> {
            de.deserialize_map(self)
        }
    }
    impl<'de> Visitor<'de> for Fields {
        type Value = BTreeMap<String, &'de RawValue>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON object with unique fields")
        }
        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut fields = BTreeMap::new();
            while let Some(key) = map.next_key::<String>()? {
                let raw = map.next_value::<&'de RawValue>()?;
                if fields.insert(key, raw).is_some() {
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

pub(super) fn text(fields: &BTreeMap<String, &RawValue>, name: &str) -> Option<String> {
    serde_json::from_str(fields.get(name)?.get()).ok()
}
