//! Reject duplicate keys before decoding settings with native JSON number handling.

use serde::Deserialize;
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use std::collections::HashSet;
use std::fmt;
use std::io;

pub(super) fn settings(text: &str) -> io::Result<serde_json::Value> {
    let mut input = serde_json::Deserializer::from_str(text);
    UniqueKeys::deserialize(&mut input).map_err(io::Error::other)?;
    input.end().map_err(io::Error::other)?;
    // The validation pass discards values. Value still owns number decoding,
    // including arbitrary-precision integers and native unknown fields.
    serde_json::from_str(text).map_err(io::Error::other)
}

struct UniqueKeys;

impl<'de> Deserialize<'de> for UniqueKeys {
    fn deserialize<D: de::Deserializer<'de>>(input: D) -> Result<Self, D::Error> {
        struct Keys;
        impl<'de> Visitor<'de> for Keys {
            type Value = UniqueKeys;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueKeys)
            }
            fn visit_bool<E: de::Error>(self, _: bool) -> Result<Self::Value, E> {
                Ok(UniqueKeys)
            }
            fn visit_i64<E: de::Error>(self, _: i64) -> Result<Self::Value, E> {
                Ok(UniqueKeys)
            }
            fn visit_u64<E: de::Error>(self, _: u64) -> Result<Self::Value, E> {
                Ok(UniqueKeys)
            }
            fn visit_f64<E: de::Error>(self, _: f64) -> Result<Self::Value, E> {
                Ok(UniqueKeys)
            }
            fn visit_str<E: de::Error>(self, _: &str) -> Result<Self::Value, E> {
                Ok(UniqueKeys)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut values: A) -> Result<Self::Value, A::Error> {
                while values.next_element::<UniqueKeys>()?.is_some() {}
                Ok(UniqueKeys)
            }
            fn visit_map<A: MapAccess<'de>>(self, mut entries: A) -> Result<Self::Value, A::Error> {
                let mut keys = HashSet::new();
                while let Some(key) = entries.next_key::<String>()? {
                    if !keys.insert(key.clone()) {
                        return Err(de::Error::custom(format!("duplicate settings key {key:?}")));
                    }
                    entries.next_value::<UniqueKeys>()?;
                }
                Ok(UniqueKeys)
            }
        }
        input.deserialize_any(Keys)
    }
}
