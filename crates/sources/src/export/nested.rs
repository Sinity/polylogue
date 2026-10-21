//! A named array inside a JSON object. Serde visits each element; the returned
//! metadata keeps the surrounding fields with that array replaced by `[]`.
use super::*;
use polylogue_model::Json;
use serde::de::{Error as _, MapAccess};

pub fn scan_nested(
    reader: impl Read,
    path: &[&str],
    item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<(Json, ScanStats), ScanError> {
    scan_metadata(reader, path, false, false, item)
}

/// A named array whose enclosing objects must have unique field names.
/// Use this for typed wrappers; record schemas remain the caller's concern.
pub fn scan_nested_unique(
    reader: impl Read,
    path: &[&str],
    item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<(Json, ScanStats), ScanError> {
    scan_metadata(reader, path, false, true, item)
}

/// Accepts either a bare record array or the named array in a wrapper object.
/// A bare array returns `[]` as metadata; a wrapper retains its other fields.
pub fn scan_array_or_nested(
    reader: impl Read,
    path: &[&str],
    item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<(Json, ScanStats), ScanError> {
    scan_metadata(reader, path, true, false, item)
}

fn scan_metadata(
    reader: impl Read,
    path: &[&str],
    root_array: bool,
    unique_fields: bool,
    mut item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<(Json, ScanStats), ScanError> {
    let tap = Rc::new(RefCell::new(Tap::default()));
    let input = Tapped {
        reader: BufReader::new(reader),
        tap: Rc::clone(&tap),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    let result = Nested {
        path,
        root_array,
        unique_fields,
        tap: &tap,
        item: &mut item,
    }
    .deserialize(&mut de);
    let metadata = result.map_err(|error| scan_error(error, &tap))?;
    let stats = finish_scan(de.end(), &tap)?;
    Ok((metadata, stats))
}

struct Nested<'a, F> {
    path: &'a [&'a str],
    root_array: bool,
    unique_fields: bool,
    tap: &'a Rc<RefCell<Tap>>,
    item: &'a mut F,
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> DeserializeSeed<'de>
    for Nested<'_, F>
{
    type Value = Json;
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Json, D::Error> {
        de.deserialize_any(self)
    }
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> Visitor<'de> for Nested<'_, F> {
    type Value = Json;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("JSON metadata or the selected array")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut value = serde_json::Map::new();
        let mut selected = false;
        while let Some(key) = map.next_key::<String>()? {
            if self.unique_fields && value.contains_key(&key) {
                return Err(A::Error::custom("duplicate export metadata field"));
            }
            let child = if self.path.first() == Some(&key.as_str()) {
                if selected {
                    return Err(A::Error::custom("duplicate streamed-array field"));
                }
                selected = true;
                map.next_value_seed(Nested {
                    path: &self.path[1..],
                    root_array: false,
                    unique_fields: self.unique_fields,
                    tap: self.tap,
                    item: self.item,
                })?
            } else {
                map.next_value::<Json>()?
            };
            value.insert(key, child);
        }
        Ok(Json::Object(value))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        if self.path.is_empty() || self.root_array {
            ArraySeed {
                tap: self.tap,
                item: self.item,
            }
            .visit_seq(seq)?;
            return Ok(Json::Array(Vec::new()));
        }
        let mut items = Vec::new();
        while let Some(value) = seq.next_element::<Json>()? {
            items.push(value);
        }
        Ok(Json::Array(items))
    }
    fn visit_bool<E: serde::de::Error>(self, v: bool) -> Result<Json, E> {
        Ok(v.into())
    }
    fn visit_i64<E: serde::de::Error>(self, v: i64) -> Result<Json, E> {
        Ok(v.into())
    }
    fn visit_u64<E: serde::de::Error>(self, v: u64) -> Result<Json, E> {
        Ok(v.into())
    }
    fn visit_f64<E: serde::de::Error>(self, v: f64) -> Result<Json, E> {
        Ok(Json::from(v))
    }
    fn visit_str<E: serde::de::Error>(self, v: &str) -> Result<Json, E> {
        Ok(v.into())
    }
    fn visit_string<E: serde::de::Error>(self, v: String) -> Result<Json, E> {
        Ok(v.into())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Json, E> {
        Ok(Json::Null)
    }
}
