//! Stream selected object fields, array elements or map values while retaining
//! only named metadata. Selected text never enters the metadata object.
use super::*;
use polylogue_model::Json;
use serde::de::{Error as _, IgnoredAny, MapAccess};
use std::collections::HashSet;

#[derive(Clone, Copy)]
pub(crate) enum FieldShape {
    Value,
    Array,
    Map,
}
pub(crate) struct ObjectFields {
    pub start: u64,
    pub fields: Json,
}

pub(crate) fn scan_objects(
    reader: impl Read,
    array: bool,
    frames: &[(&str, FieldShape)],
    keep: &[&str],
    mut item: impl FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<(Vec<ObjectFields>, ScanStats), ScanError> {
    let tap = Rc::new(RefCell::new(Tap::default()));
    let input = Tapped {
        reader: BufReader::new(reader),
        tap: Rc::clone(&tap),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    let mut objects = Vec::new();
    let object = Object {
        tap: &tap,
        frames,
        keep,
        item: &mut item,
    };
    let result = if array {
        Objects {
            object,
            out: &mut objects,
        }
        .deserialize(&mut de)
    } else {
        object
            .deserialize(&mut de)
            .map(|object| objects.push(object))
    };
    result.map_err(|error| scan_error(error, &tap))?;
    let stats = finish_scan(de.end(), &tap)?;
    Ok((objects, stats))
}

struct Objects<'a, 'b, F> {
    object: Object<'a, F>,
    out: &'b mut Vec<ObjectFields>,
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    DeserializeSeed<'de> for Objects<'_, '_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_seq(self)
    }
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    Visitor<'de> for Objects<'_, '_, F>
{
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an array of export objects")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        while let Some(fields) = seq.next_element_seed(Object {
            tap: self.object.tap,
            frames: self.object.frames,
            keep: self.object.keep,
            item: &mut *self.object.item,
        })? {
            self.out.push(fields);
        }
        Ok(())
    }
}
struct Object<'a, F> {
    tap: &'a Rc<RefCell<Tap>>,
    frames: &'a [(&'a str, FieldShape)],
    keep: &'a [&'a str],
    item: &'a mut F,
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    DeserializeSeed<'de> for Object<'_, F>
{
    type Value = ObjectFields;
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Self::Value, D::Error> {
        de.deserialize_map(self)
    }
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    Visitor<'de> for Object<'_, F>
{
    type Value = ObjectFields;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an export object with unique fields")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ObjectFields, A::Error> {
        let Some((start, b'{')) = self.tap.borrow().last else {
            return Err(A::Error::custom("missing export object boundary"));
        };
        let mut fields = serde_json::Map::new();
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom("duplicate export object field"));
            }
            if let Some((_, shape)) = self.frames.iter().find(|(name, _)| *name == key) {
                let mut emit = |range, raw: &RawValue| (self.item)(start, &key, None, range, raw);
                match shape {
                    FieldShape::Value => map_value(&mut map, self.tap, &mut emit)?,
                    FieldShape::Array => map.next_value_seed(ArraySeed {
                        tap: self.tap,
                        item: &mut emit,
                    })?,
                    FieldShape::Map => map.next_value_seed(MapItems {
                        tap: self.tap,
                        parent: start,
                        field: &key,
                        item: &mut *self.item,
                    })?,
                }
                fields.insert(key, Json::Null);
            } else if self.keep.contains(&key.as_str()) {
                let before = self.tap.borrow().position;
                let value = map.next_value::<Json>()?;
                let bytes = (self.tap.borrow().position - before) as usize;
                {
                    let mut tap = self.tap.borrow_mut();
                    tap.stats.peak_element_bytes = tap.stats.peak_element_bytes.max(bytes);
                }
                fields.insert(key, value);
            } else {
                map.next_value::<IgnoredAny>()?;
                fields.insert(key, Json::Null);
            }
        }
        Ok(ObjectFields {
            start,
            fields: Json::Object(fields),
        })
    }
}
struct MapItems<'a, F> {
    tap: &'a Rc<RefCell<Tap>>,
    parent: u64,
    field: &'a str,
    item: &'a mut F,
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    DeserializeSeed<'de> for MapItems<'_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_map(self)
    }
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    Visitor<'de> for MapItems<'_, F>
{
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an export field map")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom("duplicate export map key"));
            }
            map_value(&mut map, self.tap, &mut |range, raw| {
                (self.item)(self.parent, self.field, Some(&key), range, raw)
            })?;
        }
        Ok(())
    }
}
