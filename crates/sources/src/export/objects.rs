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
    Objects,
}
pub(crate) struct ObjectFields {
    pub start: u64,
    pub end: u64,
    pub parent: Option<u64>,
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
    let objects = RefCell::new(Vec::new());
    let object = Object {
        tap: &tap,
        frames,
        keep,
        item: &mut item,
        objects: &objects,
        parent: None,
    };
    let result = if array {
        Objects { object }.deserialize(&mut de)
    } else {
        object
            .deserialize(&mut de)
            .map(|object| objects.borrow_mut().push(object))
    };
    result.map_err(|error| scan_error(error, &tap))?;
    let stats = finish_scan(de.end(), &tap)?;
    Ok((objects.into_inner(), stats))
}

struct Objects<'a, F> {
    object: Object<'a, F>,
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    DeserializeSeed<'de> for Objects<'_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_seq(self)
    }
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    Visitor<'de> for Objects<'_, F>
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
            objects: self.object.objects,
            parent: self.object.parent,
        })? {
            self.object.objects.borrow_mut().push(fields);
        }
        Ok(())
    }
}
struct Object<'a, F> {
    tap: &'a Rc<RefCell<Tap>>,
    frames: &'a [(&'a str, FieldShape)],
    keep: &'a [&'a str],
    item: &'a mut F,
    objects: &'a RefCell<Vec<ObjectFields>>,
    parent: Option<u64>,
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
                let marker = match shape {
                    FieldShape::Value => {
                        map_value(&mut map, self.tap, &mut emit)?;
                        Json::Null
                    }
                    FieldShape::Objects => {
                        map.next_value_seed(Objects {
                            object: Object {
                                tap: self.tap,
                                frames: self.frames,
                                keep: self.keep,
                                item: &mut *self.item,
                                objects: self.objects,
                                parent: Some(start),
                            },
                        })?;
                        Json::Array(Vec::new())
                    }
                    FieldShape::Array | FieldShape::Map => map.next_value_seed(Collection {
                        tap: self.tap,
                        shape: *shape,
                        parent: start,
                        field: &key,
                        item: &mut *self.item,
                    })?,
                };
                fields.insert(key, marker);
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
        let end = self.tap.borrow().position;
        if self.tap.borrow().last != Some((end - 1, b'}')) {
            return Err(A::Error::custom("missing export object end boundary"));
        }
        Ok(ObjectFields {
            start,
            end,
            parent: self.parent,
            fields: Json::Object(fields),
        })
    }
}
struct Collection<'a, F> {
    tap: &'a Rc<RefCell<Tap>>,
    shape: FieldShape,
    parent: u64,
    field: &'a str,
    item: &'a mut F,
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    DeserializeSeed<'de> for Collection<'_, F>
{
    type Value = Json;
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Json, D::Error> {
        de.deserialize_any(self)
    }
}
impl<'de, F: FnMut(u64, &str, Option<&str>, Range<u64>, &RawValue) -> Result<(), String>>
    Visitor<'de> for Collection<'_, F>
{
    type Value = Json;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an export collection or explicit null")
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<Json, E> {
        // Serde has consumed the literal null. Its four bytes are an exact
        // scalar record, distinct from an absent or empty collection.
        let end = self.tap.borrow().position;
        if self.tap.borrow().last != Some((end - 1, b'l')) {
            return Err(E::custom("missing null source boundary"));
        }
        let raw: &RawValue = serde_json::from_str("null").expect("static JSON null");
        (self.item)(self.parent, self.field, None, end - 4..end, raw).map_err(|error| {
            self.tap.borrow_mut().failure = Some(ScanError::Callback(error));
            E::custom("export callback failed")
        })?;
        let mut tap = self.tap.borrow_mut();
        tap.stats.elements += 1;
        tap.stats.peak_element_bytes = tap.stats.peak_element_bytes.max(4);
        Ok(Json::Null)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
        if !matches!(self.shape, FieldShape::Array) {
            return Err(A::Error::custom("expected export field map"));
        }
        let mut emit =
            |range, raw: &RawValue| (self.item)(self.parent, self.field, None, range, raw);
        while seq
            .next_element_seed(ItemSeed {
                tap: self.tap,
                item: &mut emit,
            })?
            .is_some()
        {}
        Ok(Json::Array(Vec::new()))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        if !matches!(self.shape, FieldShape::Map) {
            return Err(A::Error::custom("expected export field array"));
        }
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(A::Error::custom("duplicate export map key"));
            }
            map_value(&mut map, self.tap, &mut |range, raw| {
                (self.item)(self.parent, self.field, Some(&key), range, raw)
            })?;
        }
        Ok(Json::Object(serde_json::Map::new()))
    }
}
