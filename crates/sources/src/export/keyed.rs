//! A named message map: stream its values and keep surrounding metadata.
use super::*;
use polylogue_model::Json;
use serde::de::{Error as _, MapAccess};

pub fn scan_map(
    reader: impl Read,
    field: &str,
    mut item: impl FnMut(Range<u64>, &str, &RawValue) -> Result<(), String>,
) -> Result<(Json, ScanStats), ScanError> {
    let scan_state = Rc::new(RefCell::new(ScanState::default()));
    let input = PositionReader {
        reader: BufReader::new(reader),
        scan_state: Rc::clone(&scan_state),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    let metadata = MapDocument {
        field,
        scan_state: &scan_state,
        item: &mut item,
    }
    .deserialize(&mut de)
    .map_err(|e| scan_error(e, &scan_state))?;
    let stats = finish_scan(de.end(), &scan_state)?;
    Ok((metadata, stats))
}

struct MapDocument<'a, F> {
    field: &'a str,
    scan_state: &'a Rc<RefCell<ScanState>>,
    item: &'a mut F,
}
impl<'de, F: FnMut(Range<u64>, &str, &RawValue) -> Result<(), String>> DeserializeSeed<'de>
    for MapDocument<'_, F>
{
    type Value = Json;
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<Json, D::Error> {
        de.deserialize_map(self)
    }
}
impl<'de, F: FnMut(Range<u64>, &str, &RawValue) -> Result<(), String>> Visitor<'de>
    for MapDocument<'_, F>
{
    type Value = Json;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an export object")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
        let mut metadata = serde_json::Map::new();
        let mut selected = false;
        while let Some(key) = map.next_key::<String>()? {
            let value = if key == self.field {
                if selected {
                    return Err(A::Error::custom("duplicate streamed-map field"));
                }
                selected = true;
                map.next_value_seed(Entries {
                    scan_state: self.scan_state,
                    item: self.item,
                })?;
                Json::Object(Default::default())
            } else {
                map.next_value::<Json>()?
            };
            metadata.insert(key, value);
        }
        Ok(Json::Object(metadata))
    }
}

struct Entries<'a, F> {
    scan_state: &'a Rc<RefCell<ScanState>>,
    item: &'a mut F,
}
impl<'de, F: FnMut(Range<u64>, &str, &RawValue) -> Result<(), String>> DeserializeSeed<'de>
    for Entries<'_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_map(self)
    }
}
impl<'de, F: FnMut(Range<u64>, &str, &RawValue) -> Result<(), String>> Visitor<'de>
    for Entries<'_, F>
{
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a map of export records")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        while let Some(key) = map.next_key::<String>()? {
            map_value(&mut map, self.scan_state, &mut |range, raw| {
                (self.item)(range, &key, raw)
            })?;
        }
        Ok(())
    }
}
