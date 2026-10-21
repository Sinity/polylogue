//! A named message map: stream its values and retain surrounding metadata.
use super::*;
use polylogue_model::Json;
use serde::de::{Error as _, MapAccess};

pub fn scan_map(
    reader: impl Read,
    field: &str,
    mut item: impl FnMut(Range<u64>, &str, &RawValue) -> Result<(), String>,
) -> Result<(Json, ScanStats), ScanError> {
    let tap = Rc::new(RefCell::new(Tap::default()));
    let input = Tapped {
        reader: BufReader::new(reader),
        tap: Rc::clone(&tap),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    let metadata = MapDocument {
        field,
        tap: &tap,
        item: &mut item,
    }
    .deserialize(&mut de)
    .map_err(|e| scan_error(e, &tap))?;
    let stats = finish_scan(de.end(), &tap)?;
    Ok((metadata, stats))
}

struct MapDocument<'a, F> {
    field: &'a str,
    tap: &'a Rc<RefCell<Tap>>,
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
                    tap: self.tap,
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
    tap: &'a Rc<RefCell<Tap>>,
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
            let start = self.tap.borrow().position;
            self.tap.borrow_mut().captured = Some((start, Vec::new()));
            let raw = map.next_value::<Box<RawValue>>()?;
            let (start, captured) = self
                .tap
                .borrow_mut()
                .captured
                .take()
                .expect("capture started");
            if !captured.starts_with(raw.get().as_bytes()) {
                self.tap.borrow_mut().failure = Some(ScanError::Span);
                return Err(A::Error::custom(
                    "Serde map value differs from observed source bytes",
                ));
            }
            {
                let mut tap = self.tap.borrow_mut();
                tap.stats.elements += 1;
                tap.stats.peak_element_bytes = tap
                    .stats
                    .peak_element_bytes
                    .max(captured.len() + raw.get().len());
            }
            drop(captured);
            (self.item)(start..start + raw.get().len() as u64, &key, &raw).map_err(|error| {
                self.tap.borrow_mut().failure = Some(ScanError::Callback(error));
                A::Error::custom("export callback failed")
            })?;
        }
        Ok(())
    }
}
