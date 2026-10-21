//! Repeated arrays along an OTLP-style field path; wrapper values are discarded.
use super::*;
use serde::de::{Error as _, IgnoredAny, MapAccess};

pub fn scan_path(
    reader: impl Read,
    path: &[&str],
    mut item: impl FnMut(Range<u64>, &RawValue) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let tap = Rc::new(RefCell::new(Tap::default()));
    let input = Tapped {
        reader: BufReader::new(reader),
        tap: Rc::clone(&tap),
    };
    let mut de = serde_json::Deserializer::from_reader(input);
    let result = (|| {
        Node {
            path,
            arrays: false,
            tap: &tap,
            item: &mut item,
        }
        .deserialize(&mut de)?;
        de.end()
    })();
    finish_scan(result, &tap)
}

struct Node<'a, F> {
    path: &'a [&'a str],
    arrays: bool,
    tap: &'a Rc<RefCell<Tap>>,
    item: &'a mut F,
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> DeserializeSeed<'de>
    for Node<'_, F>
{
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_any(self)
    }
}
impl<'de, F: FnMut(Range<u64>, &RawValue) -> Result<(), String>> Visitor<'de> for Node<'_, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an export container")
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
        let mut selected = false;
        while let Some(key) = map.next_key::<String>()? {
            if self.path.first() == Some(&key.as_str()) {
                if selected {
                    return Err(A::Error::custom("duplicate streamed container field"));
                }
                selected = true;
                map.next_value_seed(Node {
                    path: &self.path[1..],
                    arrays: true,
                    tap: self.tap,
                    item: &mut *self.item,
                })?;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
        if self.arrays && self.path.is_empty() {
            return ArraySeed {
                tap: self.tap,
                item: self.item,
            }
            .visit_seq(seq);
        }
        if self.arrays {
            while seq
                .next_element_seed(Node {
                    path: self.path,
                    arrays: false,
                    tap: self.tap,
                    item: &mut *self.item,
                })?
                .is_some()
            {}
        } else {
            while seq.next_element::<IgnoredAny>()?.is_some() {}
        }
        Ok(())
    }
    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<(), E> {
        Ok(())
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<(), E> {
        Ok(())
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<(), E> {
        Ok(())
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<(), E> {
        Ok(())
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<(), E> {
        Ok(())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<(), E> {
        Ok(())
    }
}
