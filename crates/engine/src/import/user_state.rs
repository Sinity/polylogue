//! User requests stream in their recorded order after the immutable export
//! has been validated. The surrounding sessions are skipped, not buffered.
use polylogue_store::primary::RegisterChange;
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};

pub(super) fn replay(
    reader: impl std::io::Read,
    mut apply: impl FnMut(RegisterChange) -> Result<(), String>,
) -> Result<(), serde_json::Error> {
    struct Requests<'a, F>(&'a mut F);
    struct Entries<'a, F>(&'a mut F);
    impl<'de, F: FnMut(RegisterChange) -> Result<(), String>> DeserializeSeed<'de> for Requests<'_, F> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            d.deserialize_map(self)
        }
    }
    impl<'de, F: FnMut(RegisterChange) -> Result<(), String>> Visitor<'de> for Requests<'_, F> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("an export object")
        }
        fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<(), M::Error> {
            while let Some(key) = map.next_key::<String>()? {
                if key == "user_state" {
                    map.next_value_seed(Entries(&mut *self.0))?;
                } else {
                    map.next_value::<IgnoredAny>()?;
                }
            }
            Ok(())
        }
    }
    impl<'de, F: FnMut(RegisterChange) -> Result<(), String>> DeserializeSeed<'de> for Entries<'_, F> {
        type Value = ();
        fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> Result<(), D::Error> {
            d.deserialize_seq(self)
        }
    }
    impl<'de, F: FnMut(RegisterChange) -> Result<(), String>> Visitor<'de> for Entries<'_, F> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("user requests")
        }
        fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<(), S::Error> {
            while let Some(request) = seq.next_element::<RegisterChange>()? {
                (self.0)(request).map_err(serde::de::Error::custom)?;
            }
            Ok(())
        }
    }
    let mut de = serde_json::Deserializer::from_reader(std::io::BufReader::new(reader));
    Requests(&mut apply).deserialize(&mut de)?;
    de.end()
}
