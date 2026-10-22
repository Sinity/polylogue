use super::*;
use serde::de::{DeserializeSeed, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor};

struct Load<'a> {
    cache: &'a mut Refused,
    error: Option<StoreError>,
}

impl Load<'_> {
    fn insert<E: serde::de::Error>(&mut self, table: &str, id: &str) -> std::result::Result<(), E> {
        match self.cache.insert(table, id) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.error = Some(error);
                Err(E::custom("exclusion lookup construction failed"))
            }
        }
    }
}

pub(super) fn policy(cache: &mut Refused, json: &str) -> Result<bool> {
    let mut load = Load { cache, error: None };
    let mut de = serde_json::Deserializer::from_str(json);
    let result = Policy(&mut load).deserialize(&mut de);
    if let Some(error) = load.error {
        return Err(error);
    }
    let known = result?;
    de.end()?;
    Ok(known)
}

pub(super) fn messages(cache: &mut Refused, json: &str) -> Result<()> {
    let mut load = Load { cache, error: None };
    let mut de = serde_json::Deserializer::from_str(json);
    let result = Ids {
        load: &mut load,
        sessions: false,
    }
    .deserialize(&mut de);
    if let Some(error) = load.error {
        return Err(error);
    }
    result?;
    de.end()?;
    Ok(())
}

struct Ids<'a, 'b> {
    load: &'a mut Load<'b>,
    sessions: bool,
}
impl<'de> DeserializeSeed<'de> for Ids<'_, '_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> std::result::Result<(), D::Error> {
        de.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for Ids<'_, '_> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an array of exclusion IDs")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        while let Some(id) = seq.next_element::<String>()? {
            let table = if self.sessions {
                SessionId::parse(&id).map_err(A::Error::custom)?;
                "sessions"
            } else {
                "messages"
            };
            self.load.insert(table, &id)?;
        }
        Ok(())
    }
}

struct Policy<'a, 'b>(&'a mut Load<'b>);
impl<'de> DeserializeSeed<'de> for Policy<'_, '_> {
    type Value = bool;
    fn deserialize<D: serde::Deserializer<'de>>(
        self,
        de: D,
    ) -> std::result::Result<bool, D::Error> {
        de.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Policy<'_, '_> {
    type Value = bool;
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("an exclusion policy object or its legacy empty array")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<bool, A::Error> {
        if seq.next_element::<IgnoredAny>()?.is_some() {
            return Err(A::Error::custom("legacy exclusion array must be empty"));
        }
        Ok(false)
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<bool, A::Error> {
        let mut known = false;
        while let Some(key) = map.next_key::<String>()? {
            if key == "sessions" {
                if known {
                    return Err(A::Error::duplicate_field("sessions"));
                }
                map.next_value_seed(Ids {
                    load: &mut *self.0,
                    sessions: true,
                })?;
                known = true;
            } else {
                map.next_value::<IgnoredAny>()?;
            }
        }
        Ok(known)
    }
}
