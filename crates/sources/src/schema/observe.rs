use super::{Node, field_name};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Read,
};

pub(super) fn read(reader: impl Read, names: &BTreeSet<String>) -> Result<Node, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_reader(reader);
    de.disable_recursion_limit();
    let mut node = Node::default();
    Seed {
        node: &mut node,
        names,
    }
    .deserialize(&mut de)?;
    de.end()?;
    node.present = 1;
    Ok(node)
}

struct Seed<'a> {
    node: &'a mut Node,
    names: &'a BTreeSet<String>,
}
impl<'de> DeserializeSeed<'de> for Seed<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, de: D) -> Result<(), D::Error> {
        de.deserialize_any(self)
    }
}
impl<'de> Visitor<'de> for Seed<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("JSON")
    }
    fn visit_bool<E: serde::de::Error>(self, _: bool) -> Result<(), E> {
        self.node.kind("boolean");
        Ok(())
    }
    fn visit_i64<E: serde::de::Error>(self, _: i64) -> Result<(), E> {
        self.node.kind("integer");
        Ok(())
    }
    fn visit_u64<E: serde::de::Error>(self, _: u64) -> Result<(), E> {
        self.node.kind("integer");
        Ok(())
    }
    fn visit_f64<E: serde::de::Error>(self, _: f64) -> Result<(), E> {
        self.node.kind("number");
        Ok(())
    }
    fn visit_str<E: serde::de::Error>(self, _: &str) -> Result<(), E> {
        self.node.kind("string");
        Ok(())
    }
    fn visit_unit<E: serde::de::Error>(self) -> Result<(), E> {
        self.node.kind("null");
        Ok(())
    }
    fn visit_none<E: serde::de::Error>(self) -> Result<(), E> {
        self.visit_unit()
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut array: A) -> Result<(), A::Error> {
        self.node.kind("array");
        let mut child = Node::default();
        while array
            .next_element_seed(Seed {
                node: &mut child,
                names: self.names,
            })?
            .is_some()
        {}
        if child.occurrences > 0 {
            child.present = 1;
            self.node
                .children
                .entry("[]".into())
                .or_default()
                .merge(child);
        }
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<(), A::Error> {
        let first = object.next_key::<Key>()?;
        if first.as_ref().is_some_and(|key| key.number) {
            let number = object.next_value::<String>()?;
            self.node.kind(if number.contains(['.', 'e', 'E']) {
                "number"
            } else {
                "integer"
            });
            return Ok(());
        }
        self.node.kind("object");
        let mut next = first;
        let mut children = BTreeMap::<String, Node>::new();
        while let Some(key) = next {
            let name = if self.names.contains(&key.name) {
                field_name(&key.name)
            } else {
                "*".into()
            };
            object.next_value_seed(Seed {
                node: children.entry(name).or_default(),
                names: self.names,
            })?;
            next = object.next_key::<Key>()?;
        }
        for (key, mut child) in children {
            child.present = 1;
            self.node.children.entry(key).or_default().merge(child);
        }
        Ok(())
    }
}

struct Key {
    name: String,
    number: bool,
}
impl<'de> serde::Deserialize<'de> for Key {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        struct KeyVisitor;
        impl<'de> Visitor<'de> for KeyVisitor {
            type Value = Key;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object key")
            }
            fn visit_str<E: serde::de::Error>(self, key: &str) -> Result<Key, E> {
                Ok(Key {
                    name: key.into(),
                    number: false,
                })
            }
            fn visit_borrowed_str<E: serde::de::Error>(self, key: &'de str) -> Result<Key, E> {
                // from_reader cannot borrow source keys. Serde JSON uses a borrowed
                // marker to present arbitrary-precision numbers through MapAccess.
                Ok(Key {
                    name: key.into(),
                    number: key == "$serde_json::private::Number",
                })
            }
        }
        de.deserialize_string(KeyVisitor)
    }
}
