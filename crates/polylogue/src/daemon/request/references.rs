//! Observe declared references while Serde still has their model types.
use super::References;
use serde::Serialize;
use serde::ser::{
    self, SerializeMap, SerializeSeq, SerializeStruct, SerializeStructVariant, SerializeTuple,
    SerializeTupleStruct, SerializeTupleVariant, Serializer,
};
use serde_json::{Map, Value};

type Result<T> = std::result::Result<T, serde_json::Error>;

pub(super) fn value(
    references: &mut References,
    path: &str,
    value: &impl Serialize,
) -> Result<Value> {
    value.serialize(Collect {
        references,
        path: path.into(),
        observe: false,
    })
}

pub(super) fn observe(references: &mut References, path: &str, value: &impl Serialize) {
    value
        .serialize(Collect {
            references,
            path: path.into(),
            observe: true,
        })
        .expect("model serializes");
}

struct Collect<'a> {
    references: &'a mut References,
    path: String,
    observe: bool,
}
impl Collect<'_> {
    fn child(&mut self, name: impl std::fmt::Display) -> Collect<'_> {
        Collect {
            references: self.references,
            observe: self.observe,
            path: format!(
                "{}/{}",
                self.path,
                name.to_string().replace('~', "~0").replace('/', "~1")
            ),
        }
    }
}

macro_rules! scalar {
    ($($method:ident($kind:ty)),* $(,)?) => {$ (
        fn $method(self, value: $kind) -> Result<Value> { if self.observe {Ok(Value::Null)} else {serde_json::to_value(value)} }
    )*};
}

impl<'a> Serializer for Collect<'a> {
    type Ok = Value;
    type Error = serde_json::Error;
    type SerializeSeq = Sequence<'a>;
    type SerializeTuple = Sequence<'a>;
    type SerializeTupleStruct = Sequence<'a>;
    type SerializeTupleVariant = Sequence<'a>;
    type SerializeMap = Object<'a>;
    type SerializeStruct = Object<'a>;
    type SerializeStructVariant = Object<'a>;
    scalar! { serialize_bool(bool), serialize_i8(i8), serialize_i16(i16), serialize_i32(i32), serialize_i64(i64), serialize_i128(i128), serialize_u8(u8), serialize_u16(u16), serialize_u32(u32), serialize_u64(u64), serialize_u128(u128), serialize_f32(f32), serialize_f64(f64), serialize_char(char), serialize_str(&str), serialize_bytes(&[u8]) }
    fn serialize_none(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Value> {
        value.serialize(self)
    }
    fn serialize_unit(self) -> Result<Value> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _: &'static str) -> Result<Value> {
        self.serialize_unit()
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        variant: &'static str,
    ) -> Result<Value> {
        self.serialize_str(variant)
    }
    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Value> {
        if name == "polylogue_reference" {
            let value = serde_json::to_value(value)?;
            self.references.0.push((self.path, value.clone()));
            Ok(value)
        } else {
            value.serialize(self)
        }
    }
    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        mut self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value> {
        let value = value.serialize(self.child(variant))?;
        Ok(Value::Object(Map::from_iter([(variant.into(), value)])))
    }
    fn serialize_seq(self, _: Option<usize>) -> Result<Sequence<'a>> {
        Ok(Sequence {
            collect: self,
            values: Vec::new(),
            variant: None,
        })
    }
    fn serialize_tuple(self, len: usize) -> Result<Sequence<'a>> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> Result<Sequence<'a>> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_variant(
        mut self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Sequence<'a>> {
        self.path = format!(
            "{}/{}",
            self.path,
            variant.replace('~', "~0").replace('/', "~1")
        );
        Ok(Sequence {
            collect: self,
            values: Vec::new(),
            variant: Some(variant),
        })
    }
    fn serialize_map(self, _: Option<usize>) -> Result<Object<'a>> {
        Ok(Object {
            collect: self,
            values: Map::new(),
            key: None,
            json: false,
            variant: None,
        })
    }
    fn serialize_struct(self, name: &'static str, len: usize) -> Result<Object<'a>> {
        let mut object = self.serialize_map(Some(len))?;
        object.json = matches!(
            name,
            "$serde_json::private::Number" | "$serde_json::private::RawValue"
        );
        Ok(object)
    }
    fn serialize_struct_variant(
        mut self,
        _: &'static str,
        _: u32,
        variant: &'static str,
        _: usize,
    ) -> Result<Object<'a>> {
        self.path = format!(
            "{}/{}",
            self.path,
            variant.replace('~', "~0").replace('/', "~1")
        );
        Ok(Object {
            collect: self,
            values: Map::new(),
            key: None,
            json: false,
            variant: Some(variant),
        })
    }
}

struct Sequence<'a> {
    collect: Collect<'a>,
    values: Vec<Value>,
    variant: Option<&'static str>,
}
impl Sequence<'_> {
    fn push<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        let value = value.serialize(self.collect.child(self.values.len()))?;
        self.values.push(value);
        Ok(())
    }
    fn finish(self) -> Value {
        if self.collect.observe {
            return Value::Null;
        }
        let value = Value::Array(self.values);
        if let Some(variant) = self.variant {
            Value::Object(Map::from_iter([(variant.into(), value)]))
        } else {
            value
        }
    }
}
macro_rules! sequence {
    ($trait:ident,$element:ident) => {
        impl $trait for Sequence<'_> {
            type Ok = Value;
            type Error = serde_json::Error;
            fn $element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
                self.push(value)
            }
            fn end(self) -> Result<Value> {
                Ok(self.finish())
            }
        }
    };
}
sequence!(SerializeSeq, serialize_element);
sequence!(SerializeTuple, serialize_element);
sequence!(SerializeTupleStruct, serialize_field);
sequence!(SerializeTupleVariant, serialize_field);

struct Object<'a> {
    collect: Collect<'a>,
    values: Map<String, Value>,
    key: Option<String>,
    json: bool,
    variant: Option<&'static str>,
}
impl Object<'_> {
    fn field<T: ?Sized + Serialize>(&mut self, key: &str, value: &T) -> Result<()> {
        let value = value.serialize(self.collect.child(key))?;
        self.values.insert(key.into(), value);
        Ok(())
    }
    fn finish(self) -> Result<Value> {
        if self.collect.observe {
            return Ok(Value::Null);
        }
        // Serde JSON represents exact numbers and raw JSON through named structs.
        if self.json {
            let text = self
                .values
                .values()
                .next()
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    <serde_json::Error as ser::Error>::custom("JSON representation has no text")
                })?;
            return serde_json::from_str(text);
        }
        let value = Value::Object(self.values);
        Ok(if let Some(variant) = self.variant {
            Value::Object(Map::from_iter([(variant.into(), value)]))
        } else {
            value
        })
    }
}
impl SerializeMap for Object<'_> {
    type Ok = Value;
    type Error = serde_json::Error;
    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<()> {
        self.key = Some(match serde_json::to_value(key)? {
            Value::String(key) => key,
            Value::Number(key) => key.to_string(),
            Value::Bool(key) => key.to_string(),
            _ => return Err(ser::Error::custom("JSON keys must be strings")),
        });
        Ok(())
    }
    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<()> {
        let key = self
            .key
            .take()
            .ok_or_else(|| <serde_json::Error as ser::Error>::custom("value has no key"))?;
        self.field(&key, value)
    }
    fn end(self) -> Result<Value> {
        self.finish()
    }
}
macro_rules! object {
    ($trait:ident) => {
        impl $trait for Object<'_> {
            type Ok = Value;
            type Error = serde_json::Error;
            fn serialize_field<T: ?Sized + Serialize>(
                &mut self,
                key: &'static str,
                value: &T,
            ) -> Result<()> {
                self.field(key, value)
            }
            fn end(self) -> Result<Value> {
                self.finish()
            }
        }
    };
}
object!(SerializeStruct);
object!(SerializeStructVariant);

#[cfg(test)]
mod tests;
