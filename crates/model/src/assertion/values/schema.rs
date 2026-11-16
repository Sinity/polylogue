//! The kind declaration constrains both accepting schemas and stored assertions.
//! Value schemas are inlined before embedding, so their references cannot
//! accidentally point at another schema document's definitions.
use super::*;
use crate::assertion::{AssertionKind, KeyShape, ValueShape};
use schemars::{JsonSchema, Schema};

fn shape<T: JsonSchema>() -> Json {
    let mut generator = schemars::generate::SchemaSettings::default()
        .with(|settings| settings.inline_subschemas = true)
        .into_generator();
    let mut schema = generator.root_schema_for::<T>();
    schema.remove("$schema");
    schema.remove("title");
    schema.to_value()
}
fn structured<T: JsonSchema>(stored: bool) -> Json {
    if stored {
        shape::<Structured<T>>()
    } else {
        shape::<T>()
    }
}
fn nullable(schema: Json) -> Json {
    serde_json::json!({"anyOf":[{"type":"null"},schema]})
}
fn either(left: Json, right: Json) -> Json {
    serde_json::json!({"anyOf":[left,right]})
}

pub(super) fn value_shape(kind: AssertionKind, stored: bool) -> Json {
    match kind.spec().value {
        ValueShape::Text => {
            if stored {
                shape::<TextValue>()
            } else {
                shape::<String>()
            }
        }
        ValueShape::OptionalText => {
            if stored {
                shape::<OptionalTextValue>()
            } else {
                shape::<Option<String>>()
            }
        }
        ValueShape::Tag => nullable(structured::<Tag>(stored)),
        ValueShape::Outcome => {
            if stored {
                shape::<OutcomeValue>()
            } else {
                shape::<Outcome>()
            }
        }
        value @ (ValueShape::Claim | ValueShape::Finding | ValueShape::Question) => {
            let mut detail = structured::<Claim>(stored);
            if value == ValueShape::Finding {
                detail["properties"]["outcomes"] = serde_json::json!({"type":"null"})
            }
            if value == ValueShape::Question {
                detail["properties"]["check"] = serde_json::json!({"type":"null"})
            }
            either(shape::<String>(), detail)
        }
        ValueShape::Decision => {
            serde_json::json!({"anyOf":[shape::<String>(),structured::<Decision>(stored),structured::<Assignment>(stored)]})
        }
        ValueShape::Forecast => structured::<Forecast>(stored),
        ValueShape::Judgment => nullable(either(shape::<String>(), structured::<Judgment>(stored))),
        ValueShape::Comparison => structured::<Comparison>(stored),
        ValueShape::Rubric => structured::<Rubric>(stored),
        ValueShape::Pass => structured::<Pass>(stored),
        ValueShape::Annotation => structured::<Annotation>(stored),
        ValueShape::Method => structured::<Method>(stored),
        ValueShape::SavedQuery => structured::<SavedQuery>(stored),
        ValueShape::Signal => structured::<shared_signals::Rule>(stored),
        ValueShape::Page => structured::<Page>(stored),
        ValueShape::Slot => structured::<Slot>(stored),
        ValueShape::Member => nullable(structured::<Member>(stored)),
        ValueShape::Role => structured::<Role>(stored),
        ValueShape::Ping => structured::<Ping>(stored),
        ValueShape::Setting => structured::<Setting>(stored),
    }
}
fn constrain(schema: &mut Schema, stored: bool) {
    let conditions:Vec<_>=AssertionKind::ALL.iter().map(|kind|{
        let mut required=Vec::new();
        let mut key=match kind.spec().key {
            KeyShape::None=>serde_json::json!({"type":"null"}),
            KeyShape::Optional=>serde_json::json!({"anyOf":[{"type":"null"},{"type":"string","minLength":1}]}),
            KeyShape::Required=>{required.push("key");serde_json::json!({"type":"string","minLength":1})}
        };
        if kind.spec().value==ValueShape::Judgment {key=serde_json::json!({"type":"string","enum":["accepted","rejected","deferred"]})}
        if !matches!(kind.spec().value,ValueShape::OptionalText|ValueShape::Tag|ValueShape::Judgment|ValueShape::Member) {required.push("value");}
        serde_json::json!({"if":{"properties":{"kind":{"const":kind.as_str()}},"required":["kind"]},"then":{"properties":{"key":key,"value":value_shape(*kind,stored)},"required":required}})
    }).collect();
    schema.insert("allOf".into(), Json::Array(conditions));
}
pub(crate) fn stored(schema: &mut Schema) {
    constrain(schema, true)
}
pub(crate) fn live(schema: &mut Schema) {
    constrain(schema, false)
}
