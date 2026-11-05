//! The kind declaration constrains both accepting schemas and canonical facts.
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
fn structured<T: JsonSchema>(canonical: bool) -> Json {
    if canonical {
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

pub(super) fn value_shape(kind: AssertionKind, canonical: bool) -> Json {
    match kind.spec().value {
        ValueShape::Text => {
            if canonical {
                shape::<TextValue>()
            } else {
                shape::<String>()
            }
        }
        ValueShape::OptionalText => {
            if canonical {
                shape::<OptionalTextValue>()
            } else {
                shape::<Option<String>>()
            }
        }
        ValueShape::Tag => nullable(structured::<Tag>(canonical)),
        ValueShape::Outcome => {
            if canonical {
                shape::<OutcomeValue>()
            } else {
                shape::<Outcome>()
            }
        }
        value @ (ValueShape::Claim | ValueShape::Finding | ValueShape::Question) => {
            let mut detail = structured::<Claim>(canonical);
            if value == ValueShape::Finding {
                detail["properties"]["outcomes"] = serde_json::json!({"type":"null"})
            }
            if value == ValueShape::Question {
                detail["properties"]["check"] = serde_json::json!({"type":"null"})
            }
            either(shape::<String>(), detail)
        }
        ValueShape::Decision => {
            serde_json::json!({"anyOf":[shape::<String>(),structured::<Decision>(canonical),structured::<Assignment>(canonical)]})
        }
        ValueShape::Forecast => structured::<Forecast>(canonical),
        ValueShape::Judgment => {
            nullable(either(shape::<String>(), structured::<Judgment>(canonical)))
        }
        ValueShape::Comparison => structured::<Comparison>(canonical),
        ValueShape::Rubric => structured::<Rubric>(canonical),
        ValueShape::Pass => structured::<Pass>(canonical),
        ValueShape::Annotation => structured::<Annotation>(canonical),
        ValueShape::Method => structured::<Method>(canonical),
        ValueShape::SavedQuery => structured::<SavedQuery>(canonical),
        ValueShape::Signal => structured::<shared_signals::Rule>(canonical),
        ValueShape::Page => structured::<Page>(canonical),
        ValueShape::Slot => structured::<Slot>(canonical),
        ValueShape::Member => nullable(structured::<Member>(canonical)),
        ValueShape::Role => structured::<Role>(canonical),
        ValueShape::Ping => structured::<Ping>(canonical),
        ValueShape::Setting => structured::<Setting>(canonical),
    }
}
fn constrain(schema: &mut Schema, canonical: bool) {
    let conditions:Vec<_>=AssertionKind::ALL.iter().map(|kind|{
        let mut required=Vec::new();
        let mut key=match kind.spec().key {
            KeyShape::None=>serde_json::json!({"type":"null"}),
            KeyShape::Optional=>serde_json::json!({"anyOf":[{"type":"null"},{"type":"string","minLength":1}]}),
            KeyShape::Required=>{required.push("key");serde_json::json!({"type":"string","minLength":1})}
        };
        if kind.spec().value==ValueShape::Judgment {key=serde_json::json!({"type":"string","enum":["accepted","rejected","deferred"]})}
        if !matches!(kind.spec().value,ValueShape::OptionalText|ValueShape::Tag|ValueShape::Judgment|ValueShape::Member) {required.push("value");}
        serde_json::json!({"if":{"properties":{"kind":{"const":kind.as_str()}},"required":["kind"]},"then":{"properties":{"key":key,"value":value_shape(*kind,canonical)},"required":required}})
    }).collect();
    schema.insert("allOf".into(), Json::Array(conditions));
}
pub(crate) fn canonical(schema: &mut Schema) {
    constrain(schema, true)
}
pub(crate) fn live(schema: &mut Schema) {
    constrain(schema, false)
}
