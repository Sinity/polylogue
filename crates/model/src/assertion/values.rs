//! Closed value shapes. Dynamic annotation and setting values still require
//! their resolved rubric or catalog at the accepting snapshot.
use crate::{BlobHash, Json, Origin, Reference, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
mod number;
pub(super) mod schema;

macro_rules! record {
    ($name:ident {$($field:ident : $ty:ty),* $(,)?}) => {
        #[derive(Debug,Clone,PartialEq,Serialize,Deserialize,schemars::JsonSchema)]
        #[serde(deny_unknown_fields)]
        pub struct $name {$(pub $field: $ty),*}
    };
}
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum ClaimValue {
    Text(String),
    Detail(Structured<Claim>),
}
impl ClaimValue {
    fn content(&self) -> (&str, Option<&Check>, &[CheckOutcome]) {
        match self {
            Self::Text(text) => (text, None, &[]),
            Self::Detail(value) => (
                &value.fields.text,
                value.fields.check.as_ref(),
                value.fields.outcomes.as_deref().unwrap_or(&[]),
            ),
        }
    }
}
impl PartialEq for ClaimValue {
    fn eq(&self, other: &Self) -> bool {
        self.content() == other.content()
    }
}

record!(Setting {machine:Option<String>,value:crate::settings::Value});
/// Historical fields are a declaration carried beside the canonical value.
/// They never participate in its schema checks or semantic equality.
#[derive(Debug, Clone)]
pub struct Structured<T> {
    pub fields: T,
    pub predecessor: Option<Json>,
}
impl<T> Structured<T> {
    pub fn plain(fields: T) -> Self {
        Self {
            fields,
            predecessor: None,
        }
    }
}
impl<T: PartialEq> PartialEq for Structured<T> {
    fn eq(&self, other: &Self) -> bool {
        self.fields == other.fields
    }
}
impl<T: Eq> Eq for Structured<T> {}
impl<T: Serialize> Serialize for Structured<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut value = serde_json::to_value(&self.fields).map_err(serde::ser::Error::custom)?;
        let object = value.as_object_mut().ok_or_else(|| {
            serde::ser::Error::custom("structured assertion fields must be an object")
        })?;
        if let Some(predecessor) = &self.predecessor {
            object.insert("predecessor".into(), predecessor.clone());
        }
        value.serialize(serializer)
    }
}
impl<'de, T: serde::de::DeserializeOwned> Deserialize<'de> for Structured<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let mut value = Json::deserialize(deserializer)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| serde::de::Error::custom("expected structured assertion fields"))?;
        let predecessor = object.remove("predecessor");
        let fields = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
        Ok(Self {
            fields,
            predecessor,
        })
    }
}
impl<T: schemars::JsonSchema> schemars::JsonSchema for Structured<T> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("Structured_{}", T::schema_name()).into()
    }
    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let mut schema = T::json_schema(generator);
        if let Some(object) = schema.as_object_mut() {
            let properties = object
                .entry("properties")
                .or_insert_with(|| serde_json::json!({}));
            if let Some(properties) = properties.as_object_mut() {
                properties.insert("predecessor".into(), Json::Bool(true));
            }
        }
        schema
    }
}
record!(WithPredecessor {
    text: String,
    predecessor: Json
});
record!(OptionalWithPredecessor {text:Option<String>,predecessor:Json});
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum TextValue {
    Plain(String),
    WithPredecessor(WithPredecessor),
}
impl TextValue {
    pub fn text(&self) -> &str {
        match self {
            Self::Plain(text) => text,
            Self::WithPredecessor(value) => &value.text,
        }
    }
}
impl PartialEq for TextValue {
    fn eq(&self, other: &Self) -> bool {
        self.text() == other.text()
    }
}
impl Eq for TextValue {}
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum OptionalTextValue {
    Plain(Option<String>),
    WithPredecessor(OptionalWithPredecessor),
}
impl OptionalTextValue {
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Plain(text) => text.as_deref(),
            Self::WithPredecessor(value) => value.text.as_deref(),
        }
    }
}
impl PartialEq for OptionalTextValue {
    fn eq(&self, other: &Self) -> bool {
        self.text() == other.text()
    }
}
impl Eq for OptionalTextValue {}

pub fn text(value: &Json) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("text").and_then(Json::as_str))
}
record!(Annotation {value:Json,confidence:Option<Probability>,rationale:Option<String>});
record!(Member {});

record!(Tag {value:Option<Json>,confidence:Option<Probability>});
record!(Check {
    query: String,
    expect: String
});
record!(CheckOutcome {
    observation: String,
    implies: String
});
record!(Claim {text:String,check:Option<Check>,outcomes:Option<Vec<CheckOutcome>>});
record!(Decision {text:String,alternatives:Option<Vec<String>>,constraints:Option<Vec<String>>,reverse_if:Option<String>});
record!(Forecast {
    probability: Probability,
    horizon: Timestamp,
    resolves_by: String
});
record!(Blind {order:u64,hidden:Vec<String>,seed:Option<String>});
record!(Judgment {reason:Option<String>,blind:Option<Blind>});
record!(Comparison {verdict:Verdict,rubric:Option<Reference>});
record!(Rubric {instructions:String,unit_query:String,fields:Vec<Field>,abstain:bool,confidence:bool,rationale:Option<bool>});
record!(Field {name:String,value_type:FieldType,enum_values:Option<Vec<Json>>,minimum:Option<serde_json::Number>,maximum:Option<serde_json::Number>,required:Option<bool>,description:Option<String>});
record!(Pass {rubric:Option<Reference>,population:Option<Reference>,counts:BTreeMap<String,u64>});
record!(Condition {
    name: String,
    check: Check
});
record!(Method {inputs:Vec<String>,conditions:Vec<Condition>,steps:Vec<String>,expected:String,stop:Vec<String>});
record!(SavedQuery {text:String,hash:BlobHash,denominator:Option<String>});
record!(Page {title:String,layout:Layout,template:Option<String>});
record!(Slot {section:String,pos:serde_json::Number,mode:SlotMode,heading:Option<String>,query:Option<String>,caption:Option<String>,x:Option<serde_json::Number>,y:Option<serde_json::Number>,w:Option<serde_json::Number>,h:Option<serde_json::Number>});
record!(Role {
    instructions: String,
    harness: Origin
});
record!(ReplyTo {
    page: Reference,
    section: String
});
record!(Ping {text:String,subject:Option<Reference>,reply_to:Option<ReplyTo>});

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FieldType {
    String,
    Integer,
    Number,
    Boolean,
    Enum,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    Linear,
    Board,
    Wiki,
    Table,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SlotMode {
    Follow,
    Fixed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Solved,
    Partial,
    Abandoned,
}

record!(OutcomeWithPredecessor {
    text: Outcome,
    predecessor: Json
});
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum OutcomeValue {
    Plain(Outcome),
    WithPredecessor(OutcomeWithPredecessor),
}
impl OutcomeValue {
    pub fn outcome(&self) -> Outcome {
        match self {
            Self::Plain(value) => *value,
            Self::WithPredecessor(value) => value.text,
        }
    }
}
impl PartialEq for OutcomeValue {
    fn eq(&self, other: &Self) -> bool {
        self.outcome() == other.outcome()
    }
}
impl Eq for OutcomeValue {}

/// A probability preserves its original JSON number rather than rounding at
/// the zero/one boundary. Huge exponents are compared without allocating their
/// expanded decimal representation.
#[derive(Debug, Clone, Serialize)]
pub struct Probability(serde_json::Number);
impl PartialEq for Probability {
    fn eq(&self, other: &Self) -> bool {
        number::compare(&self.0, &other.0).is_eq()
    }
}
impl Eq for Probability {}

impl Probability {
    pub fn number(&self) -> &serde_json::Number {
        &self.0
    }
}
impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let number = serde_json::Number::deserialize(deserializer)?;
        if number::compare(&number, &0.into()).is_lt()
            || number::compare(&number, &1.into()).is_gt()
        {
            return Err(serde::de::Error::custom(
                "expected a probability from zero through one",
            ));
        }
        Ok(Self(number))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    Prefer(Reference),
    Ranking(Vec<Reference>),
    Tie,
    Incomparable,
    Abstain,
    InsufficientEvidence,
}
record!(Ranking {ranking:Vec<Reference>});
#[derive(Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum VerdictWire {
    Name(String),
    Ranking(Ranking),
}
impl Serialize for Verdict {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Prefer(reference) => serializer.serialize_str(&format!("prefer:{reference}")),
            Self::Ranking(ranking) => {
                use serde::ser::SerializeMap;
                let mut map = serializer.serialize_map(Some(1))?;
                map.serialize_entry("ranking", ranking)?;
                map.end()
            }
            Self::Tie => serializer.serialize_str("tie"),
            Self::Incomparable => serializer.serialize_str("incomparable"),
            Self::Abstain => serializer.serialize_str("abstain"),
            Self::InsufficientEvidence => serializer.serialize_str("insufficient_evidence"),
        }
    }
}
impl<'de> Deserialize<'de> for Verdict {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        match VerdictWire::deserialize(deserializer)? {
            VerdictWire::Ranking(Ranking { ranking }) => Ok(Self::Ranking(ranking)),
            VerdictWire::Name(text) => Ok(match text.as_str() {
                "tie" => Self::Tie,
                "incomparable" => Self::Incomparable,
                "abstain" => Self::Abstain,
                "insufficient_evidence" => Self::InsufficientEvidence,
                _ => Self::Prefer(
                    Reference::parse(text.strip_prefix("prefer:").ok_or_else(|| {
                        serde::de::Error::custom("expected a comparison verdict")
                    })?)
                    .map_err(serde::de::Error::custom)?,
                ),
            }),
        }
    }
}
impl schemars::JsonSchema for Verdict {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Verdict".into()
    }
    fn json_schema(generator: &mut schemars::SchemaGenerator) -> schemars::Schema {
        let ranking = generator.subschema_for::<Ranking>();
        schemars::json_schema!({"oneOf":[{"type":"string","enum":["tie","incomparable","abstain","insufficient_evidence"]},{"type":"string","pattern":"^prefer:.+$"},ranking]})
    }
}
impl schemars::JsonSchema for Probability {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Probability".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"number","minimum":0,"maximum":1})
    }
}

impl Field {
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("rubric field name cannot be empty".into());
        }
        let numeric = matches!(self.value_type, FieldType::Integer | FieldType::Number);
        if !numeric && (self.minimum.is_some() || self.maximum.is_some()) {
            return Err(format!(
                "field {} has a range on a nonnumeric type",
                self.name
            ));
        }
        if let (Some(minimum), Some(maximum)) = (&self.minimum, &self.maximum)
            && number::compare(minimum, maximum).is_gt()
        {
            return Err(format!("field {} has minimum above maximum", self.name));
        }
        match (self.value_type, &self.enum_values) {
            (FieldType::Enum, Some(values)) if !values.is_empty() => {
                for (i, value) in values.iter().enumerate() {
                    if values[..i].iter().any(|prior| equal(value, prior)) {
                        return Err(format!("field {} has duplicate enum values", self.name));
                    }
                }
            }
            (FieldType::Enum, _) => return Err(format!("field {} needs enum values", self.name)),
            (_, Some(_)) => {
                return Err(format!(
                    "field {} has enum values on a non-enum type",
                    self.name
                ));
            }
            _ => {}
        }
        Ok(())
    }
    pub fn allows(&self, value: &Json) -> bool {
        let valid = match self.value_type {
            FieldType::String => value.is_string(),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Integer => value.as_number().is_some_and(number::integer),
            FieldType::Number => value.is_number(),
            FieldType::Enum => self
                .enum_values
                .as_ref()
                .is_some_and(|values| values.iter().any(|allowed| equal(allowed, value))),
        };
        valid
            && value.as_number().is_none_or(|value| {
                self.minimum
                    .as_ref()
                    .is_none_or(|minimum| !number::compare(value, minimum).is_lt())
                    && self
                        .maximum
                        .as_ref()
                        .is_none_or(|maximum| !number::compare(value, maximum).is_gt())
            })
    }
}
/// One text per value under [`equal`]: object keys sorted and numbers in one
/// spelling, so two values give the same text exactly when they are equal.
pub fn canonical(value: &Json) -> String {
    fn write(value: &Json, out: &mut String) {
        match value {
            Json::Number(number) => out.push_str(&number::canonical(number)),
            Json::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write(item, out);
                }
                out.push(']');
            }
            Json::Object(map) => {
                let mut entries: Vec<_> = map.iter().collect();
                entries.sort_by(|left, right| left.0.cmp(right.0));
                out.push('{');
                for (index, (key, item)) in entries.into_iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    out.push_str(&Json::String(key.clone()).to_string());
                    out.push(':');
                    write(item, out);
                }
                out.push('}');
            }
            scalar => out.push_str(&scalar.to_string()),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}
pub(super) fn equal(left: &Json, right: &Json) -> bool {
    let mut pending = vec![(left, right)];
    while let Some((left, right)) = pending.pop() {
        match (left, right) {
            (Json::Number(left), Json::Number(right)) if number::compare(left, right).is_eq() => {}
            (Json::Array(left), Json::Array(right)) if left.len() == right.len() => {
                pending.extend(left.iter().zip(right))
            }
            (Json::Object(left), Json::Object(right)) if left.len() == right.len() => {
                for (key, value) in left {
                    let Some(other) = right.get(key) else {
                        return false;
                    };
                    pending.push((value, other));
                }
            }
            (Json::Null, Json::Null) => {}
            (Json::Bool(left), Json::Bool(right)) if left == right => {}
            (Json::String(left), Json::String(right)) if left == right => {}
            _ => return false,
        }
    }
    true
}

impl Rubric {
    pub fn validate(&self) -> Result<(), String> {
        if self.instructions.is_empty() || self.unit_query.is_empty() {
            return Err("rubric instructions and unit_query cannot be empty".into());
        }
        let mut names = BTreeSet::new();
        for field in &self.fields {
            field.validate()?;
            if !names.insert(&field.name) {
                return Err(format!("duplicate rubric field {}", field.name));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

impl Setting {
    /// Resolves authored destination once, preserving the accepting machine
    /// separately on Assertion. Frozen records retain this resolved payload.
    pub fn resolve(
        mut self,
        key: &str,
        accepting_machine: &str,
        owner: crate::settings::Owner,
    ) -> Result<Self, crate::settings::Error> {
        self.value.validate_for(key, owner)?;
        self.machine =
            crate::settings::target_machine(key, self.machine.as_deref(), accepting_machine)?;
        Ok(self)
    }
}

record!(Assignment {units:Reference,arms:Vec<String>,method:String,assignment:BTreeMap<String,String>});
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
pub enum DecisionValue {
    Text(String),
    Detail(Structured<Decision>),
    Assignment(Structured<Assignment>),
}

impl PartialEq for DecisionValue {
    fn eq(&self, other: &Self) -> bool {
        let content = |value: &Structured<Decision>| {
            value.fields.alternatives.as_ref().is_none_or(Vec::is_empty)
                && value.fields.constraints.as_ref().is_none_or(Vec::is_empty)
                && value.fields.reverse_if.is_none()
        };
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => a == b,
            (Self::Text(text), Self::Detail(value)) | (Self::Detail(value), Self::Text(text)) => {
                *text == value.fields.text && content(value)
            }
            (Self::Detail(a), Self::Detail(b)) => {
                a.fields.text == b.fields.text
                    && a.fields.alternatives.as_deref().unwrap_or(&[])
                        == b.fields.alternatives.as_deref().unwrap_or(&[])
                    && a.fields.constraints.as_deref().unwrap_or(&[])
                        == b.fields.constraints.as_deref().unwrap_or(&[])
                    && a.fields.reverse_if == b.fields.reverse_if
            }
            (Self::Assignment(a), Self::Assignment(b)) => a == b,
            _ => false,
        }
    }
}

/// A checked shape retains structured values needed by the accepting store's
/// reference, rubric, configuration and stale-base checks.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Absent,
    Text(TextValue),
    Tag(Box<Structured<Tag>>),
    Outcome(Outcome),
    Claim(Box<ClaimValue>),
    Decision(Box<DecisionValue>),
    Forecast(Box<Structured<Forecast>>),
    Judgment(Box<Structured<Judgment>>),
    Comparison(Box<Structured<Comparison>>),
    Rubric(Box<Structured<Rubric>>),
    Pass(Box<Structured<Pass>>),
    /// The exact field type is supplied by the linked rubric revision.
    Annotation(Box<Structured<Annotation>>),
    Method(Box<Structured<Method>>),
    SavedQuery(Box<Structured<SavedQuery>>),
    Signal(Box<Structured<shared_signals::Rule>>),
    Page(Box<Structured<Page>>),
    Slot(Box<Structured<Slot>>),
    Role(Box<Structured<Role>>),
    Ping(Box<Structured<Ping>>),
    Setting(Box<Structured<Setting>>),
}
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid/{kind}: {message}")]
pub struct ShapeError {
    pub kind: String,
    pub message: String,
}

pub fn decode(
    kind: super::AssertionKind,
    key: Option<&str>,
    value: Option<&Json>,
) -> Result<Value, ShapeError> {
    use super::ValueShape;
    let invalid = |message: String| ShapeError {
        kind: kind.as_str().into(),
        message,
    };
    let required = || value.ok_or_else(|| "value is required".to_string());
    fn parse<T: serde::de::DeserializeOwned>(value: &Json) -> Result<T, String> {
        serde_json::from_value(value.clone()).map_err(|error| error.to_string())
    }
    let decoded = (|| -> Result<Value, String> {
        Ok(match kind.spec().value {
            ValueShape::Text => Value::Text(parse(required()?)?),
            ValueShape::OptionalText => {
                let value: OptionalTextValue = match value {
                    None => OptionalTextValue::Plain(None),
                    Some(value) => parse(value)?,
                };
                match value {
                    OptionalTextValue::Plain(None) => Value::Absent,
                    OptionalTextValue::Plain(Some(text)) => Value::Text(TextValue::Plain(text)),
                    OptionalTextValue::WithPredecessor(value) => match value.text {
                        Some(text) => Value::Text(TextValue::WithPredecessor(WithPredecessor {
                            text,
                            predecessor: value.predecessor,
                        })),
                        None => Value::Absent,
                    },
                }
            }
            ValueShape::Tag => Value::Tag(Box::new(match value {
                None | Some(Json::Null) => Structured::plain(Tag {
                    value: None,
                    confidence: None,
                }),
                Some(value) => parse(value)?,
            })),
            ValueShape::Outcome => {
                let value: OutcomeValue = parse(required()?)?;
                Value::Outcome(value.outcome())
            }
            ValueShape::Claim | ValueShape::Finding | ValueShape::Question => {
                let claim: ClaimValue = parse(required()?)?;
                if let ClaimValue::Detail(detail) = &claim {
                    if let Some(check) = &detail.fields.check {
                        if !matches!(kind.spec().value, ValueShape::Claim | ValueShape::Finding) {
                            return Err("only claim and finding carry a check".into());
                        }
                        check.validate()?;
                    }
                    if detail.fields.outcomes.is_some()
                        && !matches!(kind.spec().value, ValueShape::Claim | ValueShape::Question)
                    {
                        return Err("only claim and question carry outcomes".into());
                    }
                }
                Value::Claim(Box::new(claim))
            }
            ValueShape::Decision => {
                let decision: DecisionValue = parse(required()?)?;
                if let DecisionValue::Assignment(value) = &decision {
                    let assignment = &value.fields;
                    if !matches!(assignment.units, Reference::Result { row: None, .. }) {
                        return Err("assignment units must name a kept result".into());
                    }
                    let arms: BTreeSet<_> = assignment.arms.iter().map(String::as_str).collect();
                    if arms.len() < 2
                        || arms.len() != assignment.arms.len()
                        || arms.iter().any(|arm| arm.is_empty())
                    {
                        return Err(
                            "assignment needs at least two distinct nonempty arm names".into()
                        );
                    }
                    if assignment.method != "list"
                        && !assignment
                            .method
                            .strip_prefix("seeded:")
                            .is_some_and(|seed| !seed.is_empty())
                    {
                        return Err("assignment method must be list or seeded:<seed>".into());
                    }
                    if assignment
                        .assignment
                        .values()
                        .any(|arm| !arms.contains(arm.as_str()))
                    {
                        return Err("assignment names an undeclared arm".into());
                    }
                }
                Value::Decision(Box::new(decision))
            }
            ValueShape::Forecast => Value::Forecast(Box::new(parse(required()?)?)),
            ValueShape::Judgment => {
                if !matches!(key, Some("accepted" | "rejected" | "deferred")) {
                    return Err("judgment key must be accepted, rejected or deferred".into());
                }
                let judgment = match value {
                    None | Some(Json::Null) => Structured::plain(Judgment {
                        reason: None,
                        blind: None,
                    }),
                    Some(Json::String(reason)) => Structured::plain(Judgment {
                        reason: Some(reason.clone()),
                        blind: None,
                    }),
                    Some(value) => parse(value)?,
                };
                Value::Judgment(Box::new(judgment))
            }
            ValueShape::Comparison => {
                let comparison: Structured<Comparison> = parse(required()?)?;
                if let Verdict::Ranking(ranking) = &comparison.fields.verdict {
                    let distinct: BTreeSet<_> = ranking.iter().collect();
                    if ranking.len() < 2 || distinct.len() != ranking.len() {
                        return Err("ranking needs at least two distinct references".into());
                    }
                }
                if comparison.fields.rubric.as_ref().is_some_and(|reference| {
                    !matches!(
                        reference,
                        Reference::Assertion(crate::RevisionReference {
                            revision: Some(_),
                            ..
                        })
                    )
                }) {
                    return Err("comparison rubric must name an exact assertion revision".into());
                }
                Value::Comparison(Box::new(comparison))
            }
            ValueShape::Rubric => {
                let rubric: Structured<Rubric> = parse(required()?)?;
                rubric.fields.validate()?;
                Value::Rubric(Box::new(rubric))
            }
            ValueShape::Pass => {
                let pass: Structured<Pass> = parse(required()?)?;
                if pass.fields.rubric.as_ref().is_some_and(|reference| {
                    !matches!(
                        reference,
                        Reference::Assertion(crate::RevisionReference {
                            revision: Some(_),
                            ..
                        })
                    )
                }) {
                    return Err("pass rubric must name an exact assertion revision".into());
                }
                if pass.fields.population.as_ref().is_some_and(|reference| {
                    !matches!(reference, Reference::Result { row: None, .. })
                }) {
                    return Err("pass population must name a kept result".into());
                }
                Value::Pass(Box::new(pass))
            }
            ValueShape::Annotation => Value::Annotation(Box::new(parse(required()?)?)),
            ValueShape::Method => {
                let method: Structured<Method> = parse(required()?)?;
                let mut names = BTreeSet::new();
                for condition in &method.fields.conditions {
                    if condition.name.is_empty() || !names.insert(&condition.name) {
                        return Err("method conditions need distinct nonempty names".into());
                    }
                    condition.check.validate()?;
                }
                Value::Method(Box::new(method))
            }
            ValueShape::SavedQuery => {
                let saved: Structured<SavedQuery> = parse(required()?)?;
                if saved.fields.denominator.as_ref().is_some_and(|reference| {
                    Reference::parse(reference)
                        .ok()
                        .and_then(|reference| reference.question_id())
                        .is_none()
                }) {
                    return Err("saved denominator must be an exact query:<hash> reference".into());
                }
                Value::SavedQuery(Box::new(saved))
            }
            ValueShape::Signal => {
                let rule: Structured<shared_signals::Rule> = parse(required()?)?;
                if Some(rule.fields.name.as_str()) != key {
                    return Err("signal name must match its assertion key".into());
                }
                Value::Signal(Box::new(rule))
            }
            ValueShape::Page => Value::Page(Box::new(parse(required()?)?)),
            ValueShape::Slot => Value::Slot(Box::new(parse(required()?)?)),
            ValueShape::Member => match value {
                None | Some(Json::Null) => Value::Absent,
                Some(value) => {
                    let _: Structured<Member> = parse(value)?;
                    Value::Absent
                }
            },
            ValueShape::Role => Value::Role(Box::new(parse(required()?)?)),
            ValueShape::Ping => Value::Ping(Box::new(parse(required()?)?)),
            ValueShape::Setting => Value::Setting(Box::new(parse(required()?)?)),
        })
    })();
    decoded.map_err(invalid)
}
impl Check {
    pub fn validate(&self) -> Result<(), String> {
        if self.query.is_empty() {
            return Err("check query cannot be empty".into());
        }
        let parts: Vec<_> = self.expect.split_whitespace().collect();
        match parts.as_slice() {
            ["rows" | "no_rows"] => Ok(()),
            [mode @ ("count" | "agg"), operator, bound]
                if matches!(*operator, "=" | "==" | "!=" | "<" | "<=" | ">" | ">=") =>
            {
                let bound: serde_json::Number = serde_json::from_str(bound)
                    .map_err(|_| "check expectation needs a JSON number")?;
                if *mode == "count"
                    && (!number::integer(&bound) || number::compare(&bound, &0.into()).is_lt())
                {
                    return Err("count expectation needs a nonnegative whole number".into());
                }
                Ok(())
            }
            _ => Err("check expectation must be rows, no_rows, count OP N or agg OP N".into()),
        }
    }
}

/// Ordinary writes use the live value form. Only the predecessor import
/// introduces predecessor metadata; canonical replay uses decode unchanged.
pub fn decode_live(
    kind: super::AssertionKind,
    key: Option<&str>,
    value: Option<&Json>,
) -> Result<Value, ShapeError> {
    if value
        .and_then(Json::as_object)
        .is_some_and(|value| value.contains_key("predecessor"))
    {
        return Err(ShapeError {
            kind: kind.as_str().into(),
            message: "predecessor metadata belongs to the predecessor import".into(),
        });
    }
    decode(kind, key, value)
}
