//! Each leaf is read at its path, retained in row attributes, or reported unused.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

use polylogue_model::Json;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

/// Expected wire shapes are declarations, never inferred from provider values.
/// Nullable fields accept null separately from an absent field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedType {
    String,
    Number,
    Boolean,
    Object,
    Array,
    StringOrArray,
    StringOrObject,
    StringOrNumber,
    NumberOrObject,
    /// Tool inputs and results have an explicitly open JSON value shape.
    Any,
}

pub struct KnownKinds {
    pub path: &'static str,
    pub values: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// Reads this value. A container does not account for its children.
    Read(ExpectedType),
    /// Retains this value, including its descendants, in row attributes.
    Kept(&'static str),
    /// A known field with no parsed representation. Its leaves remain unused.
    Excluded(&'static str),
}

/// Dotted object keys and `[]` array elements. Unknown arrays count each
/// occurrence separately. A kept value is still walked down to its leaves.
pub struct Fields {
    rules: &'static [(&'static str, Rule)],
    pub dynamic_maps: &'static [&'static str],
    pub known_kinds: &'static [KnownKinds],
    record_kind: Option<&'static str>,
    compiled: OnceLock<HashMap<String, Rule>>,
}
impl Fields {
    pub const fn new(rules: &'static [(&'static str, Rule)]) -> Self {
        Self {
            rules,
            dynamic_maps: &[],
            known_kinds: &[],
            record_kind: None,
            compiled: OnceLock::new(),
        }
    }
    pub const fn shape(
        mut self,
        maps: &'static [&'static str],
        kinds: &'static [KnownKinds],
    ) -> Self {
        self.dynamic_maps = maps;
        self.known_kinds = kinds;
        self
    }
    pub const fn for_kind(mut self, kind: &'static str) -> Self {
        self.record_kind = Some(kind);
        self
    }
    /// Public shape kinds come from declarations, never arbitrary wire scalars.
    pub fn kind(&self, supplied: &str) -> &'static str {
        if let Some(kind) = self.record_kind {
            return kind;
        }
        let tail = supplied.rsplit('/').next().unwrap_or(supplied);
        self.known_kinds
            .iter()
            .filter(|known| known.path == "type")
            .find_map(|known| known.values.iter().copied().find(|kind| *kind == tail))
            .unwrap_or("*")
    }
    pub fn declarations(&self) -> &'static [(&'static str, Rule)] {
        self.rules
    }
    fn compiled(&self) -> &HashMap<String, Rule> {
        self.compiled.get_or_init(|| {
            self.rules
                .iter()
                .map(|(path, rule)| (declared_path(path), *rule))
                .collect()
        })
    }
}

pub struct Accounting {
    pub unused: Vec<String>,
    pub shapes: BTreeMap<String, polylogue_model::WireShape>,
}

pub fn unaccounted(value: &Json, fields: &Fields) -> Vec<String> {
    inspect(value, fields).unused
}

/// One walk accounts for leaves and observes shape without keeping scalar bodies.
pub fn inspect(value: &Json, fields: &Fields) -> Accounting {
    let mut unused = Vec::new();
    let mut shapes: BTreeMap<String, polylogue_model::WireShape> = BTreeMap::new();
    let mut pending = vec![(value, String::new(), String::new(), false, false)];
    while let Some((value, path, shape_path, inherited, open_map)) = pending.pop() {
        let rule = fields.compiled().get(&shape_path);
        let kept = inherited || matches!(rule, Some(Rule::Kept(_)));
        let shape = shapes.entry(shape_path.clone()).or_default();
        shape.path.clone_from(&shape_path);
        *shape.types.entry(type_name(value).into()).or_default() += 1;
        if let Some(Rule::Read(expected)) = rule {
            shape.expected = expected
                .types()
                .iter()
                .map(|kind| (*kind).to_owned())
                .collect();
        } else if rule.is_some() || open_map {
            shape.expected = ExpectedType::Any
                .types()
                .iter()
                .map(|kind| (*kind).to_owned())
                .collect();
        } else if shape_path.is_empty() && !fields.rules.is_empty() {
            shape.expected = if fields.rules.iter().any(|(path, _)| path.starts_with("[]")) {
                ExpectedType::Array
            } else {
                ExpectedType::Object
            }
            .types()
            .iter()
            .map(|kind| (*kind).to_owned())
            .collect();
        }
        let dynamic = fields
            .dynamic_maps
            .iter()
            .any(|map| declared_path(map) == shape_path);
        shape.dynamic_map = dynamic;
        if let Some(known) = fields
            .known_kinds
            .iter()
            .find(|known| declared_path(known.path) == shape_path)
        {
            shape.known_kinds = known.values.iter().map(|kind| (*kind).to_owned()).collect();
            if value
                .as_str()
                .is_some_and(|value| !known.values.contains(&value))
            {
                shape.unknown_kind_count += 1;
            }
        }
        match value {
            Json::Object(map) if !map.is_empty() => {
                for (key, child) in map.iter().rev() {
                    let child_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    let name = if dynamic { "*" } else { key.as_str() };
                    let name = if name == "*" && dynamic {
                        "*".to_owned()
                    } else {
                        escape(name)
                    };
                    let child_shape = format!("{shape_path}/{name}");
                    pending.push((child, child_path, child_shape, kept, open_map || dynamic));
                }
            }
            Json::Array(items) if !items.is_empty() => {
                for child in items.iter().rev() {
                    pending.push((
                        child,
                        format!("{path}[]"),
                        format!("{shape_path}/[]"),
                        kept,
                        open_map || dynamic,
                    ));
                }
            }
            _ if !kept && !matches!(rule, Some(Rule::Read(_))) => {
                unused.push(path);
                shape.unused_count += 1;
            }
            _ => {}
        }
    }
    unused.sort();
    Accounting { unused, shapes }
}

fn declared_path(path: &str) -> String {
    if path.is_empty() {
        return String::new();
    }
    if path.starts_with('/') {
        return path.to_owned();
    }
    let mut out = String::new();
    for part in path.split('.') {
        let mut rest = part;
        loop {
            if let Some(at) = rest.find("[]") {
                if at > 0 {
                    out.push('/');
                    out.push_str(&if &rest[..at] == "*" {
                        "*".to_owned()
                    } else {
                        escape(&rest[..at])
                    });
                }
                out.push_str("/[]");
                rest = &rest[at + 2..];
                if rest.is_empty() {
                    break;
                }
            } else {
                out.push('/');
                out.push_str(&if rest == "*" {
                    "*".to_owned()
                } else {
                    escape(rest)
                });
                break;
            }
        }
    }
    out
}

fn escape(name: &str) -> String {
    name.replace('~', "~0")
        .replace('/', "~1")
        .replace('*', "~2")
        .replace("[]", "~3")
}

/// Parent records expose their own fields; each child record is accounted by
/// its own parser table. Empty placeholders retain the container's wire type.
pub fn header(value: &Json, records: &[&str]) -> Json {
    let Some(object) = value.as_object() else {
        return value.clone();
    };
    Json::Object(
        object
            .iter()
            .map(|(key, value)| {
                let value = if records.contains(&key.as_str()) {
                    match value {
                        Json::Object(_) => Json::Object(Default::default()),
                        Json::Array(_) => Json::Array(Vec::new()),
                        other => other.clone(),
                    }
                } else {
                    value.clone()
                };
                (key.clone(), value)
            })
            .collect(),
    )
}

fn type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Number(_) => "number",
        Json::String(_) => "string",
        Json::Array(_) => "array",
        Json::Object(_) => "object",
    }
}
impl ExpectedType {
    pub const fn types(self) -> &'static [&'static str] {
        match self {
            Self::String => &["null", "string"],
            Self::Number => &["null", "number"],
            Self::Boolean => &["null", "boolean"],
            Self::Object => &["null", "object"],
            Self::Array => &["null", "array"],
            Self::StringOrArray => &["null", "string", "array"],
            Self::StringOrObject => &["null", "string", "object"],
            Self::StringOrNumber => &["null", "string", "number"],
            Self::NumberOrObject => &["null", "number", "object"],
            Self::Any => &["null", "boolean", "number", "string", "array", "object"],
        }
    }
}

/// Copies only fields declared retained. Attribute names are their full wire
/// paths; array-bearing retained values are copied at their enclosing path.
pub fn attributes(value: &Json, fields: &Fields) -> serde_json::Map<String, Json> {
    fields
        .rules
        .iter()
        .filter_map(|(path, rule)| {
            if !matches!(rule, Rule::Kept(_)) {
                return None;
            }
            let mut current = value;
            if path.is_empty() {
                return Some((String::new(), current.clone()));
            }
            if path.starts_with('/') {
                for part in path[1..].split('/') {
                    if matches!(part, "[]" | "*") {
                        return None;
                    }
                    let key = part
                        .replace("~3", "[]")
                        .replace("~2", "*")
                        .replace("~1", "/")
                        .replace("~0", "~");
                    current = current.get(&key)?;
                }
            } else {
                if path.contains("[]") || path.split('.').any(|part| part == "*") {
                    return None;
                }
                for part in path.split('.') {
                    current = current.get(part)?;
                }
            }
            Some(((*path).to_owned(), current.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests;
