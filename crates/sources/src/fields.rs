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
    compiled: OnceLock<HashMap<&'static str, Rule>>,
}
impl Fields {
    pub const fn new(rules: &'static [(&'static str, Rule)]) -> Self {
        Self {
            rules,
            dynamic_maps: &[],
            known_kinds: &[],
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
    pub fn declarations(&self) -> &'static [(&'static str, Rule)] {
        self.rules
    }
    fn compiled(&self) -> &HashMap<&'static str, Rule> {
        self.compiled
            .get_or_init(|| self.rules.iter().copied().collect())
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
    let mut pending = vec![(value, String::new(), String::new(), false)];
    while let Some((value, path, shape_path, inherited)) = pending.pop() {
        let rule = fields.compiled().get(path.as_str());
        let kept = inherited || matches!(rule, Some(Rule::Kept(_)));
        let shape = shapes.entry(shape_path.clone()).or_default();
        *shape.types.entry(type_name(value).into()).or_default() += 1;
        if let Some(Rule::Read(expected)) = rule {
            shape.expected = expected
                .types()
                .iter()
                .map(|kind| (*kind).to_owned())
                .collect();
        }
        shape.dynamic_map = fields.dynamic_maps.contains(&path.as_str());
        if let Some(known) = fields.known_kinds.iter().find(|known| known.path == path) {
            shape.known_kinds = known.values.iter().map(|kind| (*kind).to_owned()).collect();
        }
        match value {
            Json::Object(map) if !map.is_empty() => {
                for (key, child) in map.iter().rev() {
                    let child_path = if path.is_empty() {
                        key.clone()
                    } else {
                        format!("{path}.{key}")
                    };
                    let name = if fields.dynamic_maps.contains(&path.as_str()) {
                        "*"
                    } else {
                        key.as_str()
                    };
                    let child_shape = if shape_path.is_empty() {
                        name.to_owned()
                    } else {
                        format!("{shape_path}.{name}")
                    };
                    pending.push((child, child_path, child_shape, kept));
                }
            }
            Json::Array(items) if !items.is_empty() => {
                for child in items.iter().rev() {
                    pending.push((child, format!("{path}[]"), format!("{shape_path}[]"), kept));
                }
            }
            _ if !kept && !matches!(rule, Some(Rule::Read(_))) => unused.push(path),
            _ => {}
        }
    }
    Accounting { unused, shapes }
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
            if !matches!(rule, Rule::Kept(_)) || path.contains("[]") {
                return None;
            }
            let mut current = value;
            for part in path.split('.') {
                current = current.get(part)?;
            }
            Some(((*path).to_owned(), current.clone()))
        })
        .collect()
}

#[cfg(test)]
mod tests;
