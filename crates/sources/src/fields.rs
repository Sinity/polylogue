//! Wire fields are read at their exact path, indexed whole, kept in the
//! stored input, or deliberately ignored with a reason.

use polylogue_model::Json;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The parser reads this value. Objects and arrays still need rules
    /// for their children.
    Read,
    /// The complete value is stored in an index row, including its children.
    Indexed(&'static str),
    /// Kept in the stored input, with no parsed representation yet.
    Kept(&'static str),
    /// Not read, for the stated reason.
    Excluded(&'static str),
}

/// Dotted object keys; `[]` stands for each array element. Only `Indexed`,
/// `Kept` and `Excluded` cover descendants. `Read` never hides unread children.
pub struct Fields {
    rules: &'static [(&'static str, Rule)],
    compiled: OnceLock<Compiled>,
}

struct Compiled {
    exact: HashMap<&'static str, Rule>,
    deeper: HashSet<&'static str>,
}

impl Fields {
    pub const fn new(rules: &'static [(&'static str, Rule)]) -> Self {
        Self {
            rules,
            compiled: OnceLock::new(),
        }
    }

    fn compiled(&self) -> &Compiled {
        self.compiled.get_or_init(|| {
            let mut exact = HashMap::new();
            let mut deeper = HashSet::new();
            for &(path, rule) in self.rules {
                exact.insert(path, rule);
                for (at, ch) in path.char_indices() {
                    if matches!(ch, '.' | '[') {
                        deeper.insert(&path[..at]);
                    }
                }
            }
            Compiled { exact, deeper }
        })
    }
}

pub fn unaccounted(value: &Json, fields: &Fields) -> Vec<String> {
    let mut out = Vec::new();
    walk(value, "", fields.compiled(), &mut out);
    out
}

fn visit(value: &Json, path: &str, fields: &Compiled, out: &mut Vec<String>) {
    let rule = fields.exact.get(path);
    if matches!(
        rule,
        Some(Rule::Indexed(_) | Rule::Kept(_) | Rule::Excluded(_))
    ) {
        return;
    }
    let container = match value {
        Json::Object(map) => !map.is_empty(),
        Json::Array(items) => !items.is_empty(),
        _ => false,
    };
    if container {
        walk(value, path, fields, out);
    } else if rule.is_none() && !fields.deeper.contains(path) {
        out.push(path.to_owned());
    }
}

fn walk(value: &Json, path: &str, fields: &Compiled, out: &mut Vec<String>) {
    match value {
        Json::Object(map) => {
            for (key, child) in map {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                visit(child, &child_path, fields, out);
            }
        }
        Json::Array(items) => {
            let element = format!("{path}[]");
            for item in items {
                visit(item, &element, fields, out);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
