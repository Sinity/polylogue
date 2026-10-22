//! Every wire field is read, or excluded with a declared reason. An origin
//! declares its fields per record kind; [`unaccounted`] lists the fields a
//! record carries that the declaration does not mention.

use polylogue_model::Json;
use std::collections::HashSet;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The parser reads it into a row.
    Read,
    /// Kept whole in the stored record or an event's data, not interpreted
    /// yet, for the stated reason.
    Kept(&'static str),
    /// Not read, for the stated reason.
    Excluded(&'static str),
}

/// Paths are dotted object keys; `[]` stands for every element of an array
/// (`message.content[].text`). A declared path covers everything below it
/// unless deeper paths are declared too.
pub struct Fields {
    rules: &'static [(&'static str, Rule)],
    compiled: OnceLock<Compiled>,
}

struct Compiled {
    exact: HashSet<&'static str>,
    deeper: HashSet<&'static str>,
}

impl Fields {
    /// Declare as `static` so parsing threads share the lookup.
    pub const fn new(rules: &'static [(&'static str, Rule)]) -> Self {
        Self {
            rules,
            compiled: OnceLock::new(),
        }
    }

    fn compiled(&self) -> &Compiled {
        self.compiled.get_or_init(|| {
            let mut exact = HashSet::new();
            let mut deeper = HashSet::new();
            for &(path, _) in self.rules {
                exact.insert(path);
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

fn walk(value: &Json, path: &str, fields: &Compiled, out: &mut Vec<String>) {
    match value {
        Json::Object(map) => {
            for (key, child) in map {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                if fields.deeper.contains(child_path.as_str()) {
                    walk(child, &child_path, fields, out);
                } else if !fields.exact.contains(child_path.as_str()) {
                    out.push(child_path);
                }
            }
        }
        Json::Array(items) => {
            let element = format!("{path}[]");
            if fields.deeper.contains(element.as_str()) {
                for item in items {
                    walk(item, &element, fields, out);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
