//! Mask decoded output strings before an encoder escapes their contents.

use serde_json::Value;

/// Text with credentials masked, as one block.
pub fn mask(text: &str) -> String {
    let mut masker = credential_masking::Masker::new();
    let mut out = Vec::with_capacity(text.len());
    masker
        .push(text, &mut out)
        .expect("writing a Vec cannot fail");
    masker.finish(&mut out).expect("writing a Vec cannot fail");
    String::from_utf8(out).expect("masking preserves UTF-8")
}

/// Preserve JSON types and all values, including empty fields. Credential-shaped
/// keys are masked too. If keys collapse to one label, append a number rather
/// than overwriting a value; ordinary keys retain their original names.
pub fn mask_json(value: &mut Value) {
    let mut pending = vec![value];
    while let Some(value) = pending.pop() {
        match value {
            Value::String(text) => *text = mask(text),
            Value::Array(values) => pending.extend(values.iter_mut()),
            Value::Object(values) => {
                let mut renamed = Vec::new();
                for (key, value) in std::mem::take(values) {
                    let masked = mask(&key);
                    if masked == key {
                        values.insert(key, value);
                    } else {
                        renamed.push((masked, value));
                    }
                }
                for (label, value) in renamed {
                    let mut key = label.clone();
                    let mut suffix = 2;
                    while values.contains_key(&key) {
                        key = format!("{label} #{suffix}");
                        suffix += 1;
                    }
                    values.insert(key, value);
                }
                pending.extend(values.values_mut());
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
