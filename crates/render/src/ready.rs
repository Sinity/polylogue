//! Pure links to complete operation-table supplied HTTP calls.
use command_surface::{Failure, FailureKind};
use polylogue_model::read::HttpCall;
use serde_json::Value;

/// Encode the operation's complete HTTP form; no cursor parsing, target
/// selection or option defaults occur here. Complex fields use JSON.
pub fn http_url(call: &HttpCall) -> Result<String, Failure> {
    if call.method != "GET"
        || !call.path.starts_with('/')
        || call.path.starts_with("//")
        || call.path.contains(['?', '#'])
        || !crate::html::safe_url(&call.path)
    {
        return Err(Failure::new(
            FailureKind::BadRequest,
            "read navigation requires a local GET operation",
        ));
    }
    let fields = call.arguments.as_object().ok_or_else(|| {
        Failure::new(
            FailureKind::BadRequest,
            "read HTTP arguments must be an object",
        )
    })?;
    let mut url = call.path.clone();
    let mut first = true;
    for (name, value) in fields {
        // Null is a supplied wire value; only an absent field selects defaults.
        let values = match value {
            Value::Array(values) => values.as_slice(),
            _ => std::slice::from_ref(value),
        };
        for value in values {
            let value = match value {
                Value::String(value) => value.clone(),
                _ => value.to_string(),
            };
            url.push(if first { '?' } else { '&' });
            first = false;
            url.push_str(&crate::html::encode_path_segment(name));
            url.push('=');
            url.push_str(&crate::html::encode_path_segment(&value));
        }
    }
    Ok(url)
}

#[cfg(test)]
mod tests;
