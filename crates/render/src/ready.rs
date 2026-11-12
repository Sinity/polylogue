//! HTTP links derived from the request and its operation declaration.
use command_surface::{Access, Failure, FailureKind, OperationDeclarations, RunnableRequest};
use serde_json::Value;

/// Encode the operation's complete HTTP form; no cursor parsing, target
/// selection or option defaults occur here. Complex fields use JSON.
pub fn http_url(
    declarations: &OperationDeclarations,
    request: &RunnableRequest,
) -> Result<String, Failure> {
    let operation = declarations
        .descriptions()
        .iter()
        .find(|description| description.spec.name == request.operation)
        .ok_or_else(|| Failure::new(FailureKind::BadRequest, "unknown operation"))?;
    let path = operation
        .spec
        .http_route
        .as_deref()
        .ok_or_else(|| Failure::new(FailureKind::BadRequest, "operation has no HTTP route"))?;
    if operation.spec.access != Access::Read
        || !path.starts_with('/')
        || path.starts_with("//")
        || path.contains(['?', '#'])
        || !crate::html::safe_url(path)
    {
        return Err(Failure::new(
            FailureKind::BadRequest,
            "read navigation requires a local GET operation",
        ));
    }
    let arguments = serde_json::to_value(request)
        .map_err(|error| Failure::new(FailureKind::BadRequest, error.to_string()))?;
    let fields = arguments.as_object().ok_or_else(|| {
        Failure::new(
            FailureKind::BadRequest,
            "read HTTP arguments must be an object",
        )
    })?;
    let mut url = path.to_owned();
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
