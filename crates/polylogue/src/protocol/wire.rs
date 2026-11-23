//! Caller details from the transport surround the shared request without entering operation arguments.
use command_surface::CallerEvidence;
use serde::Serialize;

#[derive(Serialize)]
pub struct Outgoing<'a> {
    #[serde(flatten)]
    pub request: &'a command_surface::RunnableRequest,
    pub caller: CallerEvidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_deadline_ms: Option<i64>,
}

pub struct Incoming {
    pub request: command_surface::RunnableRequest,
    pub caller: CallerEvidence,
    pub parent_deadline_ms: Option<i64>,
}

/// Reads one request line; the error names what is wrong with it, including
/// text that is not JSON or names no operation.
pub fn incoming(text: &str) -> Result<Incoming, serde_json::Error> {
    use serde::de::Error as _;
    let mut value: serde_json::Value = serde_json::from_str(text)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| serde_json::Error::custom("a request is a JSON object"))?;
    let caller = match object.remove("caller") {
        Some(caller) => serde_json::from_value(caller)?,
        None => CallerEvidence::default(),
    };
    let parent_deadline_ms = object
        .remove("parent_deadline_ms")
        .map(serde_json::from_value)
        .transpose()?;
    let request = serde_json::from_value(value)?;
    Ok(Incoming {
        request,
        caller,
        parent_deadline_ms,
    })
}

#[cfg(test)]
mod tests;
