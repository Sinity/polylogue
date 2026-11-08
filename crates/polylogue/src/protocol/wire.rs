//! Transport evidence surrounds the shared request without entering operation arguments.
use polylogue_model::authorship::CallerEvidence;
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

pub fn incoming(text: &str) -> Option<Result<Incoming, serde_json::Error>> {
    let mut value: serde_json::Value = serde_json::from_str(text).ok()?;
    value.get("operation")?;
    Some((|| {
        let caller = match value
            .as_object_mut()
            .expect("operation belongs to an object")
            .remove("caller")
        {
            Some(caller) => serde_json::from_value(caller)?,
            None => CallerEvidence::default(),
        };
        let parent_deadline_ms = value
            .as_object_mut()
            .expect("operation object")
            .remove("parent_deadline_ms")
            .map(serde_json::from_value)
            .transpose()?;
        let request = serde_json::from_value(value)?;
        Ok(Incoming {
            request,
            caller,
            parent_deadline_ms,
        })
    })())
}

#[cfg(test)]
mod tests;
