//! Transport evidence surrounds the shared request without entering operation arguments.
use polylogue_model::authorship::CallerEvidence;
use serde::Serialize;

#[derive(Serialize)]
pub struct Outgoing<'a> {
    #[serde(flatten)]
    pub request: &'a command_surface::RunnableRequest,
    pub caller: CallerEvidence,
}

pub fn incoming(
    text: &str,
) -> Option<Result<(command_surface::RunnableRequest, CallerEvidence), serde_json::Error>> {
    let mut value: serde_json::Value = serde_json::from_str(text).ok()?;
    value.get("operation")?;
    Some((|| {
        let caller = serde_json::from_value(
            value
                .as_object_mut()
                .expect("operation belongs to an object")
                .remove("caller")
                .unwrap_or(serde_json::Value::Null),
        )?;
        let request = serde_json::from_value(value)?;
        Ok((request, caller))
    })())
}
