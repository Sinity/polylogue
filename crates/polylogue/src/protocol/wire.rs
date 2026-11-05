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
        let caller = match value
            .as_object_mut()
            .expect("operation belongs to an object")
            .remove("caller")
        {
            Some(caller) => serde_json::from_value(caller)?,
            None => CallerEvidence::default(),
        };
        let request = serde_json::from_value(value)?;
        Ok((request, caller))
    })())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_shared_envelope_needs_no_client_supplied_identity_evidence() {
        let request = command_surface::Request {
            operation: "status".into(),
            arguments: serde_json::json!({}),
            request_id: "bare-shared-envelope".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let (accepted, caller) = incoming(&serde_json::to_string(&request).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(accepted, request);
        assert_eq!(caller, CallerEvidence::default());
    }
}
