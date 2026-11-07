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
mod tests {
    use super::*;
    #[test]
    fn parent_deadline_surrounds_the_unchanged_request() {
        let request: command_surface::RunnableRequest = serde_json::from_value(serde_json::json!({
            "operation":"brief", "arguments":{"cwd":"/synthetic"}, "request_id":"bounded-prepare", "deadline":null, "budget_tokens":200
        })).unwrap();
        let wire = serde_json::to_string(&Outgoing {
            request: &request,
            caller: CallerEvidence::default(),
            parent_deadline_ms: Some(123),
        })
        .unwrap();
        let Incoming {
            request: accepted,
            parent_deadline_ms: parent,
            ..
        } = incoming(&wire).unwrap().unwrap();
        assert_eq!(accepted, request);
        assert_eq!(parent, Some(123));
    }

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
        let Incoming {
            request: accepted,
            caller,
            parent_deadline_ms,
        } = incoming(&serde_json::to_string(&request).unwrap())
            .unwrap()
            .unwrap();
        assert_eq!(accepted, request);
        assert_eq!(parent_deadline_ms, None);
        assert_eq!(caller, CallerEvidence::default());
    }
}
