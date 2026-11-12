use super::*;
use command_surface::{Deadline, RunnableRequest};
use serde_json::json;
#[test]
fn explicit_null_is_a_wire_value_and_absence_alone_omits_a_field() {
    let commands = crate::tests::read_commands();
    let supplied = RunnableRequest {
        deadline: Deadline::None,
        ..RunnableRequest::new(
            "read",
            json!({"nullable":null,"flag":false,"zero":0}),
            "synthetic",
        )
    };
    let url = http_url(commands, &supplied).unwrap();
    assert!(url.contains("deadline=null"), "{url}");
    let fields: serde_json::Value = serde_json::to_value(&supplied).unwrap();
    assert_eq!(
        fields["arguments"],
        json!({"nullable":null,"flag":false,"zero":0})
    );
    assert!(url.contains("%22nullable%22%3Anull"), "{url}");
    let omitted = RunnableRequest {
        deadline: Deadline::Default,
        ..supplied
    };
    let url = http_url(commands, &omitted).unwrap();
    assert!(!url.contains("deadline="), "{url}");
    assert!(url.contains("%22nullable%22%3Anull"), "{url}");
}
