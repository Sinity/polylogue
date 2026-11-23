use super::*;
use command_surface::{Deadline, RunnableRequest};
use serde_json::json;
#[test]
fn explicit_null_is_a_wire_value_and_absence_alone_omits_a_field() {
    let commands = crate::tests::read_commands();
    let arguments = "%7B%22nullable%22%3Anull%2C%22flag%22%3Afalse%2C%22zero%22%3A0%7D";
    let supplied = RunnableRequest {
        deadline: Deadline::None,
        ..RunnableRequest::new(
            "read",
            json!({"nullable":null,"flag":false,"zero":0}),
            "synthetic",
        )
    };
    assert_eq!(
        http_url(commands, &supplied).unwrap(),
        format!(
            "/api/v1/read?operation=read&arguments={arguments}&request_id=synthetic&deadline=null"
        )
    );
    let omitted = RunnableRequest {
        deadline: Deadline::Default,
        ..supplied
    };
    assert_eq!(
        http_url(commands, &omitted).unwrap(),
        format!("/api/v1/read?operation=read&arguments={arguments}&request_id=synthetic")
    );
}
