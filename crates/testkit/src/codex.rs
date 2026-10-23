//! A native item-only rollout with orphan usage, delayed tool results and quota.
use serde_json::json;

pub fn observations(thread: &str) -> String {
    let counts = |input, output| {
        json!({"input_tokens":input,"cached_input_tokens":0,
        "output_tokens":output,"reasoning_output_tokens":0,"total_tokens":input+output})
    };
    let info = |input, output, last_input, last_output| {
        json!({"total_token_usage":counts(input,output),
        "last_token_usage":counts(last_input,last_output)})
    };
    let rows = [
        (
            "session_meta",
            json!({"id":thread,"source":"cli","cli_version":"0.151.0"}),
        ),
        ("turn_context", json!({"model":"gpt-6.1-sol"})),
        (
            "event_msg",
            json!({"type":"token_count","info":info(10,2,10,2),"rate_limits":null}),
        ),
        (
            "response_item",
            json!({"type":"reasoning","id":"r1","summary":[{"type":"summary_text",
            "text":"-----BEGIN PRIVATE KEY----- synthetic-private-prefix"}]}),
        ),
        (
            "response_item",
            json!({"type":"function_call","id":"f1","call_id":"call1","name":"synthetic_tool","arguments":"{}"}),
        ),
        (
            "response_item",
            json!({"type":"function_call_output","id":"out1","call_id":"call1","output":"synthetic tool output"}),
        ),
        (
            "response_item",
            json!({"type":"message","id":"c1","role":"assistant","content":[{"type":"output_text",
            "text":"synthetic-private-tail -----END PRIVATE KEY----- publicneedle"}]}),
        ),
        (
            "event_msg",
            json!({"type":"item_completed","item":{"type":"CommandExecution","id":"native-child",
            "command":["echo","synthetic"],"status":"completed","exit_code":0,"aggregated_output":"synthetic child output"}}),
        ),
        (
            "event_msg",
            json!({"type":"token_count","info":info(30,5,20,3),"rate_limits":null}),
        ),
        (
            "event_msg",
            json!({"type":"token_count","info":info(30,5,20,3),"rate_limits":null}),
        ),
        (
            "event_msg",
            json!({"type":"token_count","info":null,"rate_limits":{"primary":{
            "window_minutes":300,"used_percent":12,"resets_in_seconds":60}}}),
        ),
    ];
    rows.into_iter()
        .enumerate()
        .map(|(i, (ty, payload))| {
            format!(
                "{}\n",
                json!({
        "timestamp":format!("2026-10-23T10:00:{i:02}Z"),"ordinal":i+1,"type":ty,"payload":payload})
            )
        })
        .collect()
}
