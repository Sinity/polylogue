use polylogue_model::{Block, Message, Timestamp, Usage};
use serde_json::json;

pub fn rows() -> Result<Vec<Vec<u8>>, serde_json::Error> {
    let mut rows = Vec::new();
    for n in 0..512_u64 {
        for origin in ["claude-code", "codex", "chatgpt", "claude-ai"] {
            let mut message: Message = serde_json::from_value(json!({
                "id":format!("{origin}:sample-{n}"),
                "session":format!("{origin}:sample-session"),
                "parent":format!("{origin}:sample-parent"),
                "role":if n % 2 == 0 {"user"} else {"assistant"},
                "model":"synthetic-model",
                "prompt_kind":if n % 2 == 0 {"prose"} else {"model"},
                "blocks":[],
                "usage":{},
                "stop_reason":if n % 3 == 0 {"tool_use"} else {"end_turn"},
                "reported_cost_usd":0.125,
                "duration_ms":n + 1,
                "production":{
                    "effort":"high",
                    "requested_model":"requested-model",
                    "advisor_model":"advisor-model",
                    "thinking_ms":n,
                    "first_token_ms":n + 1,
                    "permission":{"decision":"allowed","source":"synthetic"},
                    "request":format!("request-{n}"),
                    "request_id":format!("api-request-{n}"),
                    "turn_origin":"sdk",
                    "attribution":{"agent":"sample-agent","skill":"sample-skill",
                        "plugin":"sample-plugin","mcp_server":"sample-server","mcp_tool":"sample-tool"}
                }
            }))?;
            message.at = Timestamp::from_unix_ms(1_000 + n as i64);
            if n % 3 != 0 {
                message.usage = Usage {
                    input: Some(n),
                    cached_input: Some(n * 2),
                    cache_write: Some(n * 3),
                    output: Some(n + 1),
                    reasoning: Some(n / 2),
                    cache_write_1h: Some(n),
                    service_tier: Some("standard".into()),
                    speed: Some("standard".into()),
                };
            }
            rows.push(serde_json::to_vec(&message)?);
        }
        for value in [
            json!({"type":"text","text":""}),
            json!({"type":"thinking","text":"","signature":format!("signature-{n}"),"redacted":false}),
            json!({"type":"thinking","text":null,"redacted":true}),
            json!({"type":"tool_call","call_id":format!("call-{n}"),"tool":"shell","action":"shell",
                "input":{"command":format!("printf sample-{n}"),"path":"/synthetic/file"},
                "command":format!("printf sample-{n}"),"path":"/synthetic/file"}),
            json!({"type":"tool_result","call_id":format!("call-{n}"),"output":"",
                "outcome":{"kind":"ok"},"data":{"nested":[null,true,n]},
                "attachments":[{"name":"sample-file","direction":"output"}]}),
            json!({"type":"tool_result","call_id":null,"output":null,"outcome":{"kind":"no_result"}}),
            json!({"type":"image","name":"sample-image","direction":"input"}),
            json!({"type":"file","name":"sample-file","direction":"output"}),
            json!({"type":"other","type_name":"synthetic-future","data":{"nested":[null,false,n]}}),
        ] {
            let block: Block = serde_json::from_value(value)?;
            rows.push(serde_json::to_vec(&block)?);
        }
        rows.push(serde_json::to_vec(&json!({
            "event":"synthetic-event",
            "request_id":format!("request-{n}"),
            "record_kind":"synthetic",
            "permission":{"decision":"allowed"},
            "nested":[null,true,n]
        }))?);
    }
    Ok(rows)
}
