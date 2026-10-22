//! Codex response-item fields: each read or excluded with a reason. Records
//! that become events keep all their fields in the event's data, so only
//! items that become messages are checked.

use crate::fields::{Fields, Rule::*};

const PASSTHROUGH: (&str, crate::fields::Rule) = (
    "internal_chat_message_metadata_passthrough",
    Excluded("opaque metadata the harness hands back to the API"),
);

pub fn for_item(ty: &str) -> &'static Fields {
    match ty {
        "message" => &MESSAGE,
        "reasoning" => &REASONING,
        "function_call" | "custom_tool_call" | "local_shell_call" | "tool_search_call" => &CALL,
        "function_call_output" | "custom_tool_call_output" | "tool_search_output" => &OUTPUT,
        "web_search_call" => &WEB_SEARCH,
        "agent_message" => &AGENT_MESSAGE,
        _ => &TYPE_ONLY,
    }
}

static MESSAGE: Fields = Fields::new(&[
    ("type", Read),
    ("id", Read),
    ("timestamp", Read),
    ("role", Read),
    ("content", Read),
    ("content[].type", Read),
    ("content[].text", Read),
    ("content[].image_url", Read),
    ("content[].file_url", Read),
    ("content[].filename", Read),
    // A `code` part (2025 assistant items) is kept whole as a block.
    ("content[].code", Read),
    ("content[].language", Read),
    (
        "content[].detail",
        Excluded("image resolution hint sent to the API"),
    ),
    (
        "content[].annotations",
        Excluded("not read yet; stays in the stored bytes"),
    ),
    (
        "content[].logprobs",
        Excluded("not read yet; stays in the stored bytes"),
    ),
    (
        "phase",
        Excluded("commentary or final answer; not used by any view yet"),
    ),
    ("status", Excluded("always \"completed\" when written")),
    (
        "end_turn",
        Excluded("the turn's end is its task_complete record"),
    ),
    PASSTHROUGH,
]);

static REASONING: Fields = Fields::new(&[
    ("type", Read),
    ("id", Read),
    ("timestamp", Read),
    ("summary", Read),
    ("summary[].type", Read),
    ("summary[].text", Read),
    ("content", Read),
    ("content[].type", Read),
    ("content[].text", Read),
    ("encrypted_content", Read),
    PASSTHROUGH,
]);

static CALL: Fields = Fields::new(&[
    ("type", Read),
    ("id", Read),
    ("timestamp", Read),
    ("call_id", Read),
    ("name", Read),
    ("namespace", Read),
    ("arguments", Read),
    ("input", Read),
    ("action", Read),
    (
        "execution",
        Excluded("client or server side; the tool name already says"),
    ),
    (
        "status",
        Excluded("the call's outcome is read from its output"),
    ),
    PASSTHROUGH,
]);

static OUTPUT: Fields = Fields::new(&[
    ("type", Read),
    ("id", Read),
    ("call_id", Read),
    ("output", Read),
    ("tools", Read),
    ("name", Excluded("repeats the call's tool name")),
    ("namespace", Excluded("repeats the call's namespace")),
    (
        "execution",
        Excluded("client or server side; the call says"),
    ),
    (
        "status",
        Excluded("the outcome is read from the output text"),
    ),
    PASSTHROUGH,
]);

static WEB_SEARCH: Fields = Fields::new(&[
    ("type", Read),
    ("id", Read),
    ("status", Read),
    ("action", Read),
    PASSTHROUGH,
]);

static AGENT_MESSAGE: Fields = Fields::new(&[
    ("type", Read),
    ("id", Read),
    ("author", Read),
    ("recipient", Read),
    ("content", Read),
    ("content[].type", Read),
    ("content[].text", Read),
    (
        "content[].encrypted_content",
        Excluded("a message between agents only the API can read"),
    ),
    PASSTHROUGH,
]);

static TYPE_ONLY: Fields = Fields::new(&[("type", Read)]);
