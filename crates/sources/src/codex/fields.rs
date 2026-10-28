//! Codex response-item fields: each read or excluded with a reason. Records
//! that become events keep all their fields in the event's data, so only
//! items that become messages are checked.

use crate::fields::ExpectedType as T;
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
    ("type", Read(T::String)),
    ("id", Read(T::String)),
    ("timestamp", Read(T::StringOrNumber)),
    ("role", Read(T::String)),
    ("content", Read(T::StringOrArray)),
    ("content[].type", Read(T::String)),
    ("content[].text", Read(T::String)),
    ("content[].image_url", Read(T::String)),
    ("content[].file_url", Read(T::String)),
    ("content[].filename", Read(T::String)),
    // A `code` part (2025 assistant items) is kept whole as a block.
    ("content[].code", Read(T::String)),
    ("content[].language", Read(T::String)),
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
    ("phase", Read(T::String)),
    ("status", Excluded("always \"completed\" when written")),
    (
        "end_turn",
        Excluded("the turn's end is its task_complete record"),
    ),
    PASSTHROUGH,
])
.shape(
    &[],
    &[
        crate::fields::KnownKinds {
            path: "type",
            values: &["message"],
        },
        crate::fields::KnownKinds {
            path: "content[].type",
            values: &[
                "input_text",
                "output_text",
                "text",
                "Text",
                "input_image",
                "image",
                "input_file",
                "code",
            ],
        },
    ],
);

static REASONING: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("id", Read(T::String)),
    ("timestamp", Read(T::StringOrNumber)),
    ("summary", Read(T::Array)),
    ("summary[].type", Read(T::String)),
    ("summary[].text", Read(T::String)),
    ("content", Read(T::StringOrArray)),
    ("content[].type", Read(T::String)),
    ("content[].text", Read(T::String)),
    ("encrypted_content", Read(T::String)),
    PASSTHROUGH,
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &["reasoning"],
    }],
);

static CALL: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("id", Read(T::String)),
    ("timestamp", Read(T::StringOrNumber)),
    ("call_id", Read(T::String)),
    ("name", Read(T::String)),
    ("namespace", Read(T::String)),
    ("arguments", Kept("complete call input in tool block")),
    ("input", Kept("complete custom input in tool block")),
    ("action", Kept("complete action in tool block")),
    (
        "execution",
        Excluded("client or server side; the tool name already says"),
    ),
    (
        "status",
        Excluded("the call's outcome is read from its output"),
    ),
    PASSTHROUGH,
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &[
            "function_call",
            "custom_tool_call",
            "local_shell_call",
            "tool_search_call",
        ],
    }],
);

static OUTPUT: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("id", Read(T::String)),
    ("call_id", Read(T::String)),
    ("output", Kept("complete structured output in tool result")),
    ("tools", Kept("tool search output in structured result")),
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
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &[
            "function_call_output",
            "custom_tool_call_output",
            "tool_search_output",
        ],
    }],
);

static WEB_SEARCH: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("id", Read(T::String)),
    ("status", Read(T::String)),
    ("action", Kept("complete action in tool block")),
    PASSTHROUGH,
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &["web_search_call"],
    }],
);

static AGENT_MESSAGE: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("id", Read(T::String)),
    ("author", Read(T::String)),
    ("recipient", Read(T::String)),
    ("content", Read(T::StringOrArray)),
    ("content[].type", Read(T::String)),
    ("content[].text", Read(T::String)),
    (
        "content[].encrypted_content",
        Excluded("a message between agents only the API can read"),
    ),
    PASSTHROUGH,
])
.shape(
    &[],
    &[crate::fields::KnownKinds {
        path: "type",
        values: &["agent_message"],
    }],
);

static TYPE_ONLY: Fields = Fields::new(&[("type", Read(T::String))]);
