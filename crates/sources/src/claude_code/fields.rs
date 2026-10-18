//! Claude Code's message-record fields: each read or excluded with a
//! reason. Records that become events keep all their fields in the event's
//! data, so only message records are checked.

use crate::fields::{Fields, Rule::*};

pub const MESSAGE_FIELDS: Fields = &[
    ("type", Read),
    ("uuid", Read),
    ("parentUuid", Read),
    ("sessionId", Read),
    ("agentId", Read),
    ("timestamp", Read),
    ("cwd", Read),
    ("gitBranch", Read),
    ("version", Read),
    ("isSidechain", Read),
    ("isMeta", Read),
    ("isCompactSummary", Read),
    ("requestId", Read),
    ("toolUseResult", Read),
    ("toolDenialKind", Read),
    ("costUSD", Read),
    ("durationMs", Read),
    ("origin", Read),
    (
        "promptId",
        Excluded("groups the records of one prompt; the uuid chain already does"),
    ),
    (
        "logicalParentUuid",
        Excluded("parent across a compaction; the compaction event keeps it"),
    ),
    ("userType", Excluded("always \"external\" in observed data")),
    (
        "entrypoint",
        Excluded("launch surface (cli, sdk); not used by any view yet"),
    ),
    (
        "slug",
        Excluded("display label the tool derives from the title"),
    ),
    (
        "isVisibleInTranscriptOnly",
        Excluded("display hint for the tool's own UI"),
    ),
    (
        "isApiErrorMessage",
        Excluded("the <synthetic> model name already marks these"),
    ),
    (
        "promptSource",
        Excluded("not read yet; written since 2.1.161"),
    ),
    (
        "permissionMode",
        Excluded("mode changes arrive as permission-mode records"),
    ),
    ("teamName", Excluded("not read yet")),
    ("message", Read),
    ("message.role", Read),
    ("message.id", Read),
    ("message.model", Read),
    ("message.usage", Read),
    ("message.stop_reason", Read),
    (
        "message.stop_sequence",
        Excluded("the matched text; stop_reason is read"),
    ),
    ("message.type", Excluded("always \"message\"")),
    ("message.content", Read),
    ("message.content[].type", Read),
    ("message.content[].text", Read),
    ("message.content[].thinking", Read),
    ("message.content[].signature", Read),
    ("message.content[].data", Read),
    ("message.content[].id", Read),
    ("message.content[].name", Read),
    ("message.content[].input", Read),
    ("message.content[].tool_use_id", Read),
    ("message.content[].content", Read),
    ("message.content[].is_error", Read),
    ("message.content[].source", Read),
    ("message.content[].toolDenialKind", Read),
    ("message.content[].backgroundTaskId", Read),
    (
        "message.content[].cache_control",
        Excluded("prompt-cache hint sent to the API"),
    ),
    (
        "message.content[].citations",
        Excluded("not read yet; stays in the stored bytes"),
    ),
];
