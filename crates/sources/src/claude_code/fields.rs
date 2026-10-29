//! Claude Code's message-record fields: each read or excluded with a
//! reason. Records that become events keep all their fields in the event's
//! data, so only message records are checked.

use crate::fields::ExpectedType as T;
use crate::fields::{Fields, Rule::*};

pub static MESSAGE_FIELDS: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("uuid", Read(T::String)),
    ("parentUuid", Read(T::String)),
    ("sessionId", Read(T::String)),
    ("agentId", Read(T::String)),
    ("timestamp", Read(T::StringOrNumber)),
    ("cwd", Read(T::String)),
    ("gitBranch", Read(T::String)),
    ("version", Read(T::String)),
    ("isSidechain", Read(T::Boolean)),
    ("isMeta", Read(T::Boolean)),
    ("isCompactSummary", Read(T::Boolean)),
    ("requestId", Read(T::String)),
    (
        "toolUseResult",
        Kept("structured tool results retained beside paired blocks"),
    ),
    ("toolDenialKind", Read(T::String)),
    ("costUSD", Read(T::Number)),
    ("durationMs", Read(T::Number)),
    ("origin", Read(T::Object)),
    ("origin.kind", Read(T::String)),
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
    ("slug", Read(T::String)),
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
    ("teamName", Read(T::String)),
    ("forkedFrom", Read(T::Object)),
    ("forkedFrom.sessionId", Read(T::String)),
    ("forkedFrom.messageUuid", Read(T::String)),
    ("sessionKind", Read(T::String)),
    ("effort", Read(T::String)),
    ("perTurnEffort", Read(T::String)),
    ("requestedModel", Read(T::String)),
    ("advisorModel", Read(T::String)),
    ("attributionAgent", Read(T::String)),
    ("attributionSkill", Read(T::String)),
    ("attributionPlugin", Read(T::String)),
    ("attributionMcpServer", Read(T::String)),
    ("attributionMcpTool", Read(T::String)),
    ("thinkingDurationMs", Read(T::Number)),
    ("error", Read(T::String)),
    ("apiErrorStatus", Read(T::Number)),
    (
        "permissionDecision",
        Kept("complete permission decision in production"),
    ),
    ("requestRef", Read(T::String)),
    ("turnOrigin", Read(T::String)),
    ("quotaLimits", Kept("complete quota reading in production")),
    (
        "toolEndsTurn",
        Kept("whether the tool's result ends the turn; no reader yet"),
    ),
    (
        "sourceToolUseID",
        Kept("the tool call a harness message came from; no reader yet"),
    ),
    ("queuePriority", Kept("queued-prompt handling hint")),
    ("queueOrigin", Kept("queued-prompt handling hint")),
    ("queueTranscriptOnly", Kept("queued-prompt handling hint")),
    (
        "queueSkipSlashCommands",
        Kept("queued-prompt handling hint"),
    ),
    (
        "interruptedMessageId",
        Kept("the message this prompt interrupted; no reader yet"),
    ),
    (
        "interruptedByShutdown",
        Kept("the turn ended because the tool shut down; no reader yet"),
    ),
    ("mcpMeta", Kept("MCP result metadata; no reader yet")),
    (
        "turnCompanion",
        Kept("extra text the harness showed; no reader yet"),
    ),
    (
        "planContent",
        Kept("plan text approved in plan mode; the plans side files are read"),
    ),
    (
        "joinedNoticeLengths",
        Kept("background notices merged into this prompt"),
    ),
    (
        "joinedNoticeOwn",
        Kept("background notices merged into this prompt"),
    ),
    (
        "joinedNoticeRunIds",
        Kept("background notices merged into this prompt"),
    ),
    (
        "scheduledFireId",
        Kept("the scheduled task run that sent this prompt; no reader yet"),
    ),
    (
        "scheduledTaskId",
        Kept("the scheduled task that sent this prompt; no reader yet"),
    ),
    ("imagePasteIds", Kept("paste-cache IDs of pasted images")),
    (
        "usageLimitNote",
        Kept("usage-limit notice text; quota state is read from quotaLimits"),
    ),
    (
        "userFeedback",
        Kept("feedback the operator gave on a response; no reader yet"),
    ),
    (
        "stackedExpansion",
        Kept("prompt-stack expansion; no reader yet"),
    ),
    (
        "stackedOriginalInput",
        Kept("prompt-stack input before expansion"),
    ),
    (
        "apiError",
        Kept("API error detail; its kind and status are read"),
    ),
    (
        "errorDetails",
        Kept("API error detail; its kind and status are read"),
    ),
    (
        "apiErrorIsTransient",
        Kept("API error detail; its kind and status are read"),
    ),
    (
        "apiErrorParams",
        Kept("API error detail; its kind and status are read"),
    ),
    (
        "isAbortedMidStream",
        Kept("the response stream stopped early; no stop reason marks it incomplete"),
    ),
    (
        "session_id",
        Excluded("SDK copy of sessionId; equal in all but 1 of 18,210 sampled records"),
    ),
    (
        "sourceToolAssistantUUID",
        Excluded("names the message holding the tool call; results pair to calls by call ID"),
    ),
    (
        "apiBlockIndex",
        Excluded("position of the block in the API response; record order gives it"),
    ),
    (
        "thinkingDisplay",
        Excluded("how the tool's UI shows thinking"),
    ),
    (
        "queueSkipAttachments",
        Excluded("UI hint for a queued prompt"),
    ),
    (
        "wireToolInputs",
        Kept("tool inputs as sent on the wire, by call ID; no reader yet"),
    ),
    (
        "wireIngestContext",
        Kept("context the harness attached to tool inputs, by call ID; no reader yet"),
    ),
    (
        "serverClassifierRequest",
        Kept("auto-mode permission classifier request; the decision is read"),
    ),
    (
        "serverClassifierContext",
        Kept("auto-mode permission classifier input; the decision is read"),
    ),
    (
        "classifierMetaLines",
        Kept("auto-mode permission classifier input; the decision is read"),
    ),
    (
        "thinkingMetadata",
        Kept("thinking budget and triggers the prompt set; no reader yet"),
    ),
    (
        "turnPosition",
        Kept("prompt and turn index; record order gives the same"),
    ),
    (
        "todos",
        Kept("the todo list as of this prompt; the todo side files are read"),
    ),
    ("message", Read(T::Object)),
    ("message.role", Read(T::String)),
    ("message.id", Read(T::String)),
    ("message.model", Read(T::String)),
    ("message.usage", Read(T::Object)),
    ("message.usage.input_tokens", Read(T::Number)),
    ("message.usage.output_tokens", Read(T::Number)),
    ("message.usage.thinking_tokens", Read(T::Number)),
    ("message.usage.cache_read_input_tokens", Read(T::Number)),
    ("message.usage.cache_creation_input_tokens", Read(T::Number)),
    (
        "message.usage.cache_creation.ephemeral_1h_input_tokens",
        Read(T::Number),
    ),
    ("message.usage.service_tier", Read(T::String)),
    (
        "message.usage.speed",
        Kept("speed as reported; numeric throughput differs from a named generation mode"),
    ),
    (
        "message.usage.inference_geo",
        Kept("provider inference region"),
    ),
    (
        "message.usage.cache_creation.ephemeral_5m_input_tokens",
        Kept("provider cache lifetime breakdown"),
    ),
    ("message.usage.server_tool_use", Read(T::Object)),
    (
        "message.usage.server_tool_use.web_search_requests",
        Read(T::Number),
    ),
    (
        "message.usage.server_tool_use.web_fetch_requests",
        Read(T::Number),
    ),
    ("message.usage.iterations", Read(T::Number)),
    (
        "message.usage.output_tokens_details.thinking_tokens",
        Read(T::Number),
    ),
    ("message.stop_reason", Read(T::String)),
    (
        "message.stop_details",
        Kept("complete refusal details in production"),
    ),
    (
        "message.diagnostics.cache_miss_reason",
        Kept("cache diagnostics in production"),
    ),
    ("message.ttftMs", Read(T::Number)),
    (
        "message.context_management",
        Kept("context edits the API applied (cleared tool results); no reader yet"),
    ),
    (
        "message.input_transformations",
        Kept("how the API transformed the input; no reader yet"),
    ),
    (
        "message.container",
        Excluded("code-execution container; null in observed data"),
    ),
    (
        "message.stop_sequence",
        Excluded("the matched text; stop_reason is read"),
    ),
    ("message.type", Excluded("always \"message\"")),
    ("message.content", Read(T::StringOrArray)),
    ("message.content[].type", Read(T::String)),
    ("message.content[].text", Read(T::String)),
    ("message.content[].thinking", Read(T::String)),
    ("message.content[].signature", Read(T::String)),
    ("message.content[].data", Read(T::String)),
    ("message.content[].id", Read(T::String)),
    ("message.content[].name", Read(T::String)),
    (
        "message.content[].input",
        Kept("call input retained in the tool block"),
    ),
    ("message.content[].tool_use_id", Read(T::String)),
    (
        "message.content[].content",
        Kept("every result part retained in structured result data"),
    ),
    ("message.content[].is_error", Read(T::Boolean)),
    ("message.content[].source", Read(T::Object)),
    ("message.content[].source.type", Read(T::String)),
    ("message.content[].source.media_type", Read(T::String)),
    ("message.content[].source.data", Read(T::String)),
    ("message.content[].source.url", Read(T::String)),
    ("message.content[].toolDenialKind", Read(T::String)),
    (
        "message.content[].from",
        Kept("observed 3 times; no reader yet"),
    ),
    (
        "message.content[].to",
        Kept("observed 3 times; no reader yet"),
    ),
    (
        "message.content[].caller",
        Excluded("who issued the tool call; always \"direct\" in observed data"),
    ),
    ("message.content[].backgroundTaskId", Read(T::String)),
    (
        "message.content[].cache_control",
        Excluded("prompt-cache hint sent to the API"),
    ),
    (
        "message.content[].citations",
        Kept("per-block citations in message attrs"),
    ),
])
.shape(
    &["wireToolInputs", "wireIngestContext"],
    &[
        crate::fields::KnownKinds {
            path: "type",
            values: &["assistant", "user"],
        },
        crate::fields::KnownKinds {
            path: "message.content[].type",
            values: &[
                "text",
                "thinking",
                "redacted_thinking",
                "tool_use",
                "server_tool_use",
                "tool_result",
                "image",
                "document",
            ],
        },
    ],
);

pub static TITLE_FIELDS: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("sessionId", Read(T::String)),
    ("agentId", Read(T::String)),
    ("summary", Read(T::String)),
    ("customTitle", Read(T::String)),
    ("aiTitle", Read(T::String)),
    ("cwd", Read(T::String)),
    ("timestamp", Read(T::StringOrNumber)),
]);
pub static CONTINUATION_FIELDS: Fields = Fields::new(&[
    ("type", Read(T::String)),
    ("continuedInSessionId", Read(T::String)),
]);

pub static AGENT_PROGRESS_FIELDS: Fields = Fields::new(&[
    ("", Kept("complete progress record in the event")),
    ("type", Read(T::String)),
    ("sessionId", Read(T::String)),
    ("data.type", Read(T::String)),
    ("data.agentId", Read(T::String)),
]);
