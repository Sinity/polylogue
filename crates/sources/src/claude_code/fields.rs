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
    ("forkedFrom", Read),
    ("sessionKind", Read),
    ("effort", Read),
    ("perTurnEffort", Read),
    ("requestedModel", Read),
    ("advisorModel", Read),
    ("attributionAgent", Read),
    ("attributionSkill", Read),
    ("attributionPlugin", Read),
    ("attributionMcpServer", Read),
    ("attributionMcpTool", Read),
    ("thinkingDurationMs", Read),
    ("error", Read),
    ("apiErrorStatus", Read),
    ("permissionDecision", Read),
    ("requestRef", Read),
    ("turnOrigin", Read),
    ("quotaLimits", Read),
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
        Kept("companion text the harness showed; no reader yet"),
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
    ("message", Read),
    ("message.role", Read),
    ("message.id", Read),
    ("message.model", Read),
    ("message.usage", Read),
    ("message.stop_reason", Read),
    ("message.stop_details", Read),
    ("message.diagnostics", Read),
    ("message.ttftMs", Read),
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
