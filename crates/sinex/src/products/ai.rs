//! AI facts exposed to Sinex readers independently of the local row transport.
use crate::kinds::{GitIdentity, Objective, RecordRef, TerminalIdentity};
use polylogue_model::{
    Block, MessageId, MessageRow, Production, PromptKind, Role, Session, SessionId, SessionKind,
    TimeEvidence, Timestamp, Usage, WrittenBy,
};
use schemars::JsonSchema;
use serde::Serialize;

#[derive(Serialize, JsonSchema)]
pub(super) struct MessageFields<'a> {
    message: &'a MessageId,
    session: &'a SessionId,
    parent: &'a Option<MessageId>,
    role: Role,
    written_by: WrittenBy,
    prompt_kind: Option<PromptKind>,
    at: Option<Timestamp>,
    at_evidence: TimeEvidence,
    model: &'a Option<String>,
    text: String,
    blocks: &'a [Block],
    usage: &'a Usage,
    stop_reason: &'a Option<String>,
    reported_cost_usd: &'a Option<serde_json::Number>,
    duration_ms: Option<u64>,
    production: &'a Production,
    attrs: &'a serde_json::Map<String, serde_json::Value>,
    source_start: u64,
    source_end: u64,
    id_kind: polylogue_model::IdKind,
    page_text: bool,
    order: Option<u64>,
    projection: &'a Option<polylogue_model::bundle::MessageProjection>,
    field_claims: &'a Option<Vec<polylogue_model::HookFieldClaim>>,
    injections: &'a Option<Vec<polylogue_model::injection::InjectionEvidence>>,
    attachment_bindings: &'a Option<Vec<polylogue_model::attachment_binding::AttachmentBinding>>,
}

impl<'a> MessageFields<'a> {
    pub(super) fn new(
        row: &'a MessageRow,
        order: Option<u64>,
        projection: &'a Option<polylogue_model::bundle::MessageProjection>,
        field_claims: &'a Option<Vec<polylogue_model::HookFieldClaim>>,
        injections: &'a Option<Vec<polylogue_model::injection::InjectionEvidence>>,
        attachment_bindings: &'a Option<
            Vec<polylogue_model::attachment_binding::AttachmentBinding>,
        >,
    ) -> Self {
        let message = &row.message;
        Self {
            message: &message.id,
            session: &message.session,
            parent: &message.parent,
            role: message.role,
            written_by: WrittenBy::of(message),
            prompt_kind: message.prompt_kind,
            at: message.at,
            at_evidence: message.time_evidence(),
            model: &message.model,
            text: message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n\n"),
            blocks: &message.blocks,
            usage: &message.usage,
            stop_reason: &message.stop_reason,
            reported_cost_usd: &message.reported_cost_usd,
            duration_ms: message.duration_ms,
            production: &message.production,
            attrs: &message.attrs,
            source_start: row.offset,
            source_end: row.end,
            id_kind: row.id_kind,
            page_text: row.page_text,
            order,
            projection,
            field_claims,
            injections,
            attachment_bindings,
        }
    }
}

#[derive(Serialize, JsonSchema)]
pub(super) struct SessionFields<'a> {
    session: &'a SessionId,
    origin: &'a str,
    kind: SessionKind,
    title: Option<&'a str>,
    title_source: Option<polylogue_model::TitleSource>,
    aliases: &'a [SessionId],
    cwd: &'a [String],
    repo: &'a Option<String>,
    branch: &'a Option<String>,
    git_commit: &'a Option<String>,
    model_provider: &'a Option<String>,
    instructions: &'a Option<String>,
    tool_version: &'a Option<String>,
    mode: &'a Option<String>,
    started: Option<Timestamp>,
    started_evidence: TimeEvidence,
    machine: Option<&'a str>,
    objectives: Vec<Objective>,
    git_identity: Option<GitIdentity>,
    terminal: Option<TerminalIdentity>,
    boot_id: Option<String>,
    native_sources: Vec<RecordRef>,
    native_unavailable: Vec<String>,
}

impl<'a> From<&'a Session> for SessionFields<'a> {
    fn from(session: &'a Session) -> Self {
        Self {
            session: &session.id,
            origin: session.id.origin().as_str(),
            kind: session.kind,
            title: session.title.as_ref().map(|title| title.text.as_str()),
            title_source: session.title.as_ref().map(|title| title.source),
            aliases: &session.aliases,
            cwd: &session.cwd,
            repo: &session.repo,
            branch: &session.git_branch,
            git_commit: &session.git_commit,
            model_provider: &session.model_provider,
            instructions: &session.instructions,
            tool_version: &session.tool_version,
            mode: &session.mode,
            started: session.started,
            started_evidence: session.time_evidence(),
            machine: None,
            objectives: Vec::new(),
            git_identity: None,
            terminal: None,
            boot_id: None,
            native_sources: Vec::new(),
            native_unavailable: Vec::new(),
        }
    }
}

pub(super) fn schema(product: &str) -> Option<serde_json::Value> {
    let (mut schema, renamed) = match product {
        "message" => (
            serde_json::to_value(schemars::schema_for!(MessageFields)).unwrap(),
            None,
        ),
        "session" => (
            serde_json::to_value(schemars::schema_for!(SessionFields)).unwrap(),
            None,
        ),
        "tool_call" => (
            serde_json::to_value(schemars::schema_for!(ToolFields)).unwrap(),
            None,
        ),
        "document" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::Document)).unwrap(),
            Some(("id", "document")),
        ),
        "fragment" => (
            serde_json::to_value(schemars::schema_for!(
                polylogue_model::bundle::fragments::TextFragmentRecord
            ))
            .unwrap(),
            None,
        ),
        "event" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::Event)).unwrap(),
            Some(("kind", "event")),
        ),
        "link" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::Link)).unwrap(),
            Some(("child", "session")),
        ),
        "prefix_copy" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::PrefixCopy)).unwrap(),
            Some(("child", "session")),
        ),
        "prefix_member" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::PrefixMember)).unwrap(),
            Some(("child", "session")),
        ),
        "background" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::BackgroundUpdate)).unwrap(),
            Some(("task_id", "task")),
        ),
        "repository" => (
            serde_json::to_value(schemars::schema_for!(polylogue_model::RepoLookup)).unwrap(),
            None,
        ),
        _ => return None,
    };
    if let Some((from, to)) = renamed {
        rename_schema_field(&mut schema, from, to);
    }
    if product == "link" {
        rename_schema_field(&mut schema, "at", "branch_point");
    }
    if matches!(product, "document" | "event") {
        let mut evidence = serde_json::to_value(schemars::schema_for!(TimeEvidence)).unwrap();
        if let Some(definitions) = evidence
            .as_object_mut()
            .expect("time evidence schema")
            .remove("$defs")
        {
            let fields = schema
                .as_object_mut()
                .expect("domain schema")
                .entry("$defs")
                .or_insert_with(|| serde_json::json!({}))
                .as_object_mut()
                .expect("domain definitions");
            for (name, definition) in definitions.as_object().expect("time evidence definitions") {
                if let Some(existing) = fields.get(name) {
                    assert_eq!(existing, definition);
                } else {
                    fields.insert(name.clone(), definition.clone());
                }
            }
        }
        schema["properties"]["at_evidence"] = evidence;
        schema["required"]
            .as_array_mut()
            .expect("required domain fields")
            .push("at_evidence".into());
    }
    Some(schema)
}

fn rename_schema_field(schema: &mut serde_json::Value, from: &str, to: &str) {
    let properties = schema["properties"]
        .as_object_mut()
        .expect("domain object fields");
    let field = properties.remove(from).expect("domain identity field");
    properties.insert(to.into(), field);
    if let Some(required) = schema["required"].as_array_mut() {
        for field in required {
            if field.as_str() == Some(from) {
                *field = to.into();
            }
        }
    }
}

#[derive(Serialize, JsonSchema)]
pub(super) struct ToolFields<'a> {
    #[serde(flatten)]
    pub call: &'a polylogue_store::primary::products::ToolCall,
    pub context: crate::native::Facts,
}
