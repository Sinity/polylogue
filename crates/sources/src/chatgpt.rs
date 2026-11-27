//! ChatGPT conversations: the export's `conversations.json` (newer exports
//! shard it as `conversations-NNN.json`), and the same conversation JSON a
//! browser capture carries (see `browser_capture`).
//!
//! A conversation is a tree (`mapping`): an edited prompt or a regenerated
//! answer is a sibling branch. Every branch is kept, each message pointing
//! at its parent; `current_node` names the leaf the operator last saw and
//! becomes an `active_leaf` event. Messages are ordered by their creation
//! time, then by their place in the tree (a walk through `children`).
//!
//! Tool use is spread over messages: an assistant message addressed to a
//! tool (`recipient` other than `all`) is the call; the `role: tool`
//! messages after it are its results. Later results of one call hang off
//! the earlier result, so a result's call is found by skipping tool
//! ancestors. Images and files the export stores
//! separately are referred to by their asset pointer and file ID.

use crate::document::first_byte;
use polylogue_model::JsonExt;
/// Version of the rows this module produces.
pub const VERSION: u32 = 4;

parser_modules! {}

use crate::fields::ExpectedType as T;
pub mod assets;

use crate::RecordOutcome;
use crate::document::{Sink, message_id, session_id, time_of};
use crate::fields::{Fields, Rule};
use crate::parse::{MessageRow, Parse};
use polylogue_model::{
    Attachment, AttachmentDirection, Block, ErrorReason, Event, Json, Link, LinkKind, Message,
    PromptKind, Role, Session, SessionKind, TitleSource, ToolAction, ToolOutcome, UnknownReason,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub const ORIGIN: &str = "chatgpt";

/// `conversations.json` or a shard `conversations-NNN.json`, holding an
/// array of ChatGPT conversations. claude.ai's export uses the same name;
/// its conversations start with `uuid`, ChatGPT's never do.
///
/// One conversation alone, in the conversation API's shape (`mapping`,
/// `current_node`, `conversation_id`), is recognized whatever the file is
/// named, once the whole document is read.
pub fn recognize(path: &Path, head: &crate::document::Head<'_>) -> bool {
    use crate::document::Kind;
    if head.shape.is_some()
        && head.holds("mapping", Kind::Object)
        && head.has("current_node")
        && head.holds("conversation_id", Kind::String)
    {
        return true;
    }
    let head = head.bytes;
    if !recognize_name(path) || first_byte(head) != Some(b'[') {
        return false;
    }
    match crate::document::first_key(head) {
        Some(key) => CONVERSATION_KEYS.contains(&key),
        // An empty array, or a first key past the bytes read.
        None => true,
    }
}

/// Keys a ChatGPT conversation object starts with, across export versions.
const CONVERSATION_KEYS: &[&str] = &[
    "title",
    "conversation_id",
    "id",
    "create_time",
    "update_time",
    "mapping",
    "current_node",
    "gizmo_id",
    "conversation_template_id",
];

pub fn open(path: &Path) -> Option<Parse> {
    path.extension()
        .is_some_and(|ext| ext == "json")
        .then(|| Parse::Export(Box::new(Export)))
}

fn recognize_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with("conversations") && n.ends_with(".json"))
}

/// The export's conversation array.
pub struct Export;

impl crate::export::ExportParser for Export {
    fn scan(
        &mut self,
        _: &str,
        reader: &mut dyn crate::export::Input,
        item: &mut dyn crate::export::ItemSink,
    ) -> Result<crate::export::ScanStats, crate::export::ScanError> {
        crate::export::scan_array_or_object(reader, |range, raw| {
            item.push(range, self.order_key(raw), None, raw.get().as_bytes())
        })
    }

    fn order_key(&self, raw: &serde_json::value::RawValue) -> Vec<u8> {
        #[derive(serde::Deserialize)]
        struct Header {
            create_time: Option<Json>,
        }
        // Unrelated fields deserialize as IgnoredAny, including the mapping.
        let time = serde_json::from_str::<Header>(raw.get())
            .ok()
            .and_then(|h| h.create_time)
            .and_then(|v| v.as_f64());
        crate::export::number_order(time)
    }
    fn item(
        &mut self,
        bytes: &[u8],
        sink: &mut Sink<'_>,
        first_sessions: &mut dyn crate::export::ExportState,
    ) -> Result<(), String> {
        let value: Json = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        let value = &value;
        conversation_with_sessions(value, sink, first_sessions)
    }
}

/// A mapping node's own fields.
static NODE_FIELDS: Fields = Fields::new(
    "node",
    &[
        ("id", Rule::Read(T::String)),
        ("message", Rule::Read(T::Object)),
        ("parent", Rule::Read(T::String)),
        (
            "children",
            Rule::Excluded("the parent links carry the same tree"),
        ),
    ],
);

/// A message's fields. `metadata` holds dozens of provider keys (search
/// results, citations, task state); it is kept whole in the stored export
/// and read where it says something about the message (model, finish,
/// attachments, hidden prompts).
static MESSAGE_FIELDS: Fields = Fields::new(
    "message",
    &[
        ("id", Rule::Read(T::String)),
        ("author", Rule::Read(T::Object)),
        ("author.role", Rule::Read(T::String)),
        ("author.name", Rule::Read(T::String)),
        ("author.metadata", Rule::Kept("tool display details")),
        ("create_time", Rule::Read(T::StringOrNumber)),
        (
            "update_time",
            Rule::Excluded("edit time; the creation time orders messages"),
        ),
        ("content", Rule::Read(T::Object)),
        ("content.content_type", Rule::Read(T::String)),
        ("content.parts[]", Rule::Read(T::StringOrObject)),
        ("content.parts[].content_type", Rule::Read(T::String)),
        ("content.parts[].asset_pointer", Rule::Read(T::String)),
        ("content.parts[].mime_type", Rule::Read(T::String)),
        ("content.parts[].size_bytes", Rule::Read(T::Number)),
        ("content.parts[].text", Rule::Read(T::String)),
        ("content.text", Rule::Read(T::String)),
        ("content.url", Rule::Read(T::String)),
        ("content.title", Rule::Read(T::String)),
        ("content.domain", Rule::Read(T::String)),
        ("content.snippet", Rule::Read(T::String)),
        ("content.result", Rule::Read(T::String)),
        ("content.tether_id", Rule::Read(T::String)),
        ("content.ref_id", Rule::Read(T::String)),
        ("content.language", Rule::Read(T::String)),
        ("content.thoughts", Rule::Read(T::Array)),
        ("content.thoughts[].summary", Rule::Read(T::String)),
        ("content.thoughts[].content", Rule::Read(T::String)),
        ("content.model_set_context", Rule::Read(T::String)),
        ("content.user_profile", Rule::Read(T::String)),
        ("content.user_instructions", Rule::Read(T::String)),
        ("status", Rule::Read(T::String)),
        (
            "end_turn",
            Rule::Excluded("turn boundaries follow from the roles"),
        ),
        ("weight", Rule::Kept("message attrs weight")),
        ("metadata", Rule::Kept("provider metadata in message attrs")),
        ("metadata.async_task_id", Rule::Read(T::String)),
        ("metadata.async_task_title", Rule::Read(T::String)),
        ("metadata.async_task_type", Rule::Read(T::String)),
        ("recipient", Rule::Read(T::String)),
        ("channel", Rule::Kept("message attrs channel")),
    ],
)
.shape(
    &[],
    &[
        crate::fields::KnownKinds {
            path: "author.role",
            values: &["user", "assistant", "system", "tool"],
        },
        crate::fields::KnownKinds {
            path: "content.content_type",
            values: &[
                "text",
                "multimodal_text",
                "thoughts",
                "reasoning_recap",
                "user_editable_context",
                "code",
                "execution_output",
                "tether_quote",
                "tether_browsing_display",
                "sonic_webpage",
            ],
        },
    ],
);

static CONVERSATION_FIELDS: Fields = Fields::new(
    "conversation",
    &[
        ("", Rule::Kept("conversation settings event")),
        ("id", Rule::Read(T::String)),
        ("conversation_id", Rule::Read(T::String)),
        ("title", Rule::Read(T::String)),
        ("gizmo_id", Rule::Read(T::String)),
        ("conversation_template_id", Rule::Read(T::String)),
        ("create_time", Rule::Read(T::StringOrNumber)),
        ("update_time", Rule::Read(T::StringOrNumber)),
        ("mapping", Rule::Read(T::Object)),
    ],
);

/// One conversation object: its session, messages and events. Shared by
/// the export and by browser captures of the same conversation, so both
/// arrivals produce the same IDs and unite.
pub fn conversation(
    conv: &Json,
    sink: &mut Sink<'_>,
    first_sessions: &mut crate::export::MemoryState,
) {
    conversation_with_sessions(conv, sink, first_sessions).expect("in-memory session state");
}

fn conversation_with_sessions(
    conv: &Json,
    sink: &mut Sink<'_>,
    first_sessions: &mut dyn crate::export::ExportState,
) -> Result<(), String> {
    let Some(obj) = conv.as_object() else {
        sink.records
            .record("conversation", &RecordOutcome::unreadable("not an object"));
        return Ok(());
    };
    if obj.is_empty() {
        sink.records.record(
            "conversation",
            &RecordOutcome::skipped("empty object in the export"),
        );
        return Ok(());
    }
    let Some(native) = conv
        .str("conversation_id")
        .filter(|s| !s.is_empty())
        .or_else(|| conv.str("id").filter(|s| !s.is_empty()))
    else {
        sink.records.record(
            "conversation",
            &RecordOutcome::unreadable("no conversation ID"),
        );
        return Ok(());
    };
    let Some(sid) = session_id(ORIGIN, native) else {
        sink.records.record(
            "conversation",
            &RecordOutcome::unreadable("unusable conversation ID"),
        );
        return Ok(());
    };
    let Some(mapping) = conv.object("mapping") else {
        sink.records
            .record("conversation", &RecordOutcome::unreadable("no mapping"));
        return Ok(());
    };

    let mut session = Session::new(sid.clone(), SessionKind::Main);
    if let Some(title) = conv.str("title").filter(|s| !s.is_empty()) {
        session.offer_title(title, TitleSource::Generated);
    }
    if let Some(at) = conv.get("create_time").and_then(time_of) {
        session
            .offer_start(at.timestamp(), at.accuracy())
            .expect("decoded source time is valid");
    }
    sink.out.sessions.push(session);
    sink.records.record("conversation", &RecordOutcome::Used);

    // The conversation's own settings (custom GPT, archive and star state,
    // memory scope) are kept whole as one event.
    let settings: serde_json::Map<String, Json> = obj
        .iter()
        .filter(|(k, _)| k.as_str() != "mapping")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    for (kind, data) in [
        ("conversation_settings", Json::Object(settings)),
        ("conversation_facts", conversation_facts(conv, native)),
    ] {
        let (at, time_accuracy) = crate::time::parts(conv.get("update_time").and_then(time_of));
        sink.out.events.push(Event {
            session: Some(sid.clone()),
            kind: kind.into(),
            at,
            at_quality: time_accuracy.quality,
            at_precision: time_accuracy.precision,
            at_until: time_accuracy.until,
            after: None,
            range: None,
            data,
        });
    }

    sink.records.fields(
        "conversation",
        &crate::fields::header(conv, &["mapping"]),
        &CONVERSATION_FIELDS,
    );
    let tree = Tree::new(mapping);
    let places = tree.places();
    let default_model = conv
        .str("default_model_slug")
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let mut rows: Vec<(i128, usize, Message)> = Vec::new();
    let mut nodes: Vec<(&String, &Json)> = mapping.iter().collect();
    nodes.sort_by_key(|(id, _)| places.get(id.as_str()).copied());
    for (index, (node_id, node)) in nodes.into_iter().enumerate() {
        sink.records.fields(
            "node",
            &crate::fields::header(node, &["message"]),
            &NODE_FIELDS,
        );
        let Some(msg) = node.get("message").filter(|m| !m.is_null()) else {
            continue; // The root and other empty nodes hold no record.
        };
        let kind = msg
            .pointer("/content/content_type")
            .and_then(Json::as_str)
            .unwrap_or("?")
            .to_owned();
        sink.records.fields(&kind, msg, &MESSAGE_FIELDS);
        match message(&sid, node_id, msg, &tree, default_model.as_deref()) {
            Ok(m) => {
                // The same reading as the stored time, so a string time
                // orders a message where it is stored.
                let time = msg
                    .get("create_time")
                    .and_then(time_of)
                    .map_or(i128::MIN, |time| time.unix_ns());
                rows.push((time, index, m));
                sink.records.record(&kind, &RecordOutcome::Used);
            }
            Err(d) => sink.records.record(&kind, &d),
        }
    }
    // A message without a time keeps its place in the tree among its
    // neighbours: it takes the time of the message before it.
    let mut last = i128::MIN;
    for row in &mut rows {
        if row.0 == i128::MIN {
            row.0 = last;
        } else {
            last = row.0;
        }
    }
    rows.sort_by_key(|row| (row.0, row.1));
    // Messages an earlier conversation holds are that conversation's: this
    // one is a branch of it. It keeps only its own tail and a fork link at
    // the last message they share.
    let mut branch: Option<(polylogue_model::SessionId, Message)> = None;
    for (position, (_, _, message)) in rows.into_iter().enumerate() {
        let native = message.id.native().as_str().to_owned();
        match first_sessions.claim_native_id(&native, &sid)? {
            Some(first_session) if first_session != sid => {
                if !sink
                    .out
                    .prefix_copies
                    .iter()
                    .any(|copy| copy.parent == first_session && copy.child == sid)
                {
                    sink.out.prefix_copies.push(polylogue_model::PrefixCopy {
                        parent: first_session.clone(),
                        child: sid.clone(),
                        explicit_members: true,
                    });
                }
                sink.out.prefix_members.push(polylogue_model::PrefixMember {
                    parent: first_session.clone(),
                    child: sid.clone(),
                    message: message.id.clone(),
                    source: None,
                    position: [0, position as u64],
                });
                branch = Some((first_session, message));
            }
            _ => {
                let offset = sink.position();
                sink.out.messages.push(MessageRow {
                    response: None,
                    page_text: false,
                    id_kind: polylogue_model::IdKind::Native,
                    offset,
                    end: offset + 1,
                    message,
                });
            }
        }
    }
    if let Some((parent, at)) = branch {
        sink.out.links.push(Link {
            kind: LinkKind::Fork,
            copies_prefix: true,
            reason: None,
            parent,
            child: sid.clone(),
            branch_point: Some(at.id),
            call_id: None,
            evidence: "message IDs repeated from the parent conversation".into(),
        });
    }

    if let Some(leaf) = conv
        .str("current_node")
        .filter(|s| !s.is_empty())
        .and_then(|n| tree.nearest_message(n))
        && let Some(id) = message_id(ORIGIN, &leaf)
    {
        sink.out.events.push({
            let (at, time_accuracy) = crate::time::parts(None);
            Event {
                session: Some(sid),
                kind: "active_leaf".into(),
                at,
                at_quality: time_accuracy.quality,
                at_precision: time_accuracy.precision,
                at_until: time_accuracy.until,
                after: Some(id),
                range: None,
                data: Json::Null,
            }
        });
    }
    Ok(())
}

/// The mapping as a tree of node IDs, for parents and the calls that tool results answer.
struct Tree<'a> {
    nodes: &'a serde_json::Map<String, Json>,
}

impl<'a> Tree<'a> {
    fn new(nodes: &'a serde_json::Map<String, Json>) -> Self {
        Self { nodes }
    }

    fn parent(&self, node: &str) -> Option<&'a str> {
        self.nodes.get(node)?.str("parent")
    }

    /// Each node's place in a walk from the roots through every node's
    /// `children`, in their listed order: the tree's own order, which no
    /// JSON object's key order changes. Roots, and nodes no walk reaches,
    /// follow by ID.
    fn places(&self) -> HashMap<&'a str, usize> {
        let mut ids: Vec<&'a str> = self.nodes.keys().map(String::as_str).collect();
        ids.sort_unstable();
        let mut place = HashMap::new();
        for root in ids
            .iter()
            .copied()
            .filter(|n| self.parent(n).is_none_or(|p| !self.nodes.contains_key(p)))
        {
            let mut stack = vec![root];
            while let Some(n) = stack.pop() {
                if place.contains_key(n) {
                    continue;
                }
                place.insert(n, place.len());
                if let Some(children) = self.nodes.get(n).and_then(|v| v.array("children")) {
                    stack.extend(
                        children
                            .iter()
                            .rev()
                            .filter_map(Json::as_str)
                            .filter(|c| self.nodes.contains_key(*c)),
                    );
                }
            }
        }
        for n in ids {
            let next = place.len();
            place.entry(n).or_insert(next);
        }
        place
    }

    /// The message ID a node's message has: its own `id`, else the node's.
    fn message_native(&self, node: &str) -> Option<String> {
        let n = self.nodes.get(node)?;
        let msg = n.get("message").filter(|m| !m.is_null())?;
        Some(
            msg.str("id")
                .filter(|s| !s.is_empty())
                .or_else(|| n.str("id").filter(|s| !s.is_empty()))
                .unwrap_or(node)
                .to_owned(),
        )
    }

    fn author_role(&self, node: &str) -> Option<&'a str> {
        self.nodes
            .get(node)?
            .pointer("/message/author/role")?
            .as_str()
    }

    /// The nearest node at or above `node` that holds a message.
    fn nearest_message(&self, node: &str) -> Option<String> {
        let mut seen = HashSet::new();
        let mut at = Some(node);
        while let Some(n) = at {
            if !seen.insert(n) {
                return None;
            }
            if let Some(id) = self.message_native(n) {
                return Some(id);
            }
            at = self.parent(n);
        }
        None
    }

    /// The message a node's parent link points at, skipping empty nodes.
    fn parent_message(&self, node: &str) -> Option<String> {
        self.nearest_message(self.parent(node)?)
    }

    /// The call a `role: tool` node answers: the first ancestor that is
    /// not itself a tool message. Several results of one call chain off
    /// each other, and all of them answer the same call.
    fn answered_call(&self, node: &str) -> Option<String> {
        let mut seen = HashSet::new();
        let mut at = self.parent(node);
        while let Some(n) = at {
            if !seen.insert(n) {
                return None;
            }
            if self.message_native(n).is_some() && self.author_role(n) != Some("tool") {
                return self.message_native(n);
            }
            at = self.parent(n);
        }
        None
    }
}

fn message(
    sid: &polylogue_model::SessionId,
    node_id: &str,
    msg: &Json,
    tree: &Tree<'_>,
    default_model: Option<&str>,
) -> Result<Message, RecordOutcome> {
    let content = msg
        .get("content")
        .filter(|c| c.is_object())
        .ok_or_else(|| RecordOutcome::unreadable("no content object"))?;
    let role_name = msg
        .pointer("/author/role")
        .and_then(Json::as_str)
        .ok_or_else(|| RecordOutcome::unreadable("no author role"))?;
    let native = msg.str("id").filter(|s| !s.is_empty()).unwrap_or(node_id);
    let id = message_id(ORIGIN, native).ok_or_else(|| RecordOutcome::unreadable("unusable ID"))?;
    let meta = msg.get("metadata").unwrap_or(&Json::Null);
    let recipient = msg
        .str("recipient")
        .filter(|s| !s.is_empty())
        .filter(|r| *r != "all");

    let (role, prompt_kind) = match role_name {
        "user" => {
            let hidden = meta.bool("is_visually_hidden_from_conversation") == Some(true)
                || meta.bool("is_user_system_message") == Some(true);
            (
                Role::User,
                Some(if hidden {
                    PromptKind::Harness
                } else {
                    PromptKind::Prose
                }),
            )
        }
        "assistant" => (Role::Assistant, None),
        "system" => (Role::System, None),
        // Tool output is written by the harness in the conversation's
        // name, as in Claude Code: a user-side message of harness text.
        "tool" => (Role::User, Some(PromptKind::Harness)),
        other => return Err(RecordOutcome::skipped(format!("author role {other}"))),
    };
    let direction = if role == Role::Assistant || role_name == "tool" {
        AttachmentDirection::Output
    } else {
        AttachmentDirection::Input
    };

    let mut blocks = if role_name == "tool" {
        let call_id = tree.answered_call(node_id);
        vec![tool_result(content, msg, meta, call_id, direction)]
    } else if let Some(tool) = recipient {
        vec![tool_call(native, tool, content)]
    } else {
        content_blocks(content, direction)
    };
    blocks.extend(metadata_attachments(meta, direction));
    if role_name == "tool" && blocks.len() > 1 {
        // Files a tool produced belong to its result.
        let extra: Vec<Block> = blocks.drain(1..).collect();
        if let Block::ToolResult { attachments, .. } = &mut blocks[0] {
            attachments.extend(extra.into_iter().filter_map(|b| match b {
                Block::Image(a) | Block::File(a) => Some(a),
                _ => None,
            }));
        }
    }

    for key in [
        "citations",
        "content_references",
        "search_result_groups",
        "canvas",
        "search_queries",
        "selected_sources",
        "image_results",
        "async_tasks",
        "_cite_metadata",
    ] {
        if let Some(data) = meta.get(key).filter(|value| {
            !value.is_null()
                && !value.as_array().is_some_and(Vec::is_empty)
                && !value.as_object().is_some_and(serde_json::Map::is_empty)
        }) {
            blocks.push(Block::Other {
                type_name: key.into(),
                data: data.clone(),
            });
        }
    }

    if let Some(task) = research_task(meta) {
        blocks.push(task);
    }

    // The model that answered, else the message's default, else the
    // conversation's.
    let model = meta
        .str("model_slug")
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .or_else(|| {
            (role == Role::Assistant && recipient.is_none())
                .then(|| {
                    meta.str("default_model_slug")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                        .or(default_model.map(str::to_owned))
                })
                .flatten()
        });
    Ok({
        let (at, time_accuracy) = crate::time::parts(msg.get("create_time").and_then(time_of));
        Message {
            id,
            session: sid.clone(),
            parent: tree
                .parent_message(node_id)
                .and_then(|p| message_id(ORIGIN, &p)),
            role,
            at,
            at_quality: time_accuracy.quality,
            at_precision: time_accuracy.precision,
            at_until: time_accuracy.until,
            model,
            prompt_kind,
            blocks,
            usage: Default::default(),
            stop_reason: meta
                .pointer("/finish_details/type")
                .and_then(Json::as_str)
                .map(str::to_owned),
            reported_cost_usd: None,
            duration_ms: meta
                .get("finished_duration_sec")
                .and_then(Json::as_f64)
                .filter(|s| s.is_finite() && *s >= 0.0)
                .map(|s| (s * 1000.0).round())
                // u64::MAX rounds to 2^64 as f64, so that boundary is exclusive.
                .filter(|ms| ms.is_finite() && *ms < u64::MAX as f64)
                .map(|ms| ms as u64),
            attrs: message_attrs(msg),
            production: production(meta),
        }
    })
}

fn conversation_facts(conv: &Json, native: &str) -> Json {
    let mut facts = serde_json::Map::new();
    let gizmo = conv.str("gizmo_id").filter(|value| !value.is_empty());
    let web_url = if let Some(gizmo) = gizmo {
        let kind = if gizmo.starts_with("g-p-") {
            "project"
        } else {
            "gpt"
        };
        facts.insert(kind.into(), Json::String(gizmo.into()));
        format!("https://chatgpt.com/g/{gizmo}/c/{native}")
    } else {
        format!("https://chatgpt.com/c/{native}")
    };
    facts.insert("web_url".into(), Json::String(web_url));
    if let Some(template) = conv
        .str("conversation_template_id")
        .filter(|value| !value.is_empty())
    {
        facts.insert("template".into(), Json::String(template.into()));
    }
    Json::Object(facts)
}

fn research_task(meta: &Json) -> Option<Block> {
    let mut data = serde_json::Map::new();
    for (name, key) in [
        ("id", "async_task_id"),
        ("title", "async_task_title"),
        ("kind", "async_task_type"),
    ] {
        if let Some(value) = meta.str(key).filter(|value| !value.is_empty()) {
            data.insert(name.into(), Json::String(value.into()));
        }
    }
    if data.is_empty() {
        return None;
    }
    Some(Block::Other {
        type_name: "research_task".into(),
        data: Json::Object(data),
    })
}

/// Browsing content uses the same fields in a message and in a tool result.
fn web_result(kind: &str, content: &Json) -> Option<Json> {
    if !matches!(
        kind,
        "tether_quote" | "tether_browsing_display" | "sonic_webpage"
    ) {
        return None;
    }
    let mut data = serde_json::Map::new();
    data.insert("content_type".into(), Json::String(kind.into()));
    for (name, keys) in [
        ("url", &["url"][..]),
        ("title", &["title", "domain"][..]),
        ("text", &["text", "snippet", "result"][..]),
        ("source_id", &["tether_id", "ref_id"][..]),
    ] {
        if let Some(value) = keys
            .iter()
            .find_map(|key| content.str(key).filter(|value| !value.is_empty()))
        {
            data.insert(name.into(), Json::String(value.into()));
        }
    }
    Some(Json::Object(data))
}

/// Provider metadata keeps the original nested references, groups and canvas
/// data, including fields added by the provider after this parser was written.
fn message_attrs(msg: &Json) -> serde_json::Map<String, Json> {
    let mut attrs = crate::fields::attributes(msg, &MESSAGE_FIELDS);
    for key in ["metadata", "channel", "weight"] {
        if let Some(value) = msg.get(key) {
            attrs.insert(key.into(), value.clone());
        }
    }
    attrs
}

/// The request settings ChatGPT recorded beside a response.
fn production(meta: &Json) -> polylogue_model::Production {
    let answered = meta.str("model_slug").filter(|s| !s.is_empty());
    polylogue_model::Production {
        effort: meta
            .str("thinking_effort")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        requested_model: meta
            .str("requested_model_slug")
            .filter(|s| !s.is_empty())
            .filter(|r| Some(*r) != answered)
            .map(str::to_owned),
        request: meta
            .str("request_id")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        ..Default::default()
    }
}

/// Text of a content object, for the content types that carry one.
fn content_text(content: &Json) -> Option<String> {
    if let Some(parts) = content.array("parts") {
        let text: Vec<&str> = parts.iter().filter_map(Json::as_str).collect();
        let joined = text.join("\n");
        return (!joined.is_empty()).then_some(joined);
    }
    for key in ["text", "result", "content"] {
        if let Some(t) = content.str(key).filter(|s| !s.is_empty()) {
            return Some(t.to_owned());
        }
    }
    None
}

fn content_type(content: &Json) -> &str {
    content.str("content_type").unwrap_or("")
}

/// The blocks of a message that is neither a tool call nor a result.
fn content_blocks(content: &Json, direction: AttachmentDirection) -> Vec<Block> {
    let kind = content_type(content);
    if let Some(data) = web_result(kind, content) {
        return vec![Block::Other {
            type_name: "web_result".into(),
            data,
        }];
    }
    let mut blocks = Vec::new();
    match kind {
        "text" => {
            if let Some(text) = content_text(content) {
                blocks.push(Block::Text { text });
            }
        }
        "multimodal_text" => {
            for part in content.array("parts").into_iter().flatten() {
                if let Some(text) = part.as_str() {
                    if !text.is_empty() {
                        blocks.push(Block::Text {
                            text: text.to_owned(),
                        });
                    }
                    continue;
                }
                blocks.push(part_block(part, direction));
            }
        }
        "thoughts" => {
            let thoughts = content.array("thoughts");
            if thoughts.is_none()
                && let Some(text) = content_text(content)
            {
                // Some exports write the reasoning as plain parts.
                blocks.push(Block::Thinking {
                    text: Some(text),
                    signature: None,
                    redacted: false,
                });
            }
            for thought in thoughts.into_iter().flatten() {
                let summary = thought.str("summary").filter(|s| !s.is_empty());
                let body = thought.str("content").filter(|s| !s.is_empty());
                let text = match (summary, body) {
                    (Some(s), Some(b)) => Some(format!("{s}\n\n{b}")),
                    (s, b) => s.or(b).map(str::to_owned),
                };
                blocks.push(Block::Thinking {
                    text,
                    signature: None,
                    redacted: false,
                });
            }
        }
        "reasoning_recap" => {
            if let Some(text) = content_text(content) {
                blocks.push(Block::Thinking {
                    text: Some(text),
                    signature: None,
                    redacted: false,
                });
            }
        }
        "user_editable_context" => {
            // Custom instructions, sent as a hidden system message.
            let parts: Vec<String> = ["user_profile", "user_instructions"]
                .iter()
                .filter_map(|k| content.str(k).filter(|s| !s.is_empty()).map(str::to_owned))
                .collect();
            if !parts.is_empty() {
                blocks.push(Block::Text {
                    text: parts.join("\n\n"),
                });
            }
        }
        other => blocks.push(Block::Other {
            type_name: other.to_owned(),
            data: content.clone(),
        }),
    }
    blocks
}

/// One non-text part of a multimodal message.
fn part_block(part: &Json, direction: AttachmentDirection) -> Block {
    let kind = part.str("content_type").unwrap_or("");
    match kind {
        // A media part whose pointer is missing names no file to attach.
        k if k.ends_with("asset_pointer") && part.get("asset_pointer").is_none() => Block::Other {
            type_name: kind.to_owned(),
            data: part.clone(),
        },
        "image_asset_pointer" => Block::Image(asset(part, direction)),
        "audio_transcription" => match part
            .str("text")
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        {
            Some(text) => Block::Text { text },
            None => Block::Other {
                type_name: kind.to_owned(),
                data: part.clone(),
            },
        },
        k if k.ends_with("asset_pointer") => Block::File(asset(part, direction)),
        _ => Block::Other {
            type_name: kind.to_owned(),
            data: part.clone(),
        },
    }
}

/// A file the export stores apart, by its pointer (`file-service://…`,
/// `sediment://…`). The pointer's last segment is the file ID the
/// export's asset files are named by.
fn asset(part: &Json, direction: AttachmentDirection) -> Attachment {
    let pointer = part
        .str("asset_pointer")
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    let source_id = pointer
        .as_deref()
        .and_then(|p| p.rsplit('/').next())
        .map(str::to_owned);
    Attachment {
        name: None,
        media_type: part
            .str("mime_type")
            .filter(|s| !s.is_empty())
            .map(str::to_owned),
        size: part.u64("size_bytes"),
        blob: None,
        url: pointer,
        source_id,
        direction,
    }
}

/// Files listed in a message's metadata: uploads, and charts the code tool
/// made (`ada_visualizations`).
fn metadata_attachments(meta: &Json, direction: AttachmentDirection) -> Vec<Block> {
    let mut out = Vec::new();
    for a in meta.array("attachments").into_iter().flatten() {
        let attachment = Attachment {
            name: a.str("name").filter(|s| !s.is_empty()).map(str::to_owned),
            media_type: a
                .str("mime_type")
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .or_else(|| {
                    a.str("mimeType")
                        .filter(|s| !s.is_empty())
                        .map(str::to_owned)
                }),
            size: a.u64("size"),
            blob: None,
            url: None,
            source_id: a.str("id").filter(|s| !s.is_empty()).map(str::to_owned),
            direction,
        };
        let image = attachment
            .media_type
            .as_deref()
            .is_some_and(|m| m.starts_with("image/"));
        out.push(if image {
            Block::Image(attachment)
        } else {
            Block::File(attachment)
        });
    }
    for v in meta.array("ada_visualizations").into_iter().flatten() {
        if let Some(id) = v
            .str("file_id")
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
        {
            out.push(Block::File(Attachment {
                name: v.str("title").filter(|s| !s.is_empty()).map(str::to_owned),
                source_id: Some(id),
                direction: AttachmentDirection::Output,
                ..Attachment::default()
            }));
        }
    }
    out
}

/// What a ChatGPT tool does, by the recipient name a message is addressed
/// to.
fn action(tool: &str) -> ToolAction {
    match tool {
        "python" => ToolAction::Code,
        t if t.ends_with(".exec") => ToolAction::Code,
        "file_search" | "myfiles_browser" => ToolAction::Search,
        "canmore.create_textdoc" => ToolAction::FileWrite,
        "canmore.update_textdoc" => ToolAction::FileEdit,
        t if t == "web" || t.starts_with("web.") || t.starts_with("browser") => ToolAction::Web,
        _ => ToolAction::Other,
    }
}

/// An assistant message addressed to a tool. The message is the call; its
/// own ID is the call ID its results point back to.
fn tool_call(native: &str, tool: &str, content: &Json) -> Block {
    let text = content_text(content);
    let input = match content_type(content) {
        // A tool's arguments are often JSON written as text.
        "text" => text
            .as_deref()
            .and_then(|t| serde_json::from_str::<Json>(t).ok())
            .filter(Json::is_object)
            .or_else(|| text.clone().map(Json::String))
            .unwrap_or(Json::Null),
        "code" => text.clone().map(Json::String).unwrap_or(Json::Null),
        _ => content.clone(),
    };
    let action = action(tool);
    Block::ToolCall {
        written_text: None,
        run_by_script: false,
        call_id: Some(native.to_owned()),
        tool: Some(tool.to_owned()),
        action,
        input,
        // A code tool's input is the code it runs.
        command: (action == ToolAction::Code).then_some(text).flatten(),
        path: None,
    }
}

/// A `role: tool` message: one result of the call it answers.
fn tool_result(
    content: &Json,
    msg: &Json,
    meta: &Json,
    call_id: Option<String>,
    direction: AttachmentDirection,
) -> Block {
    let kind = content_type(content);
    let mut attachments = Vec::new();
    if let Some(shot) = content.get("screenshot").filter(|s| s.is_object()) {
        attachments.push(asset(shot, direction));
    }
    for part in content
        .array("parts")
        .into_iter()
        .flatten()
        .filter(|p| p.is_object())
    {
        if let Block::Image(a) | Block::File(a) = part_block(part, direction) {
            attachments.push(a);
        }
    }
    let browsing = web_result(kind, content);
    let output = match &browsing {
        Some(data) => data.str("text").map(str::to_owned),
        None => content_text(content),
    };
    let data = browsing.or_else(|| match kind {
        "text" | "execution_output" | "multimodal_text" => None,
        _ => Some(content.clone()),
    });
    Block::ToolResult {
        reported_duration_ms: None,
        call_id,
        outcome: outcome(kind, msg, meta),
        output,
        data,
        attachments,
    }
}

/// How a tool result ended: an error content type, a failed code run, or a
/// message the tool left unfinished.
fn outcome(kind: &str, msg: &Json, meta: &Json) -> ToolOutcome {
    if kind == "system_error" {
        return ToolOutcome::Error { reason: None };
    }
    if let Some(status) = meta
        .pointer("/aggregate_result/status")
        .and_then(Json::as_str)
    {
        if status == "success" {
            return ToolOutcome::Ok;
        }
        if status.contains("exception") || status.contains("fail") || status.contains("error") {
            return ToolOutcome::Error { reason: None };
        }
        if status.contains("timeout") || status.contains("timed_out") {
            return ToolOutcome::Error {
                reason: Some(ErrorReason::TimedOut),
            };
        }
    }
    match msg.str("status").filter(|s| !s.is_empty()) {
        Some("finished_successfully") => ToolOutcome::Ok,
        None => ToolOutcome::Unknown {
            reason: UnknownReason::StatusFieldUnreadable,
        },
        Some("in_progress") => ToolOutcome::NoResult,
        Some(_) => ToolOutcome::Error {
            reason: Some(ErrorReason::Interrupted),
        },
    }
}

#[cfg(test)]
mod tests;
