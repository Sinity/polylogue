//! Claude Code's single point-keyed interpretation core. The caller supplies a
//! pinned predecessor and companions; this library never opens source files.
//! Response reconstruction, ordinary-index reflow, digest work and publication
//! remain explicit obligations over addressable contribution cells.
use crate::frame::{IgnoredKind, Record, RecordOutcome, RecordParser};
use polylogue_model::RetainedJson as Value;
use polylogue_model::*;
use std::collections::{BTreeMap, HashMap, HashSet};
pub mod keyed;
pub mod origins;
pub mod projection;
mod raw;
pub mod selection;
mod tool_result;
use keyed::*;
pub const PARSER_VERSION: u32 = 8;

#[derive(Clone, Debug)]
pub enum FileCtx {
    Main {
        stem: String,
    },
    Subagent {
        folder_owner: String,
        agent: String,
        meta: Option<Value>,
    },
}
impl FileCtx {
    pub fn main(stem: &str) -> Self {
        Self::Main { stem: stem.into() }
    }
    /// Main-file context proves membership even without one parseable record.
    /// Subagent ownership needs its own records; folder lineage alone is not it.
    pub fn direct_main_session(&self) -> Option<SessionId> {
        match self {
            Self::Main { stem } => sid(stem),
            Self::Subagent { .. } => None,
        }
    }
}
fn origin() -> Origin {
    Origin::new("claude-code").unwrap_or_else(|_| unreachable!("constant valid origin"))
}
fn sid(native: &str) -> Option<SessionId> {
    SessionId::new(&origin(), native).ok()
}
/// Parser-cell keys use strict NativeId JSON, never the transport spelling alone.
/// String and encoded-object keys remain disjoint, including literal `~u:` IDs.
fn native_key(native: &NativeId) -> Result<String, String> {
    serde_json::to_string(native).map_err(|e| e.to_string())
}
fn key_native(key: &str) -> Result<NativeId, String> {
    serde_json::from_str(key).map_err(|e| e.to_string())
}
fn provider_time(at: Timestamp) -> ProviderTime {
    ProviderTime::Intrinsic {
        at,
        precision: Precision::Instant,
        until: None,
    }
}
fn parse_time(value: Option<&Value>) -> Option<ProviderTime> {
    value.and_then(Value::as_str).and_then(parse_time_text)
}
fn parse_time_text(text: &str) -> Option<ProviderTime> {
    text.parse::<jiff::Timestamp>()
        .ok()
        .map(|t| provider_time(Timestamp(t.as_millisecond())))
}

/// Record kinds that carry nothing to keep in the index (their bytes stay stored).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ignored {
    ProgressCountedOnly,
}

impl IgnoredKind for Ignored {
    fn name(self) -> &'static str {
        match self {
            Self::ProgressCountedOnly => "progress (counted only)",
        }
    }
    fn reason(self) -> &'static str {
        match self {
            Self::ProgressCountedOnly => {
                "transient progress of a running tool or query; counted per input, payload kept only in the stored bytes"
            }
        }
    }
}

/// Changed metadata and per-record interpretation cells. No `Parsed` here is
/// advertised as a complete reconstructed Message: even first publication must
/// discharge the pending contribution/layout/evidence obligations.
#[derive(Clone, Debug)]
pub struct StepOutput {
    pub header: Header,
    pub delta: Delta,
    pub events: Vec<(SessionId, Event)>,
    pub links: Vec<Link>,
}

pub struct KeyedRecords<R: ReadView> {
    ctx: FileCtx,
    view: R,
    header: Header,
    sessions: Vec<SessionCell>,
    by_id: HashMap<String, usize>,
    responses: HashMap<(String, ResponseKey), ResponseCell>,
    addresses: HashMap<(String, String), RecordAddress>,
    contributions: HashMap<(MessageId, u64), Contribution>,
    sides: HashMap<String, String>,
    emitted_links: HashSet<LinkKey>,
    link_order: Vec<LinkKey>,
    links: Vec<Link>,
    events: Vec<(SessionId, Event)>,
    pending: BTreeMap<(String, u64), PendingMessage>,
}
impl<R: ReadView> KeyedRecords<R> {
    pub fn new(ctx: FileCtx, view: R) -> Result<Self, String> {
        if let FileCtx::Subagent {
            meta: Some(meta), ..
        } = &ctx
        {
            if meta
                .get("parentAgentId")
                .and_then(Value::native)
                .is_some_and(|n| n.encoding().is_some())
            {
                return Err("non-scalar subagent parent Session routing is unsupported".into());
            }
        }
        let header = view.header().map_err(|e| e.to_string())?;
        Ok(Self {
            ctx,
            view,
            header,
            sessions: Vec::new(),
            by_id: HashMap::new(),
            responses: HashMap::new(),
            addresses: HashMap::new(),
            contributions: HashMap::new(),
            sides: HashMap::new(),
            emitted_links: HashSet::new(),
            link_order: Vec::new(),
            links: Vec::new(),
            events: Vec::new(),
            pending: BTreeMap::new(),
        })
    }
    fn session_index(&mut self, native: &str, kind: SessionKind) -> Result<Option<usize>, String> {
        if let Some(&i) = self.by_id.get(native) {
            return Ok(Some(i));
        }
        let cell = match self.view.session(native).map_err(|e| e.to_string())? {
            Some(cell) => cell,
            None => {
                let Some(id) = sid(native) else {
                    return Ok(None);
                };
                let mut session =
                    Session::new(id, origin(), native, kind).map_err(|e| e.to_string())?;
                session.harness = Some("claude-code".into());
                let ordinal = self.header.next_session_ordinal;
                self.header.next_session_ordinal = ordinal
                    .checked_add(1)
                    .ok_or("Claude session ordinal overflow")?;
                SessionCell {
                    session,
                    ordinal,
                    seen_own_user: false,
                    next_message_ordinal: 0,
                    last_message: None,
                    facts_pending: false,
                }
            }
        };
        if cell.session.native_id != native || cell.session.origin != origin() {
            return Err("Claude session cell disagrees with its lookup identity".into());
        }
        let i = self.sessions.len();
        self.sessions.push(cell);
        self.by_id.insert(native.into(), i);
        Ok(Some(i))
    }
    fn response(&self, native: &str, key: &ResponseKey) -> Result<Option<ResponseCell>, String> {
        match self.responses.get(&(native.into(), key.clone())) {
            Some(cell) => Ok(Some(cell.clone())),
            None => self.view.response(native, key).map_err(|e| e.to_string()),
        }
    }
    fn address(&self, native: &str, uuid: &str) -> Result<Option<RecordAddress>, String> {
        match self.addresses.get(&(native.into(), uuid.into())) {
            Some(cell) => Ok(Some(cell.clone())),
            None => self.view.record(native, uuid).map_err(|e| e.to_string()),
        }
    }
    fn contribution(&self, message: &MessageId, slot: u64) -> Result<Option<Contribution>, String> {
        match self.contributions.get(&(message.clone(), slot)) {
            Some(cell) => Ok(Some(cell.clone())),
            None => self
                .view
                .contribution(message, slot)
                .map_err(|e| e.to_string()),
        }
    }
    fn add_link(
        &mut self,
        kind: LinkKind,
        how: LinkHow,
        parent: &str,
        child: &str,
        evidence: &str,
        attrs: Option<Value>,
    ) -> Result<(), String> {
        let (Some(parent_id), Some(child_id)) = (sid(parent), sid(child)) else {
            return Ok(());
        };
        if parent_id == child_id {
            return Ok(());
        }
        let key = LinkKey {
            parent: parent.into(),
            child: child.into(),
            kind,
            how,
        };
        if self.emitted_links.contains(&key)
            || self.view.has_link(&key).map_err(|e| e.to_string())?
        {
            return Ok(());
        }
        self.emitted_links.insert(key.clone());
        self.link_order.push(key);
        let mut link =
            Link::new(kind, how, parent_id, child_id, evidence).map_err(|e| e.to_string())?;
        link.attrs = attrs;
        self.links.push(link);
        Ok(())
    }
    /// The session this message record belongs to, and the message it copies, if any.
    fn route(
        &mut self,
        v: &Value,
        uuid: &str,
    ) -> Result<Option<(usize, Option<(String, String)>)>, String> {
        let sid_field = v
            .get("sessionId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        Ok(match self.ctx.clone() {
            FileCtx::Main { stem } => {
                // A later supplement may omit every sidechain routing field.
                // The first UUID's established membership remains authoritative.
                let established = match self.sides.get(uuid) {
                    Some(side) => Some(side.clone()),
                    None => self.view.side(uuid).map_err(|e| e.to_string())?,
                };
                if let Some(side) = established {
                    if v.get("isSidechain").and_then(Value::as_bool) == Some(false) {
                        return Ok(None);
                    }
                    let i = match self.session_index(&side, SessionKind::Subagent)? {
                        Some(i) => i,
                        None => return Ok(None),
                    };
                    return Ok(Some((i, None)));
                }
                let parent = v
                    .get("parentUuid")
                    .and_then(Value::native)
                    .map(native_key)
                    .transpose()?
                    .unwrap_or_default();
                if v.get("isSidechain").and_then(Value::as_bool) == Some(true) {
                    // A sidechain subtree in a main file: its own session, named by its root.
                    let known = match self.sides.get(&parent) {
                        Some(side) => Some(side.clone()),
                        None => self.view.side(&parent).map_err(|e| e.to_string())?,
                    };
                    let side = if let Some(s) = known {
                        s
                    } else {
                        let side = match v.get("agentId").and_then(Value::as_str) {
                            Some(agent) => format!("{stem}/{agent}"),
                            None => {
                                let original = key_native(uuid)?;
                                let scalar = original.unicode_str().ok_or("non-scalar sidechain root needs scoped native Session identity")?;
                                format!("{stem}~side-{scalar}")
                            }
                        };
                        self.add_link(
                            LinkKind::Subagent,
                            LinkHow::Fresh,
                            &stem,
                            &side,
                            "isSidechain",
                            None,
                        )?;
                        side
                    };
                    self.sides.insert(uuid.to_owned(), side.clone());
                    let i = match self.session_index(&side, SessionKind::Subagent)? {
                        Some(i) => i,
                        None => return Ok(None),
                    };
                    return Ok(Some((i, None)));
                }
                let i = match self.session_index(&stem, SessionKind::Main)? {
                    Some(i) => i,
                    None => return Ok(None),
                };
                if let Some(ff) = v.get("forkedFrom") {
                    let Some(src) = ff.get("sessionId").and_then(Value::as_str) else {
                        return Ok(None);
                    };
                    let Some(msg) = ff.get("messageUuid").and_then(Value::native) else {
                        return Ok(None);
                    };
                    self.add_link(
                        LinkKind::Continuation,
                        LinkHow::Fork,
                        src,
                        &stem,
                        "forkedFrom",
                        None,
                    )?;
                    return Ok(Some((i, Some((src.into(), native_key(msg)?)))));
                }
                if !sid_field.is_empty() && sid_field != stem {
                    self.add_link(
                        LinkKind::Continuation,
                        LinkHow::Resume,
                        &sid_field,
                        &stem,
                        "copied-sessionId",
                        None,
                    )?;
                    return Ok(Some((i, Some((sid_field.clone(), uuid.into())))));
                }
                Some((i, None))
            }
            FileCtx::Subagent {
                folder_owner,
                agent,
                meta,
            } => {
                let owner = self
                    .header
                    .owner
                    .get_or_insert_with(|| {
                        if sid_field.is_empty() {
                            folder_owner.clone()
                        } else {
                            sid_field.clone()
                        }
                    })
                    .clone();
                let own = format!("{owner}/{agent}");
                if !self.by_id.contains_key(&own)
                    && self
                        .view
                        .session(&own)
                        .map_err(|e| e.to_string())?
                        .is_none()
                {
                    let parent = match meta
                        .as_ref()
                        .and_then(|m| m.get("parentAgentId"))
                        .and_then(Value::as_str)
                    {
                        Some(p) => format!("{owner}/{p}"),
                        None => folder_owner.clone(),
                    };
                    let evidence = if parent == folder_owner {
                        "folder"
                    } else {
                        "meta.parentAgentId"
                    };
                    let how = if meta
                        .as_ref()
                        .and_then(|m| m.get("isFork"))
                        .and_then(Value::as_bool)
                        == Some(true)
                    {
                        LinkHow::Copied
                    } else {
                        LinkHow::Fresh
                    };
                    self.add_link(
                        LinkKind::Subagent,
                        how,
                        &parent,
                        &own,
                        evidence,
                        meta.clone(),
                    )?;
                }
                let i = match self.session_index(&own, SessionKind::Subagent)? {
                    Some(i) => i,
                    None => return Ok(None),
                };
                if let Some(m) = &meta {
                    let s = &mut self.sessions[i].session;
                    s.team = m.get("teamName").and_then(Value::as_str).map(str::to_owned);
                    s.agent_name = m.get("name").and_then(Value::as_str).map(str::to_owned);
                }
                let rec_agent = v
                    .get("agentId")
                    .and_then(Value::as_str)
                    .unwrap_or(&agent)
                    .to_owned();
                if rec_agent != agent {
                    let src = format!("{owner}/{rec_agent}");
                    self.add_link(
                        LinkKind::Continuation,
                        LinkHow::Resume,
                        &src,
                        &own,
                        "copied-agentId",
                        None,
                    )?;
                    return Ok(Some((i, Some((src, uuid.into())))));
                }
                Some((i, None))
            }
        })
    }

    fn message_record(
        &mut self,
        rec: &Record<'_>,
        ty: &str,
    ) -> Result<RecordOutcome<Ignored>, String> {
        if !rec.table.is_empty()
            || rec.end.checked_sub(rec.start) != u64::try_from(rec.bytes.len()).ok()
        {
            return Err("Claude original record does not have an exact JSONL byte range".into());
        }
        let source_range = crate::selection::RecordRange {
            start: rec.start,
            end: rec.end,
        };
        let incoming = &rec.value;
        validate_structural_keys(incoming)?;
        let Some(uuid_native) = incoming
            .get("uuid")
            .and_then(Value::native)
            .filter(|s| !s.as_str().is_empty())
        else {
            return Ok(RecordOutcome::Unrecognized("message record without uuid"));
        };
        let uuid_key = native_key(uuid_native)?;
        let uuid = uuid_key.as_str();
        let Some((si, copy)) = self.route(incoming, uuid)? else {
            return Ok(RecordOutcome::Unrecognized("unroutable record"));
        };
        let native = self.sessions[si].session.native_id.clone();
        let address = self.address(&native, uuid)?;
        let old = match &address {
            Some(a) => Some(
                self.contribution(&a.message, a.item)?
                    .ok_or("missing committed Claude contribution")?,
            ),
            None => None,
        };
        if old.as_ref().is_some_and(|cell| cell.uuid != uuid) {
            return Err("committed Claude contribution UUID disagrees with its address".into());
        }
        let raw = raw::overlay(old.as_ref().map(|c| c.raw.as_slice()), rec.bytes)?;
        let v = raw::decode_effective(&raw, incoming)?;
        if v.get("type").and_then(Value::as_str) != Some(ty) {
            return Ok(RecordOutcome::Unrecognized("UUID record type conflict"));
        }
        let Some(msg) = v.get("message").filter(|m| m.is_object()) else {
            return Ok(RecordOutcome::Unrecognized(
                "message record without typed message",
            ));
        };
        let key = if ty == "assistant" {
            let empty = NativeId::unicode("");
            let request = v.get("requestId").and_then(Value::native).unwrap_or(&empty);
            let response = msg.get("id").and_then(Value::native);
            ResponseKey {
                role: Role::Assistant,
                response: native_key(response.unwrap_or(if request.as_str().is_empty() {
                    uuid_native
                } else {
                    &empty
                }))?,
                request: native_key(request)?,
            }
        } else {
            ResponseKey {
                role: Role::User,
                response: uuid.to_owned(),
                request: native_key(&NativeId::unicode(""))?,
            }
        };
        if address
            .as_ref()
            .is_some_and(|a| a.response != key || a.role != ty)
        {
            return Ok(RecordOutcome::Unrecognized(
                "UUID response membership conflict",
            ));
        }
        let record_origins = origins::RecordOrigins::overlay(
            old.as_ref().map(|c| &c.origins),
            incoming,
            &v,
            Some(source_range),
        );
        let (mut blocks, bodies) = self.blocks_and_bodies(&v, &raw)?;
        let content = msg.get("content").unwrap_or(&Value::Null);
        let old_same = old
            .as_ref()
            .is_some_and(|c| c.raw == raw && c.bodies == bodies);
        let mut response = match self.response(&native, &key)? {
            Some(cell) => cell,
            None => {
                if address.is_some() {
                    return Err(
                        "missing committed Claude response referenced by UUID address".into(),
                    );
                }
                let id = MessageId::from_native(self.sessions[si].session.id.clone(), uuid_native)
                    .map_err(|e| e.to_string())?;
                let context = selection::AuthorContext {
                    role: if ty == "assistant" {
                        Role::Assistant
                    } else {
                        Role::User
                    },
                    first_subagent_instruction: ty == "user"
                        && !blocks
                            .iter()
                            .any(|b| matches!(b, BlockContent::ToolResult(_)))
                        && copy.is_none()
                        && !self.sessions[si].seen_own_user
                        && self.sessions[si].session.kind == SessionKind::Subagent,
                    copied: copy.is_some(),
                };
                let (written_by, prompt_kind, from_name) =
                    selection::classify_author(&v, content, context);
                if ty == "user"
                    && !blocks
                        .iter()
                        .any(|b| matches!(b, BlockContent::ToolResult(_)))
                    && copy.is_none()
                {
                    self.sessions[si].seen_own_user = true;
                }
                let ordinal = self.sessions[si].next_message_ordinal;
                self.sessions[si].next_message_ordinal = ordinal
                    .checked_add(1)
                    .ok_or("Claude message ordinal overflow")?;
                let mut message = Message::new(
                    id,
                    IdKind::Provider,
                    if ty == "assistant" {
                        Role::Assistant
                    } else {
                        Role::User
                    },
                    written_by,
                );
                message.prompt_kind = prompt_kind;
                message.from_name = from_name;
                message.at = parse_time(v.get("timestamp"));
                if let Some((source, msg)) = &copy {
                    message.inherited_from = sid(source).and_then(|s| {
                        key_native(msg)
                            .ok()
                            .and_then(|n| MessageId::from_native(s, &n).ok())
                    });
                    message.copy_origin = Some(CopyOrigin::Known);
                }
                ResponseCell {
                    message,
                    ordinal,
                    author_selection: Some(selection::AuthorSelection::new(source_range, context)),
                    items: 0,
                    revision: 0,
                    selected: Default::default(),
                    incomplete_bodies: 0,
                    selected_from: Default::default(),
                    selected_facts: Default::default(),
                    selected_origins: Default::default(),
                    stop_sequence: None,
                    stop_sequence_raw: None,
                }
            }
        };
        if response.ordinal >= self.sessions[si].next_message_ordinal {
            return Err("Claude response ordinal exceeds its session counter".into());
        }
        if address
            .as_ref()
            .is_some_and(|a| a.message != response.message.id || a.item >= response.items)
        {
            return Err("committed Claude UUID address disagrees with its response".into());
        }
        if response.message.id.session() != &self.sessions[si].session.id {
            return Err("committed Claude response belongs to a different session".into());
        }
        if old_same {
            return Ok(RecordOutcome::Used);
        }
        let item = match &address {
            Some(a) => a.item,
            None => {
                let item = response.items;
                response.items = item
                    .checked_add(1)
                    .ok_or("Claude contribution ordinal overflow")?;
                item
            }
        };
        let revision = response
            .revision
            .checked_add(1)
            .ok_or("Claude evidence revision overflow")?;
        let raw_changed = old.as_ref().is_none_or(|previous| previous.raw != raw);
        let mut projected = projection::record(
            &v,
            if raw_changed {
                &rec.value
            } else {
                &Value::Null
            },
            &blocks,
            revision,
            old.as_ref().map(|c| &c.projection),
        )?;
        for fact in projected.facts.iter_mut().chain(&mut projected.retained) {
            fact.selection = selection::for_fact(fact, &v, &record_origins);
        }
        if item == 0 {
            response.message.parent = None;
            if let Some(parent_uuid) = v
                .get("parentUuid")
                .and_then(Value::native)
                .filter(|p| !p.as_str().is_empty())
            {
                let resolved = self
                    .address(&native, &native_key(parent_uuid)?)?
                    .filter(|parent| {
                        parent.item == 0
                            && parent.message != response.message.id
                            && parent.message.session() == response.message.id.session()
                    });
                let parent_fact = projected.facts.iter_mut().find(|fact| {
                    matches!(
                        fact.field,
                        projection::Field::FirstMessage(
                            polylogue_model::merge::MessageField::Parent
                        )
                    )
                });
                if let (Some(parent), Some(fact)) = (resolved, parent_fact) {
                    fact.selection =
                        selection::parent(parent_uuid, parent.message.clone(), &record_origins);
                    if fact.selection.is_some() {
                        response.message.parent = Some(parent.message);
                    } else {
                        projected
                            .obligations
                            .push(projection::CountingObligation::ParentResolution);
                    }
                } else {
                    projected
                        .obligations
                        .push(projection::CountingObligation::ParentResolution);
                }
            }
            if let Some(previous) = &old {
                let old_raw: Value =
                    serde_json::from_slice(&previous.raw).map_err(|e| e.to_string())?;
                if previous
                    .projection
                    .obligations
                    .contains(&projection::CountingObligation::FirstAuthorUpdate)
                    || response.author_selection.as_ref().is_some_and(|selection| {
                        selection::classify_author(&v, content, selection.context)
                            != (
                                response.message.written_by,
                                response.message.prompt_kind,
                                response.message.from_name.clone(),
                            )
                    })
                    || [
                        "origin",
                        "promptSource",
                        "isMeta",
                        "isVisibleInTranscriptOnly",
                        "isApiErrorMessage",
                    ]
                    .iter()
                    .any(|key| old_raw.get(*key) != v.get(*key))
                {
                    projected
                        .obligations
                        .push(projection::CountingObligation::FirstAuthorUpdate);
                }
            }
        }
        // Stable item slots own blocks until the body finalizer assigns ordinary
        // one-counter BlockIds. A changed count never mutates a historical body.
        let contribution = Contribution {
            uuid: uuid.into(),
            raw,
            origins: record_origins,
            blocks: std::mem::take(&mut blocks),
            bodies,
            projection: projected,
        };
        let old_missing = old.as_ref().map_or(0, |c| {
            c.bodies.iter().filter(|b| b.payload.is_none()).count() as u64
        });
        let missing = contribution
            .bodies
            .iter()
            .filter(|b| b.payload.is_none())
            .count() as u64;
        response.incomplete_bodies = response
            .incomplete_bodies
            .checked_sub(old_missing)
            .and_then(|n| n.checked_add(missing))
            .ok_or("Claude required-body counter mismatch")?;
        response.revision = revision;
        // Only newly supplied fields advance their response-level selection.
        // Inherited fields in a same-UUID overlay do not become later updates.
        if old
            .as_ref()
            .is_none_or(|previous| previous.raw != contribution.raw)
        {
            if incoming
                .get("message")
                .and_then(|message| message.get("stop_sequence"))
                .is_some()
            {
                let fact=contribution.projection.facts.iter().find(|fact|matches!(&fact.field,projection::Field::SelectedExtra{key} if key=="stop_sequence")).cloned().ok_or("Claude stop_sequence supplier lost its source fact")?;
                let value = v
                    .get("message")
                    .and_then(|message| message.get("stop_sequence"))
                    .cloned()
                    .ok_or("Claude stop_sequence normalized value unavailable")?;
                response.stop_sequence_raw = Some(raw::canonical(
                    raw::optional_field(rec.bytes, &["message", "stop_sequence"])?
                        .ok_or("Claude stop_sequence raw supplier unavailable")?,
                )?);
                response.stop_sequence = Some(fact);
                let attrs = response
                    .message
                    .attrs
                    .get_or_insert_with(|| Value::Object(Default::default()))
                    .as_object_mut()
                    .ok_or("Claude normalized extras carrier is not an object")?;
                attrs.insert("stop_sequence".into(), value);
            }
            for (index, path) in METADATA_PATHS.iter().enumerate() {
                if let Some(value) = raw::optional_field(rec.bytes, path)? {
                    response.selected[index] = Some(raw::canonical(value)?);
                    response.selected_from[index] = Some(item);
                    let semantic = path
                        .iter()
                        .try_fold(&v, |node, key| node.get(key))
                        .ok_or("Claude supplied metadata lost its retained value")?;
                    // Typed metadata is a text projection too; keep diagnostics
                    // on the original record rather than a canonical reparse.
                    if index == 0 {
                        let _ = semantic.get("service_tier").and_then(Value::text);
                    } else {
                        let _ = semantic.text();
                    }
                    response.selected_facts[index] =
                        projection::selected(index, semantic, revision);
                    let selected = crate::selection::SelectedRaw {
                        record: source_range,
                        path: path
                            .iter()
                            .map(|key| crate::selection::PathPart::Key((*key).into()))
                            .collect(),
                    };
                    response.selected_origins[index] = Some(selected.clone());
                    for fact in &mut response.selected_facts[index] {
                        fact.selection = selection::for_selected(fact, index, &semantic, &selected);
                    }
                }
            }
        }
        if item == 0 {
            response.message.at = parse_time(v.get("timestamp"));
        }
        apply_selected(&mut response)?;
        if v.get("isAbortedMidStream").and_then(Value::as_bool) == Some(true) {
            response.message.stop_reason = Some("aborted".into());
            response.message.in_progress = Some(false);
        }
        response.message.version_hash = None;
        let pending = PendingMessage {
            session_native: native.clone(),
            response: key.clone(),
            ordinal: response.ordinal,
            message: response.message.id.clone(),
            revision: response.revision,
            items: response.items,
            incomplete: response.incomplete_bodies != 0,
        };
        self.pending.insert(
            (
                response.message.id.session().as_str().into(),
                response.ordinal,
            ),
            pending,
        );
        self.addresses.insert(
            (native.clone(), uuid.into()),
            RecordAddress {
                response: key.clone(),
                message: response.message.id.clone(),
                item,
                role: ty.into(),
            },
        );
        self.contributions
            .insert((response.message.id.clone(), item), contribution);
        self.sessions[si].last_message = Some(response.message.id.clone());
        self.sessions[si].facts_pending = true;
        let session = &mut self.sessions[si].session;
        if session.cwd.is_none() {
            session.cwd = v.get("cwd").and_then(Value::as_str).map(str::to_owned);
        }
        if session.branch.is_none() {
            session.branch = v
                .get("gitBranch")
                .and_then(Value::as_str)
                .map(str::to_owned);
        }
        if session.harness_version.is_none() {
            session.harness_version = v.get("version").and_then(Value::as_str).map(str::to_owned);
        }
        self.responses.insert((native, key), response);
        Ok(RecordOutcome::Used)
    }
    /// The session a non-message record of this file belongs to.
    fn file_session(&mut self, v: &Value) -> Result<Option<usize>, String> {
        match self.ctx.clone() {
            FileCtx::Main { stem } => self.session_index(&stem, SessionKind::Main),
            FileCtx::Subagent {
                folder_owner,
                agent,
                ..
            } => {
                let sid_field = v
                    .get("sessionId")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_owned();
                let owner = self
                    .header
                    .owner
                    .get_or_insert_with(|| {
                        if sid_field.is_empty() {
                            folder_owner.clone()
                        } else {
                            sid_field
                        }
                    })
                    .clone();
                self.session_index(&format!("{owner}/{agent}"), SessionKind::Subagent)
            }
        }
    }

    fn event(&mut self, rec: &Record<'_>, kind: &str) -> Result<RecordOutcome<Ignored>, String> {
        let Some(si) = self.file_session(&rec.value)? else {
            return Ok(RecordOutcome::Unrecognized("bad session id"));
        };
        let at = parse_time(rec.value.get("timestamp"));
        let after_message = self.sessions[si].last_message.clone();
        self.events.push((
            self.sessions[si].session.id.clone(),
            Event {
                kind: kind.into(),
                at,
                after_message,
                model_saw: None,
                data: rec.value.opaque(),
            },
        ));
        Ok(RecordOutcome::Used)
    }
}

/// The first Claude Code version writing `promptSource` (spec/sources/claude-code.md;
/// measured over the operator's files, evidence/claude-code-real-accounting.md).
const PROMPT_SOURCE_SINCE: [u32; 3] = [2, 1, 161];

fn predates_prompt_source(version: &str) -> bool {
    let parts: Vec<u32> = version.split('.').map(|p| p.parse().unwrap_or(0)).collect();
    let mut v = [0u32; 3];
    for (i, p) in parts.iter().take(3).enumerate() {
        v[i] = *p;
    }
    v < PROMPT_SOURCE_SINCE
}

fn prompt_kind_of(content: &Value) -> PromptKind {
    let text = match content {
        Value::String(s) => s.as_str(),
        Value::Array(a) => a
            .iter()
            .find_map(|b| b.get("text").and_then(Value::as_str))
            .unwrap_or(""),
        _ => "",
    };
    let t = text.trim_start();
    // A bare slash command typed as plain text (`/compact`, `/model opus`): one line whose first
    // word is `/name`. Seen on real data without the `<command-name>` wrapper
    // (evidence/briefing-trial.md), where it had been read as an unanswered prose prompt.
    let slash = t.starts_with('/')
        && !t.trim_end().contains('\n')
        && t[1..].split_whitespace().next().is_some_and(|w| {
            !w.is_empty()
                && w.chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == ':')
        });
    if slash
        || t.starts_with("<command-name>")
        || t.starts_with("<bash-input>")
        || t.starts_with('!')
        || t.starts_with('#')
    {
        PromptKind::Command
    } else {
        PromptKind::Prose
    }
}

fn usage_of(u: &Value) -> Usage {
    let g = |k: &str| u.get(k).and_then(Value::as_u64);
    Usage {
        input: g("input_tokens"),
        output: g("output_tokens"),
        cache_read: g("cache_read_input_tokens"),
        cache_write: g("cache_creation_input_tokens"),
        cache_write_5m: u
            .get("cache_creation")
            .and_then(|m| m.get("ephemeral_5m_input_tokens"))
            .and_then(Value::as_u64),
        cache_write_1h: u
            .get("cache_creation")
            .and_then(|m| m.get("ephemeral_1h_input_tokens"))
            .and_then(Value::as_u64),
        service_tier: u
            .get("service_tier")
            .and_then(Value::as_str)
            .map(str::to_owned),
        reasoning: u
            .get("output_tokens_details")
            .and_then(|m| m.get("thinking_tokens"))
            .and_then(Value::as_u64),
        ..Usage::default()
    }
}

fn text_of(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.text().to_owned()),
        Value::Array(a) => {
            let parts: Vec<&str> = a
                .iter()
                .filter_map(|x| x.get("text").and_then(Value::as_str))
                .collect();
            if parts.is_empty() && !a.is_empty() {
                None
            } else {
                Some(parts.join("\n"))
            }
        }
        _ => None,
    }
}

fn blocks_of(record: &Value, content: &Value) -> Vec<BlockContent> {
    match content {
        Value::String(s) => vec![BlockContent::Text {
            text: s.text().to_owned(),
        }],
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(
                |(index, it)| match it.get("type").and_then(Value::as_str).unwrap_or("") {
                    "text" => BlockContent::Text {
                        text: it
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_owned(),
                    },
                    "thinking" => BlockContent::Thinking(Thinking {
                        text: it
                            .get("thinking")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        signature: it
                            .get("signature")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        encrypted: it.get("encrypted").and_then(Value::as_bool),
                        data: it.get("data").and_then(Value::as_str).map(str::to_owned),
                    }),
                    "tool_use" => {
                        let input = it.get("input").map(Value::opaque);
                        BlockContent::ToolCall(ToolCall {
                            tool: it.get("name").and_then(Value::as_str).map(str::to_owned),
                            call_id: it.get("id").and_then(Value::native).cloned(),
                            command: input
                                .as_ref()
                                .and_then(|v| v.get("command"))
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                            input,
                        })
                    }
                    "tool_result" => {
                        BlockContent::ToolResult(tool_result::decide(record, index, it).result)
                    }
                    "image" => BlockContent::Image(Attachment {
                        blob: None,
                        name: None,
                        media_type: it
                            .get("source")
                            .and_then(|s| s.get("media_type"))
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                        size: None,
                        url: None,
                        direction: AttachmentDirection::Input,
                        ..Attachment::default()
                    }),
                    other => BlockContent::Other {
                        type_name: other.to_owned(),
                        data: it.opaque(),
                    },
                },
            )
            .collect(),
        _ => Vec::new(),
    }
}

/// The exit code a shell result states in its output text (`Exit code N`, `exit code: N`;
/// spec/sources/claude-code.md, tool results); the last such line wins.
fn exit_code_in(text: &str) -> Option<i64> {
    text.lines().rev().find_map(|line| {
        let l = line.trim().to_ascii_lowercase();
        let rest = l
            .strip_prefix("exit code:")
            .or_else(|| l.strip_prefix("exit code"))?;
        rest.trim().parse().ok()
    })
}

impl<R: ReadView> RecordParser for KeyedRecords<R> {
    type Error = String;
    type Ignored = Ignored;
    type Output = StepOutput;
    fn record(
        &mut self,
        rec: &Record<'_>,
    ) -> std::result::Result<RecordOutcome<Self::Ignored>, Self::Error> {
        for field in ["sessionId", "agentId", "continuedInSessionId"] {
            if rec
                .value
                .get(field)
                .and_then(Value::native)
                .is_some_and(|n| n.encoding().is_some())
            {
                return Err("non-scalar source-derived Session routing is unsupported".into());
            }
        }
        if rec
            .value
            .get("forkedFrom")
            .and_then(|f| f.get("sessionId"))
            .and_then(Value::native)
            .is_some_and(|n| n.encoding().is_some())
        {
            return Err("non-scalar fork Session routing is unsupported".into());
        }
        Ok({
            let ty = rec.value.get("type").and_then(Value::as_str).unwrap_or("");
            match ty {
                "user"
                    if rec.value.get("isCompactSummary").and_then(Value::as_bool) == Some(true) =>
                {
                    self.event(rec, "compaction_summary")?
                }
                "user" | "assistant" => self.message_record(rec, ty)?,
                "system" => match rec.value.get("subtype").and_then(Value::as_str) {
                    Some("compact_boundary") => self.event(rec, "compaction")?,
                    Some("microcompact_boundary") => self.event(rec, "microcompaction")?,
                    Some("model_refusal_no_fallback" | "model_refusal_fallback") => {
                        self.event(rec, "model_refusal")?
                    }
                    // turn_duration, api_error, away_summary, stop_hook_summary, local_command,
                    // scheduled_task_fire, bridge_status, informational and other subtypes: an event of
                    // the same name (spec/sources/claude-code.md, record table).
                    Some(sub) => {
                        let sub = sub.to_owned();
                        self.event(rec, &sub)?
                    }
                    None => RecordOutcome::Unrecognized("system without subtype"),
                },
                "attachment" => self.event(rec, "attachment")?,
                "pr-link" => self.event(rec, "pr_link")?,
                "permission-mode" => self.event(rec, "permission_mode")?,
                "mode" => self.event(rec, "mode")?,
                "queue-operation" => self.event(rec, "queue")?,
                "last-prompt" => self.event(rec, "last_prompt")?,
                "file-history-snapshot" | "file-history-delta" => {
                    self.event(rec, "file_history")?
                }
                "cost-state" => self.event(rec, "provider_cost")?,
                "bridge-session" | "frame-link" | "worktree-state" | "agent-name"
                | "agent-color" | "relocated" => {
                    let kind = ty.replace('-', "_");
                    self.event(rec, &kind)?
                }
                "started" => self.event(rec, "workflow_started")?,
                "result" => self.event(rec, "workflow_result")?,
                "continued-in" => {
                    if let (FileCtx::Main { stem }, Some(next)) = (
                        self.ctx.clone(),
                        rec.value
                            .get("continuedInSessionId")
                            .and_then(Value::as_str),
                    ) {
                        let next = next.to_owned();
                        self.add_link(
                            LinkKind::Continuation,
                            LinkHow::AfterCompaction,
                            &stem,
                            &next,
                            "continued-in",
                            None,
                        )?;
                        RecordOutcome::Used
                    } else {
                        RecordOutcome::Unrecognized("continued-in without a target")
                    }
                }
                // Evidence for the subagent link; the copied records themselves carry the lineage.
                "fork-context-ref" => {
                    RecordOutcome::Unrecognized("fork context lineage not yet derived")
                }
                "custom-title" | "ai-title" | "summary" => {
                    let (rank, field) = match ty {
                        "custom-title" => (3, "customTitle"),
                        "ai-title" => (2, "aiTitle"),
                        _ => (1, "summary"),
                    };
                    if let Some(t) = rec.value.get(field).and_then(Value::as_str)
                        && self.header.title.as_ref().is_none_or(|(r, _)| rank >= *r)
                    {
                        self.header.title = Some((rank, t.to_owned()));
                        // Metadata-only input is itself a possible Session contributor;
                        // preserve its owner even when it has no Message rows.
                        if let FileCtx::Main { stem } = self.ctx.clone() {
                            self.session_index(&stem, SessionKind::Main)?;
                        }
                    }
                    RecordOutcome::Used
                }
                "progress" => match rec
                    .value
                    .get("data")
                    .and_then(|d| d.get("type"))
                    .and_then(Value::as_str)
                {
                    // Subagent evidence and hook runs are kept (spec/sources/claude-code.md, record table).
                    Some("agent_progress") => {
                        RecordOutcome::Unrecognized("agent progress linkage not yet derived")
                    }
                    Some("hook_progress") => self.event(rec, "hook_run")?,
                    // Counted only: payload not stored in the index.
                    Some(
                        "bash_progress"
                        | "waiting_for_task"
                        | "query_update"
                        | "search_results_received"
                        | "mcp_progress",
                    ) => RecordOutcome::Ignored(Ignored::ProgressCountedOnly),
                    _ => RecordOutcome::Unrecognized("progress kind"),
                },
                // The table: counted under unrecognized, its string kept in the stored input.
                "atis-latch" => RecordOutcome::Unrecognized("atis-latch"),
                _ => RecordOutcome::Unrecognized("record type"),
            }
        })
    }
    fn finish(self) -> Result<Self::Output, Self::Error> {
        self.finish_output()
    }
}

impl<R: ReadView> KeyedRecords<R> {
    fn finish_output(mut self) -> Result<StepOutput, String> {
        if let (FileCtx::Main { stem }, Some((_, title))) = (&self.ctx, &self.header.title) {
            if let Some(&i) = self.by_id.get(stem) {
                self.sessions[i].session.title = Some(title.clone());
            }
        }
        self.sessions.sort_by_key(|c| c.ordinal);
        let mut responses: Vec<_> = self
            .responses
            .into_iter()
            .map(|((s, k), c)| (s, k, c))
            .collect();
        responses.sort_by(|a, b| (&a.0, a.2.ordinal).cmp(&(&b.0, b.2.ordinal)));
        let mut addresses: Vec<_> = self
            .addresses
            .into_iter()
            .map(|((s, u), c)| (s, u, c))
            .collect();
        addresses.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        let mut contributions: Vec<_> = self
            .contributions
            .into_iter()
            .map(|((m, i), c)| (m, i, c))
            .collect();
        contributions.sort_by(|a, b| {
            (a.0.session().as_str(), a.0.as_str(), a.1).cmp(&(
                b.0.session().as_str(),
                b.0.as_str(),
                b.1,
            ))
        });
        let mut sides: Vec<_> = self.sides.into_iter().collect();
        sides.sort();
        Ok(StepOutput {
            header: self.header,
            delta: Delta {
                sessions: self.sessions,
                responses,
                addresses,
                contributions,
                sides,
                links: self.link_order,
                pending_messages: self.pending.into_values().collect(),
            },
            events: self.events,
            links: self.links,
        })
    }
}

const METADATA_PATHS: [&[&str]; 4] = [
    &["message", "usage"],
    &["message", "stop_reason"],
    &["message", "model"],
    &["timestamp"],
];
fn apply_selected(response: &mut ResponseCell) -> Result<(), String> {
    let values: Vec<Option<Value>> = response
        .selected
        .iter()
        .map(|v| {
            v.as_ref()
                .map(|b| serde_json::from_slice(b).map_err(|e| e.to_string()))
                .transpose()
        })
        .collect::<Result<_, _>>()?;
    response.message.usage = values[0].as_ref().filter(|v| v.is_object()).map(usage_of);
    response.message.stop_reason = values[1]
        .as_ref()
        .and_then(Value::as_str)
        .map(str::to_owned);
    if response.message.written_by == WrittenBy::Model {
        response.message.model = values[2]
            .as_ref()
            .and_then(Value::as_str)
            .map(str::to_owned);
    }
    response.message.completed_at = parse_time(values[3].as_ref());
    Ok(())
}

impl<R: ReadView> KeyedRecords<R> {
    fn blocks_and_bodies(
        &self,
        record: &Value,
        _raw: &[u8],
    ) -> Result<(Vec<BlockContent>, Vec<RequiredBody>), String> {
        let content = record
            .get("message")
            .and_then(|m| m.get("content"))
            .unwrap_or(&Value::Null);
        let mut blocks = blocks_of(record, content);
        let mut bodies = Vec::new();
        let Some(items) = content.as_array() else {
            return Ok((blocks, bodies));
        };
        let result_count = items
            .iter()
            .filter(|it| it.get("type").and_then(Value::as_str) == Some("tool_result"))
            .count();
        for (index, item) in items.iter().enumerate() {
            if item.get("type").and_then(Value::as_str) != Some("tool_result") {
                continue;
            }
            let text = item.get("content").and_then(text_of).unwrap_or_default();
            let path = body_path(record, item, result_count, &text);
            if let Some(path) = path {
                let payload = match self.view.body(path).map_err(|e| e.to_string())? {
                    Some(body) => Some(body),
                    None => match path.rsplit('/').next().filter(|name| *name != path) {
                        Some(name) => self.view.body(name).map_err(|e| e.to_string())?,
                        None => None,
                    },
                };
                if payload.as_ref().is_some_and(|body| {
                    body.length == text.len() as u64
                        && body.blob == polylogue_model::BlobHash::of(text.as_bytes())
                }) {
                    continue;
                }
                let content_item =
                    u32::try_from(index).map_err(|_| "Claude content-item overflow")?;
                if let Some(BlockContent::ToolResult(result)) = blocks.get_mut(index) {
                    match &payload {
                        Some(body) => result.output = Some(ToolOutput::Blob(body.blob)),
                        None => {
                            result.outcome = ToolOutcome::Unknown {
                                reason: UnknownReason::OutputNotRecorded,
                            }
                        }
                    }
                }
                bodies.push(RequiredBody {
                    content_item,
                    payload,
                });
            }
        }
        Ok((blocks, bodies))
    }
}

fn body_path<'a>(
    record: &'a Value,
    item: &'a Value,
    result_count: usize,
    text: &'a str,
) -> Option<&'a str> {
    item.get("persistedOutputPath")
        .and_then(Value::as_str)
        .or_else(|| {
            if result_count == 1 {
                record
                    .get("toolUseResult")
                    .and_then(|r| r.get("persistedOutputPath"))
                    .and_then(Value::as_str)
            } else {
                None
            }
        })
        .or_else(|| {
            text.lines().find_map(|line| {
                line.strip_prefix("Full output saved to: ")
                    .or_else(|| line.strip_prefix("Output has been saved to "))
            })
        })
}

/// Exact content-list item reference, shared by acquisition and origin proof.
pub fn required_body_path(record: &Value, index: usize) -> Option<String> {
    let items = record.get("message")?.get("content")?.as_array()?;
    let item = items.get(index)?;
    if item.get("type").and_then(Value::as_str) != Some("tool_result") {
        return None;
    }
    let count = items
        .iter()
        .filter(|i| i.get("type").and_then(Value::as_str) == Some("tool_result"))
        .count();
    let text = item.get("content").and_then(text_of).unwrap_or_default();
    body_path(record, item, count, &text).map(str::to_owned)
}

/// Source-owned inline witnesses used only to recognize an exact redundant copy.
pub fn body_references(record: &Value) -> Vec<(String, polylogue_model::BlobHash, u64)> {
    let Some(items) = record
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    else {
        return vec![];
    };
    items
        .iter()
        .enumerate()
        .filter_map(|(index, item)| {
            let path = required_body_path(record, index)?;
            let text = item.get("content").and_then(text_of).unwrap_or_default();
            Some((
                path,
                polylogue_model::BlobHash::of(text.as_bytes()),
                text.len() as u64,
            ))
        })
        .collect()
}

fn validate_structural_keys(incoming: &Value) -> Result<(), String> {
    for object in [
        incoming.as_object(),
        incoming.get("message").and_then(Value::as_object),
        incoming.get("toolUseResult").and_then(Value::as_object),
    ]
    .into_iter()
    .flatten()
    {
        if object.keys().any(|key| key.encoding().is_some()) {
            return Err("non-scalar Claude structural field names need an encoded selector".into());
        }
    }
    let message = incoming.get("message");
    let usage = message.and_then(|m| m.get("usage"));
    let mut structural = vec![
        incoming.get("origin"),
        usage,
        usage.and_then(|u| u.get("cache_creation")),
        usage.and_then(|u| u.get("output_tokens_details")),
    ];
    if let Some(content) = message
        .and_then(|m| m.get("content"))
        .and_then(Value::as_array)
    {
        structural.extend(content.iter().map(Some));
    }
    if structural
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .any(|o| o.keys().any(|k| k.encoding().is_some()))
    {
        return Err("non-scalar Claude structural field names need an encoded selector".into());
    }
    Ok(())
}
