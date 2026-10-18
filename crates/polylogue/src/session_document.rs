//! Shared normalized Session document for JSON read and export. A pinned
//! catalog is not source-complete; the caller must preserve returned coverage.
use crate::output_json::Encoder;
use polylogue_model::{
    BlockId, BlockSlot, RawJson, ToolOutcome, ToolOutput,
    merge::{BlockField, MessageField, Slot, UsageField},
};
use polylogue_store::contributions::{
    BlockFieldAddress, BlockView, FieldRead, FieldView, MessageView, SessionCoverage, SessionStep,
    SessionView, SourceStep,
};
use serde_json::{Value, json};
use std::{
    io::{self, Write},
    time::Instant,
};
const LEAF: usize = 65536;
const PIECE: usize = 4096;
const METADATA: usize = 4 * 1_048_576;
mod opaque_fields;
use opaque_fields::OpaqueFields;
#[derive(Debug)]
pub(crate) struct Coverage {
    pub up_to_date: bool,
    pub messages: u64,
    pub catalog: SessionCoverage,
    pub qualification: Value,
    pub incomplete_reasons: Vec<String>,
    pub recovery: Option<polylogue_model::RecoveryRecord>,
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
fn stored<T>(value: Result<T, polylogue_store::Error>) -> io::Result<T> {
    value.map_err(io::Error::other)
}
fn check(deadline: Instant) -> io::Result<()> {
    if Instant::now() >= deadline {
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "session document deadline exceeded",
        ))
    } else {
        Ok(())
    }
}
fn field_bytes(field: &FieldView<'_>, origin: bool) -> io::Result<Option<Vec<u8>>> {
    let range = if origin {
        stored(field.origin_range(0, LEAF))?
    } else {
        Some(stored(field.range(0, LEAF))?)
    };
    let Some(range) = range else { return Ok(None) };
    if !range.eof {
        return Err(invalid("normalized JSON leaf exceeds response admission"));
    }
    Ok(Some(range.bytes))
}
fn retained_field(field: &FieldView<'_>, origin: bool) -> io::Result<Option<RawJson>> {
    field_bytes(field, origin)?
        .map(|bytes| {
            RawJson::parse(&bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
        })
        .transpose()
}

fn field_value(field: &FieldView<'_>, origin: bool) -> io::Result<Option<Value>> {
    retained_field(field, origin)?
        .map(|value| serde_json::from_slice(&value.canonical()).map_err(io::Error::other))
        .transpose()
}

fn resolved(field: FieldRead<'_>) -> io::Result<FieldView<'_>> {
    match field {
        FieldRead::Value(value) => Ok(value),
        FieldRead::Conflict => Err(invalid(
            "unresolved field alternatives are not yet exportable",
        )),
    }
}
// Both document encoding and bounded prefix replay use this admitted typed
// field. It is normalized JSON, not a provider payload or an arbitrary blob.
fn tool_output_field(field: &FieldView<'_>) -> io::Result<ToolOutput> {
    let value = field_value(field, false)?.ok_or_else(|| invalid("missing block field"))?;
    serde_json::from_value(value).map_err(Into::into)
}
pub(crate) fn tool_output(block: &BlockView<'_>) -> io::Result<Option<ToolOutput>> {
    let slot = Slot::Block {
        block: block.slot.clone(),
        field: BlockField::Output,
    };
    match stored(block.field(&slot, LEAF))? {
        Some(field) => tool_output_field(&resolved(field)?).map(Some),
        None => Ok(None),
    }
}
fn time(value: Value) -> io::Result<Value> {
    let millis = value
        .get("at")
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid("resolved provider time lacks milliseconds"))?;
    Ok(Value::String(timestamp(millis)?))
}
fn timestamp(millis: i64) -> io::Result<String> {
    let stamp = jiff::Timestamp::from_millisecond(millis).map_err(|e| invalid(e.to_string()))?;
    Ok(format!("{stamp:.3}"))
}
fn message_name(field: MessageField) -> &'static str {
    match field {
        MessageField::PromptKind => "prompt_kind",
        MessageField::Model => "model",
        MessageField::At => "at",
        MessageField::CompletedAt => "completed_at",
        MessageField::Parent => "parent",
        MessageField::FromSession => "from_session",
        MessageField::FromName => "from_name",
        MessageField::StopReason => "stop_reason",
        MessageField::DurationMs => "duration_ms",
    }
}
fn usage_name(field: UsageField) -> &'static str {
    match field {
        UsageField::Input => "input",
        UsageField::Output => "output",
        UsageField::CacheRead => "cache_read",
        UsageField::CacheWrite => "cache_write",
        UsageField::CacheWrite5m => "cache_write_5m",
        UsageField::CacheWrite1h => "cache_write_1h",
        UsageField::Reasoning => "reasoning",
        UsageField::ServiceTier => "service_tier",
        UsageField::ProviderCostNusd => "provider_cost_usd",
    }
}
pub(crate) fn block_name(field: BlockField) -> &'static str {
    match field {
        BlockField::Text | BlockField::ThinkingText => "text",
        BlockField::Encrypted => "encrypted",
        BlockField::ThinkingData | BlockField::OtherData => "data",
        BlockField::ThinkingSignature => "signature",
        BlockField::Tool => "tool",
        BlockField::CallId => "call_id",
        BlockField::Input => "input",
        BlockField::Command => "command",
        BlockField::Outcome => "outcome",
        BlockField::OutcomeReason => "outcome_reason",
        BlockField::ExitCode => "exit_code",
        BlockField::DurationMs => "duration_ms",
        BlockField::Output => "text",
        BlockField::Name => "name",
        BlockField::MediaType => "media_type",
        BlockField::Size => "size",
        BlockField::Url => "url",
        BlockField::Direction => "direction",
        BlockField::ProducedBy => "produced_by",
    }
}
/// Provider extra containers never share a key with a typed block payload.
pub(crate) fn block_extra_container(kind: &str) -> &'static str {
    if matches!(kind, "tool_call" | "tool_result") {
        "data"
    } else {
        "attrs"
    }
}
fn member<W: Write>(e: &mut Encoder<W>, name: &str, value: &Value) -> io::Result<()> {
    e.syntax(b",")?;
    e.key(name)?;
    e.value(value)
}
/// Refuse known pending/order failures before any document bytes. This is a
/// bounded-memory pass over the same snapshot, not a source-completeness proof.
pub(crate) fn write(
    session: &SessionView,
    out: &mut impl Write,
    deadline: Instant,
) -> io::Result<Coverage> {
    write_with_cancel(session, out, deadline, &mut || false)
}
pub(crate) fn write_with_cancel(
    session: &SessionView,
    out: &mut impl Write,
    deadline: Instant,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<Coverage> {
    let source_qualification = polylogue_engine::session_status::check_with_cancel(
        session,
        deadline,
        16 * 1024 * 1024,
        cancelled,
    )
    .map_err(|error| {
        if matches!(error, polylogue_store::Error::Cancelled) {
            io::Error::new(io::ErrorKind::TimedOut, error)
        } else {
            io::Error::other(error)
        }
    })?;
    let mut qualification = source_qualification.result(&session.catalog());
    let mut incomplete_reasons = source_qualification.incomplete_reasons();
    let recovery = stored(session.recovery())?;
    if recovery.is_some() {
        incomplete_reasons.push(polylogue_model::RecoveryRecord::NOTE.into());
        qualification["outcome"] = json!("degraded");
        qualification["notes"]
            .as_array_mut()
            .expect("qualification notes")
            .push(json!(polylogue_model::RecoveryRecord::NOTE));
    }
    let mut after = None;
    loop {
        check(deadline)?;
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "Session reader cancelled",
            ));
        }
        match stored(session.next(after.as_ref(), LEAF))? {
            SessionStep::Ready { cursor, .. } => after = Some(cursor),
            SessionStep::Pending { .. } => {
                return Err(invalid("session contains a pending Message"));
            }
            SessionStep::UnresolvedOrder { .. } => {
                return Err(invalid("session Message order is unresolved"));
            }
            SessionStep::End { .. } => break,
        }
    }
    let mut e = Encoder::new(out, deadline);
    e.syntax(b"{\"format\":\"polylogue-session\",\"version\":1,\"exported_by\":\"polylogue ")?;
    // Package version is fixed build metadata, not provider text.
    e.syntax(env!("CARGO_PKG_VERSION").as_bytes())?;
    e.syntax(b"\"")?;
    member(&mut e, "result", &qualification)?;
    if let Some(recovery) = &recovery {
        member(&mut e, "recovery", &json!(recovery))?;
    }
    e.syntax(b",\"session\":{")?;
    let h = source_qualification.header(stored(session.header())?);
    e.key("id")?;
    e.reference(h.id.as_str())?;
    member(&mut e, "origin", &json!(h.origin.as_str()))?;
    e.syntax(b",")?;
    e.key("native_id")?;
    let native =
        polylogue_model::NativeId::from_wire(&h.native_id, h.native_id_encoding.as_deref())
            .map_err(|_| invalid("invalid native Session identity"))?;
    crate::output_json::check_native_identity(&native, deadline)?;
    e.identity(&h.native_id)?;
    if let Some(encoding) = &h.native_id_encoding {
        member(&mut e, "native_id_encoding", &json!(encoding))?;
    }
    member(&mut e, "kind", &serde_json::to_value(h.kind)?)?;
    for (key, value) in [
        ("title", &h.title),
        ("cwd", &h.cwd),
        ("repo", &h.repo),
        ("checkout", &h.checkout),
        ("branch", &h.branch),
        ("commit", &h.commit),
        ("harness", &h.harness),
        ("harness_version", &h.harness_version),
        ("team", &h.team),
        ("agent_name", &h.agent_name),
    ] {
        if let Some(value) = value {
            member(&mut e, key, &json!(value))?;
        }
    }
    if let Some(title) = h
        .title
        .as_deref()
        .filter(|title| !title.is_empty() && *title != "New chat")
    {
        member(&mut e, "display_title", &json!(title))?;
        member(&mut e, "title_source", &json!("provider"))?;
    }
    if let Some(models) = source_qualification
        .metadata
        .as_ref()
        .and_then(|session| session.models.as_ref())
    {
        member(&mut e, "models", &json!(models))?;
    }
    if let Some(value) = h.started_at {
        member(&mut e, "started_at", &json!(timestamp(value)?))?;
    }
    if let Some(value) = h.last_activity_at {
        member(&mut e, "last_activity_at", &json!(timestamp(value)?))?;
    }
    // Machine labels and collection metadata are not inferred from acquisition.
    if let Some(id) = &h.collection_id {
        e.syntax(b",")?;
        e.key("collection")?;
        e.syntax(b"{")?;
        e.key("id")?;
        e.identity(id)?;
        if let Some(name) = &h.collection_name {
            member(&mut e, "name", &json!(name))?;
        }
        e.syntax(b"}")?;
    }
    if let Some(attrs) = &h.attrs {
        e.syntax(b",")?;
        e.key("attrs")?;
        e.raw_json(attrs, false)?;
    }
    e.syntax(b"},\"messages\":[")?;
    let mut after = None;
    let mut messages = 0u64;
    let catalog = loop {
        check(deadline)?;
        if cancelled() {
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "Session reader cancelled",
            ));
        }
        match stored(session.next(after.as_ref(), LEAF))? {
            SessionStep::Ready {
                cursor, message, ..
            } => {
                if messages != 0 {
                    e.syntax(b",")?;
                }
                write_message(
                    &message,
                    &mut e,
                    deadline,
                    source_qualification
                        .active_messages
                        .contains(message.id().as_str()),
                    cancelled,
                )?;
                messages = messages
                    .checked_add(1)
                    .ok_or_else(|| invalid("Message count overflow"))?;
                after = Some(cursor);
            }
            SessionStep::Pending { .. } | SessionStep::UnresolvedOrder { .. } => {
                return Err(invalid("pinned session catalog changed shape"));
            }
            SessionStep::End { coverage } => break coverage,
        }
    };
    e.syntax(b"]")?;
    if source_qualification.complete() {
        e.syntax(b",\"links\":[],\"events\":[],\"notes\":[],\"runs\":[],\"deliveries\":[]")?;
    }
    e.syntax(b",\"inputs\":")?;
    e.value(&json!(source_qualification.inputs))?;
    e.syntax(b"}")?;
    e.finish()?;
    Ok(Coverage {
        up_to_date: source_qualification.up_to_date,
        messages,
        catalog,
        qualification,
        incomplete_reasons,
        recovery,
    })
}
fn write_message<W: Write>(
    m: &MessageView,
    e: &mut Encoder<W>,
    deadline: Instant,
    active: bool,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    let header = stored(m.header())?;
    e.syntax(b"{")?;
    e.key("id")?;
    e.reference(m.id().as_str())?;
    member(
        e,
        "version_hash",
        &json!(stored(m.version_hash())?.to_string()),
    )?;
    member(e, "role", &serde_json::to_value(header.role)?)?;
    member(e, "written_by", &serde_json::to_value(header.written_by)?)?;
    if active {
        member(e, "on_active_path", &json!(true))?;
    }
    if let Some(progress) = header.in_progress {
        member(e, "in_progress", &json!(progress))?;
    }
    write_message_fields(m, e, deadline, false, cancelled)?;
    e.syntax(b",\"blocks\":[")?;
    let mut index = 0u64;
    while let Some(block) = stored(m.block_at(index))? {
        check(deadline)?;
        if index != 0 {
            e.syntax(b",")?;
        }
        write_block(m, &block, e, deadline, cancelled)?;
        index = index
            .checked_add(1)
            .ok_or_else(|| invalid("block count overflow"))?;
    }
    e.syntax(b"]")?;
    e.message_boundary()?;
    write_origins(m, e, deadline)?;
    e.syntax(b",\"sources\":[")?;
    let mut cursor = None;
    let mut first = true;
    loop {
        check(deadline)?;
        match stored(m.next_source(cursor.as_ref(), LEAF))? {
            SourceStep::Span {
                cursor: next,
                source,
            } => {
                if !first {
                    e.syntax(b",")?;
                }
                first = false;
                e.syntax(b"{")?;
                e.key("input")?;
                e.identity(&source.input)?;
                member(e, "chain", &json!(source.chain))?;
                member(e, "start", &json!(source.start))?;
                member(e, "end", &json!(source.end))?;
                if source.preferred {
                    member(e, "preferred", &json!(true))?;
                }
                e.syntax(b"}")?;
                cursor = Some(next);
            }
            SourceStep::Continue { cursor: next } => cursor = Some(next),
            SourceStep::End => break,
        }
    }
    e.syntax(b"]}")
}
fn write_message_fields<W: Write>(
    m: &MessageView,
    e: &mut Encoder<W>,
    deadline: Instant,
    model_page: bool,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    let mut cursor = None;
    while let Some(entry) = stored(m.next_field(cursor.as_ref(), LEAF))? {
        check(deadline)?;
        if let Slot::Message(kind) = entry.slot {
            let field = resolved(entry.value)?;
            let mut value = field_value(&field, false)?
                .ok_or_else(|| invalid("missing selected field value"))?;
            if matches!(kind, MessageField::At | MessageField::CompletedAt) {
                value = time(value)?;
            }
            e.syntax(b",")?;
            e.key(message_name(kind))?;
            if matches!(kind, MessageField::Parent | MessageField::FromSession) {
                let id = value
                    .as_str()
                    .or_else(|| value.get("id").and_then(Value::as_str))
                    .ok_or_else(|| invalid("resolved reference lacks opaque identity"))?;
                e.reference(id)?;
            } else {
                e.value(&value)?;
            }
        }
        cursor = Some(entry.cursor);
    }
    for extras in [false, true] {
        if extras && !model_page {
            let mut fields = ordered(m.ordered_extras(LEAF, deadline, cancelled))?;
            write_ordered_extras(&mut fields, "attrs", e, deadline, cancelled)?;
            continue;
        }
        let mut opaque = OpaqueFields::new("attrs", model_page);
        let mut cursor = None;
        let mut first = true;
        while let Some(entry) = stored(m.next_field(cursor.as_ref(), LEAF))? {
            check(deadline)?;
            let key = match &entry.slot {
                Slot::Usage(kind) if !extras => Some(usage_name(*kind)),
                Slot::Extra { block: None, key } if extras => Some(key.as_str()),
                _ => None,
            };
            if let Some(key) = key {
                let field = resolved(entry.value)?;
                if extras {
                    let bytes = field_bytes(&field, false)?
                        .ok_or_else(|| invalid("missing selected opaque value"))?;
                    opaque.push(e, key, &bytes)?;
                    cursor = Some(entry.cursor);
                    continue;
                }
                let mut value = field_value(&field, false)?
                    .ok_or_else(|| invalid("missing selected field value"))?;
                if matches!(entry.slot, Slot::Usage(UsageField::ProviderCostNusd)) {
                    let nanos = value
                        .as_i64()
                        .ok_or_else(|| invalid("provider cost is not integer nanos"))?;
                    let whole = nanos / 1_000_000_000;
                    let fraction = nanos.unsigned_abs() % 1_000_000_000;
                    let text = if nanos < 0 && whole == 0 {
                        format!("-0.{fraction:09}")
                    } else {
                        format!("{whole}.{fraction:09}")
                    };
                    value = serde_json::from_str(&text)?;
                }
                if first {
                    e.syntax(b",")?;
                    e.key(if extras { "attrs" } else { "usage" })?;
                    e.syntax(b"{")?;
                    first = false;
                } else {
                    e.syntax(b",")?;
                }
                e.key(key)?;
                e.value(&value)?;
            }
            cursor = Some(entry.cursor);
        }
        if !first {
            e.syntax(b"}")?;
        }
        opaque.finish(e)?;
    }
    Ok(())
}

fn opaque_value<W: Write>(
    e: &mut Encoder<W>,
    value: &RawJson,
    _deadline: Instant,
    model_page: bool,
) -> io::Result<()> {
    if model_page {
        e.opaque(value)
    } else {
        e.raw_json(value, false)
    }
}

fn opaque_field<W: Write>(
    field: &FieldView<'_>,
    e: &mut Encoder<W>,
    deadline: Instant,
    model_page: bool,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    if !model_page {
        let first = stored(field.range(0, LEAF))?;
        if first.eof {
            let value = RawJson::parse(&first.bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            return opaque_value(e, &value, deadline, false);
        }
        if first.bytes.first() == Some(&b'"') {
            return e.string_ranges(
                |offset| {
                    let part = stored(field.range(offset, PIECE))?;
                    if part.next_offset != offset.saturating_add(part.bytes.len() as u64) {
                        return Err(invalid("normalized string range made invalid progress"));
                    }
                    Ok((part.bytes, part.eof))
                },
                cancelled,
            );
        }
        return Err(invalid("normalized JSON leaf exceeds response admission"));
    }
    let value =
        retained_field(field, false)?.ok_or_else(|| invalid("missing opaque block field"))?;
    opaque_value(e, &value, deadline, model_page)
}

fn write_block<W: Write>(
    m: &MessageView,
    b: &BlockView<'_>,
    e: &mut Encoder<W>,
    deadline: Instant,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    let id = match &b.slot {
        BlockSlot::Item(i) => BlockId::new(m.id().clone(), *i),
        BlockSlot::Reasoning(i) => BlockId::reasoning(m.id().clone(), *i),
        BlockSlot::Attachment(native) => {
            BlockId::attachment(m.id().clone(), native).map_err(|x| invalid(x.to_string()))?
        }
    };
    e.syntax(b"{")?;
    e.key("id")?;
    e.reference(id.as_str())?;
    member(e, "kind", &json!(&b.kind))?;
    if matches!(b.kind.as_str(), "text" | "thinking") {
        if stored(b.text_range(0, PIECE))?.is_some() {
            e.syntax(b",")?;
            e.key("text")?;
            let mut offset = 0;
            let mut done = false;
            e.body_ranges(|| {
                if done {
                    return Ok(None);
                }
                let range = stored(b.text_range(offset, PIECE))?
                    .ok_or_else(|| invalid("selected text disappeared"))?;
                if range.next_offset < offset || (!range.eof && range.next_offset == offset) {
                    return Err(invalid("selected text made no progress"));
                }
                offset = range.next_offset;
                done = range.eof;
                if range.bytes.is_empty() && done {
                    Ok(None)
                } else {
                    Ok(Some(range.bytes))
                }
            })?;
        }
        if let Some(encrypted) = b.encrypted {
            member(e, "encrypted", &json!(encrypted))?;
        }
    }
    write_block_fields(b, e, deadline, true, false, cancelled)?;
    e.syntax(b"}")
}
fn write_block_fields<W: Write>(
    b: &BlockView<'_>,
    e: &mut Encoder<W>,
    deadline: Instant,
    include_output: bool,
    model_page: bool,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    if b.kind == "tool_result" {
        let result =
            stored(b.tool_result(LEAF))?.ok_or_else(|| invalid("tool result header absent"))?;
        member(e, "outcome", &json!(result.outcome.as_str()))?;
        // Reason vocabulary/unknown status comes from the typed immutable body.
        match &result.outcome {
            ToolOutcome::Unknown { reason } => {
                member(e, "outcome_reason", &json!(reason.as_str()))?
            }
            ToolOutcome::Error {
                reason: Some(reason),
            } => member(e, "outcome_reason", &json!(reason.code()))?,
            _ => {}
        }
        if let Some(code) = result.exit_code {
            member(e, "exit_code", &json!(code))?;
        }
        if let Some(duration) = result.duration_ms {
            member(e, "duration_ms", &json!(duration))?;
        }
    }
    let mut cursor = None;
    while let Some(entry) = stored(b.next_field(cursor.as_ref(), LEAF))? {
        check(deadline)?;
        if let BlockFieldAddress::Field(kind) = entry.address {
            let field = resolved(entry.value)?;
            let body_scalar = matches!(
                kind,
                BlockField::Text | BlockField::ThinkingText | BlockField::Encrypted
            );
            let result_scalar = b.kind == "tool_result"
                && matches!(
                    kind,
                    BlockField::Outcome
                        | BlockField::OutcomeReason
                        | BlockField::ExitCode
                        | BlockField::DurationMs
                );
            if !include_output && kind == BlockField::Output {
                if let ToolOutput::Blob(hash) = tool_output_field(&field)? {
                    member(e, "blob", &json!(format!("blake3:{hash}")))?;
                }
            }
            if !body_scalar && !result_scalar && (include_output || kind != BlockField::Output) {
                if kind == BlockField::Output {
                    let output = tool_output_field(&field)?;
                    match output {
                        ToolOutput::Text(text) => {
                            e.syntax(b",")?;
                            e.key("text")?;
                            e.body_text(&text)?;
                        }
                        ToolOutput::Blob(hash) => {
                            member(e, "blob", &json!(format!("blake3:{hash}")))?
                        }
                    }
                } else {
                    if matches!(
                        kind,
                        BlockField::Input | BlockField::ThinkingData | BlockField::OtherData
                    ) {
                        e.syntax(b",")?;
                        e.key(block_name(kind))?;
                        opaque_field(&field, e, deadline, model_page, cancelled)?;
                        cursor = Some(entry.cursor);
                        continue;
                    }
                    let value = field_value(&field, false)?
                        .ok_or_else(|| invalid("missing block field"))?;
                    e.syntax(b",")?;
                    e.key(block_name(kind))?;
                    if kind == BlockField::CallId {
                        let id: polylogue_model::NativeId = serde_json::from_value(value)
                            .map_err(|_| invalid("invalid native call identity"))?;
                        e.native_id(&id)?;
                    } else if kind == BlockField::ProducedBy {
                        let id = value
                            .as_str()
                            .or_else(|| value.get("id").and_then(Value::as_str))
                            .ok_or_else(|| invalid("block reference lacks identity"))?;
                        e.reference(id)?;
                    } else {
                        e.value(&value)?;
                    }
                }
            }
        }
        cursor = Some(entry.cursor);
    }
    let container = block_extra_container(&b.kind);
    if !model_page {
        let mut fields = ordered(b.ordered_extras(LEAF, deadline, cancelled))?;
        return write_ordered_extras(&mut fields, container, e, deadline, cancelled);
    }
    let mut cursor = None;
    let mut opaque = OpaqueFields::new(container, model_page);
    while let Some(entry) = stored(b.next_field(cursor.as_ref(), LEAF))? {
        check(deadline)?;
        if let BlockFieldAddress::Extra(key) = entry.address {
            let bytes = field_bytes(&resolved(entry.value)?, false)?
                .ok_or_else(|| invalid("missing block extra"))?;
            opaque.push(e, &key, &bytes)?;
        }
        cursor = Some(entry.cursor);
    }
    opaque.finish(e)?;
    Ok(())
}
/// Finite selected metadata, encoded with exactly the document's logical-value
/// rules. Body text is excluded and remains owned by the bounded body window.
struct Metadata(Vec<u8>);
impl Write for Metadata {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > METADATA.saturating_sub(self.0.len()) {
            return Err(invalid("selected metadata exceeds admission"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn metadata(
    write: impl FnOnce(&mut Encoder<Metadata>) -> io::Result<()>,
    deadline: Instant,
) -> io::Result<Value> {
    let mut e = Encoder::new(Metadata(Vec::new()), deadline);
    e.syntax(b"{\"_\":null")?;
    write(&mut e)?;
    e.syntax(b"}")?;
    let mut value: Value = serde_json::from_slice(&e.finish()?.0)?;
    value
        .as_object_mut()
        .expect("encoded metadata object")
        .remove("_");
    Ok(value)
}
pub(crate) fn message_metadata(m: &MessageView, deadline: Instant) -> io::Result<Value> {
    metadata(
        |e| write_message_fields(m, e, deadline, true, &mut || false),
        deadline,
    )
}
pub(crate) fn block_metadata(b: &BlockView<'_>, deadline: Instant) -> io::Result<Value> {
    metadata(
        |e| write_block_fields(b, e, deadline, false, true, &mut || false),
        deadline,
    )
}

fn write_origins<W: Write>(
    m: &MessageView,
    e: &mut Encoder<W>,
    deadline: Instant,
) -> io::Result<()> {
    let mut first = true;
    let mut cursor = None;
    while let Some(entry) = stored(m.next_field(cursor.as_ref(), LEAF))? {
        check(deadline)?;
        let name = match &entry.slot {
            Slot::Message(kind) => message_name(*kind).to_owned(),
            Slot::Usage(kind) => format!("usage.{}", usage_name(*kind)),
            Slot::Extra { block: None, key } => format!("attrs.{key}"),
            _ => return Err(invalid("message root contains a block slot")),
        };
        origin(e, &resolved(entry.value)?, &name, &mut first)?;
        cursor = Some(entry.cursor);
    }
    let mut index = 0u64;
    while let Some(b) = stored(m.block_at(index))? {
        let mut cursor = None;
        while let Some(entry) = stored(b.next_field(cursor.as_ref(), LEAF))? {
            check(deadline)?;
            let name = match &entry.address {
                BlockFieldAddress::Field(kind) => format!("blocks[{index}].{}", block_name(*kind)),
                BlockFieldAddress::Extra(key) => {
                    format!("blocks[{index}].{}.{key}", block_extra_container(&b.kind))
                }
            };
            origin(e, &resolved(entry.value)?, &name, &mut first)?;
            cursor = Some(entry.cursor);
        }
        index = index
            .checked_add(1)
            .ok_or_else(|| invalid("block count overflow"))?;
    }
    if !first {
        e.syntax(b"]")?;
    }
    Ok(())
}
fn origin<W: Write>(
    e: &mut Encoder<W>,
    field: &FieldView<'_>,
    name: &str,
    first: &mut bool,
) -> io::Result<()> {
    if let Some(value) = retained_field(field, true)? {
        if *first {
            e.syntax(b",\"field_origins\":[")?;
            *first = false;
        } else {
            e.syntax(b",")?;
        }
        e.syntax(b"{")?;
        e.key("field")?;
        e.identity(name)?;
        e.syntax(b",")?;
        e.key("origin")?;
        e.origin(&value)?;
        e.syntax(b"}")?;
    }
    Ok(())
}
fn ordered<T>(result: Result<T, polylogue_store::Error>) -> io::Result<T> {
    result.map_err(|error| {
        if matches!(error, polylogue_store::Error::Cancelled) {
            io::Error::new(io::ErrorKind::TimedOut, error)
        } else {
            io::Error::other(error)
        }
    })
}
fn write_ordered_extras<W: Write>(
    fields: &mut polylogue_store::contributions::OrderedExtras<'_>,
    name: &'static str,
    e: &mut Encoder<W>,
    deadline: Instant,
    cancelled: &mut dyn FnMut() -> bool,
) -> io::Result<()> {
    let mut opaque = OpaqueFields::new(name, false);
    while let Some(entry) = ordered(fields.next(deadline, cancelled))? {
        let bytes = field_bytes(&resolved(entry.value)?, false)?
            .ok_or_else(|| invalid("missing selected opaque value"))?;
        opaque.push(e, &entry.key, &bytes)?;
    }
    opaque.finish(e)
}

#[cfg(test)]
mod tests;
