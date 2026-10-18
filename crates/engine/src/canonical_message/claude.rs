//! Claude cursors and source selection. Writes and value/origin checks are in
//! the parent module, shared with Codex.
use super::*;
use polylogue_store::parser_state::Phase;
mod parent;
use crate::state::claude::View;
use polylogue_model::{
    merge::{BlockField, MessageField},
    version_evidence::Framer,
};
use polylogue_sources::claude_code::{
    self as source,
    keyed::{Contribution, PendingMessage, ReadView, ResponseCell},
    projection::{Fact, Field},
    selection::AuthorLeaf,
};

#[derive(Clone, Debug)]
pub struct ClaudeRequest {
    pub pending: PendingMessage,
    pub source_binding: BlobHash,
    pub context: Vec<String>,
    /// Archive revision; `pending.revision` is the provider evidence revision.
    pub revision: u64,
    pub logical_key: [u8; 32],
}
#[derive(Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct Head {
    pub pending: PendingMessage,
    context: Vec<String>,
    parser_version: u32,
}
/// A complete Message before a waiting tail can be reused by its current
/// source certificate, never by equal text, counts or logical IDs.
pub(crate) fn current_projection(
    projection: &SourceProjection,
    token: &Token,
    pending: &PendingMessage,
    source_binding: BlobHash,
    context: &[String],
) -> Result<bool> {
    if projection.predecessor() != token
        || projection.message() != &pending.message
        || pending.incomplete
    {
        return Ok(false);
    }
    let guard: EvidenceGuard = serde_json::from_slice(projection.evidence_guard())
        .map_err(|e| Error::Source(e.to_string()))?;
    Ok(guard
        == EvidenceGuard {
            head: ProviderHead::Claude(Head {
                pending: pending.clone(),
                context: context.to_vec(),
                parser_version: source::PARSER_VERSION,
            }),
            source_binding,
        })
}
pub(super) struct Digest {
    token: Token,
    head: Head,
    source_binding: BlobHash,
    digest: BlobHash,
}
impl Digest {
    pub(super) fn validate(&self, h: &Handle) -> Result<BlobHash> {
        if self.token != *h.build.token()
            || h.head != ProviderHead::Claude(self.head.clone())
            || self.source_binding != h.request.source_binding
        {
            return Err(Error::Invariant("Claude evidence digest inputs changed"));
        }
        Ok(self.digest)
    }
}
fn callback<T>(view: &View<'_>, result: std::result::Result<T, String>) -> Result<T> {
    if let Some(e) = view.take_error() {
        return Err(e.into());
    }
    result.map_err(Error::Source)
}
fn response(view: &View<'_>, head: &Head) -> Result<ResponseCell> {
    let p = &head.pending;
    if view.pending(&p.message)?.as_ref() != Some(p) {
        return Err(Error::Unsupported("Claude pending work locator changed"));
    }
    let r = callback(view, view.response(&p.session_native, &p.response))?
        .ok_or(Error::Unsupported("Claude response unavailable"))?;
    if r.message.id != p.message
        || r.ordinal != p.ordinal
        || r.revision != p.revision
        || r.items != p.items
        || r.incomplete_bodies != 0
        || p.incomplete
    {
        return Err(Error::Unsupported(
            "Claude pending locator no longer matches complete source evidence",
        ));
    }
    Ok(r)
}
fn contribution(
    snapshot: &Snapshot,
    head: &Head,
    item: u64,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<parent::ResolvedContribution> {
    let view = View::new(snapshot, b.read)?;
    let mut c = callback(&view, view.contribution(&head.pending.message, item))?
        .ok_or(Error::Unsupported("Claude contribution unavailable"))?;
    let records = if item == 0 {
        parent::resolve(&view, head, &mut c, b.metadata, flag)?
    } else {
        vec![]
    };
    if !c.projection.unresolved.is_empty() {
        return Err(Error::Unsupported(
            "Claude source mapping/counting obligation remains pending",
        ));
    }
    if c.bodies.iter().any(|body| body.payload.is_none()) {
        return Err(Error::Unsupported(
            "Claude required companion remains unavailable",
        ));
    }
    Ok(parent::ResolvedContribution {
        contribution: c,
        records,
    })
}
pub fn start(
    snapshot: &Snapshot,
    request: &ClaudeRequest,
    current: Option<SourceProjection>,
    budget: Budget,
    allowance: usize,
) -> Result<Action> {
    if !matches!(snapshot.head.status, Phase::Finalizing | Phase::WaitingTail)
        || !snapshot.head.finalization_pending
    {
        return Err(Error::Unsupported(
            "Claude materializer requires a committed complete-record source head",
        ));
    }
    let view = View::new(snapshot, budget)?;
    let head = Head {
        pending: request.pending.clone(),
        context: request.context.clone(),
        parser_version: source::PARSER_VERSION,
    };
    let r = response(&view, &head)?;
    let session = callback(&view, view.session(&head.pending.session_native))?
        .ok_or(Error::Unsupported("Claude source Session unavailable"))?;
    if r.message.id.session() != &session.session.id {
        return Err(Error::Invariant("Claude Message/Session relation mismatch"));
    }
    let common = Request {
        message: usize::try_from(request.pending.ordinal)
            .map_err(|_| Error::Invariant("Claude ordinal overflow"))?,
        revision: request.revision,
        logical_key: request.logical_key,
        source_binding: request.source_binding,
    };
    let basis = ProviderHead::Claude(head);
    let typed = EvidenceGuard {
        head: basis.clone(),
        source_binding: request.source_binding,
    };
    let guard = bounded_json(&typed, allowance)?;
    if let Some(previous) = &current {
        if previous.message() != &r.message.id {
            return Err(Error::Invariant(
                "Claude current reference names another Message",
            ));
        }
        let prior: EvidenceGuard = serde_json::from_slice(previous.evidence_guard())
            .map_err(|e| Error::Source(e.to_string()))?;
        if previous.predecessor() == &snapshot.head.token {
            if prior != typed {
                return Err(Error::Invariant("same-token Claude guard changed"));
            }
            return Ok(Action {
                inner: ActionKind::Existing {
                    token: snapshot.head.token.clone(),
                    request: common,
                    source: previous.clone(),
                    allowance,
                },
            });
        }
    }
    Ok(Action {
        inner: ActionKind::Prepare {
            identity: IdentityOwned {
                session: session.session.id,
                origin: session.session.origin,
                native: session.session.native_id,
                native_encoding: session.session.native_id_encoding,
                kind: session.session.kind,
                message: r.message.id,
                role: r.message.role,
                by: r.message.written_by,
            },
            request: common,
            token: snapshot.head.token.clone(),
            head: basis,
            guard,
            previous: current,
            allowance,
        },
    })
}
#[derive(Clone, Serialize, Deserialize)]
pub(super) enum Cursor {
    ParentCitations {
        index: usize,
        donor: u8,
    },
    Citations {
        item: u64,
        after: Option<Vec<String>>,
        envelope: bool,
    },
    CompletionCitation {
        item: u64,
    },
    SelectedCitations {
        index: u8,
    },
    RecordExtraCitations {
        after: Option<String>,
    },
    AuthorCitation,
    Layout,
    Author {
        index: u8,
    },
    MessageFields {
        item: u64,
        index: usize,
    },
    Block {
        item: u64,
        local: u32,
        phase: BlockPhase,
    },
    Selected {
        index: u8,
        fact: usize,
    },
    RecordExtras {
        after: Option<String>,
    },
    Finish,
}
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(super) enum BlockPhase {
    Begin,
    Command,
    Fields(usize),
    Retained(usize),
    Signature,
    Install,
}
fn stage(cursor: &mut super::Cursor, next: Cursor) {
    cursor.stage = Stage::Claude(next)
}
fn progress(state: &State, cursor: &mut super::Cursor, next: Cursor, b: Budgets) -> Result<Action> {
    stage(cursor, next);
    advance(state, cursor, b)
}
fn cite(
    state: &State,
    cursor: &mut super::Cursor,
    next: Cursor,
    range: polylogue_sources::selection::RecordRange,
    b: Budgets,
) -> Result<Action> {
    stage(cursor, next);
    Ok(action(
        state,
        b,
        Change::SourceRange {
            range: cited(range),
            next: bounded_json(cursor, b.metadata)?,
        },
    ))
}
fn slot(field: &Field) -> Result<Slot> {
    Ok(match field {
        Field::FirstMessage(f) | Field::SelectedMessage(f) => Slot::Message(*f),
        Field::Usage(f) => Slot::Usage(*f),
        Field::SelectedExtra { key } => Slot::Extra {
            block: None,
            key: key.clone(),
        },
        Field::Block { local, field } => Slot::Block {
            block: BlockSlot::Item(*local),
            field: *field,
        },
        Field::ResultData { local, key } => Slot::Extra {
            block: Some(BlockSlot::Item(*local)),
            key: key.clone(),
        },
        Field::Extra { local, path } => Slot::Extra {
            block: local.map(BlockSlot::Item),
            key: path
                .last()
                .ok_or(Error::Unsupported("Claude extra key unavailable"))?
                .clone(),
        },
    })
}
fn local(field: &Field) -> Option<u32> {
    match field {
        Field::Block { local, .. } | Field::ResultData { local, .. } => Some(*local),
        Field::Extra { local, .. } => *local,
        _ => None,
    }
}
fn is_message(field: &Field, item: u64) -> bool {
    match field {
        Field::FirstMessage(_) => item == 0,
        // Top-level record extras have one selected response. Unknown nested
        // paths are still unsupported.
        Field::Extra { local: None, path } => path.len() != 1,
        _ => false,
    }
}
fn field<'a>(fact: &'a Fact, slot: &'a Slot, supplied: bool) -> Result<FieldDescriptor<'a>> {
    Ok(FieldDescriptor {
        slot,
        time: None,
        supplied,
        selection: FieldSelector::Claude(
            fact.selection
                .as_ref()
                .ok_or(Error::Unsupported("Claude field selector unavailable"))?,
        ),
    })
}
fn leaf<'a>(
    fact: &Fact,
    slot: &'a Slot,
    r: &'a ResponseCell,
    c: Option<&'a Contribution>,
    content: Option<&'a BlockContent>,
) -> Result<ValueSource<'a>> {
    if matches!(fact.field, Field::FirstMessage(MessageField::Parent))
        && let Some(recipe) = &fact.selection
    {
        let resolved = match &recipe.transform {
            source::selection::Transform::ParentReferenceV1 { resolved, .. }
            | source::selection::Transform::ParentViaRecordsV1 { resolved, .. } => Some(resolved),
            _ => None,
        };
        if let Some(resolved) = resolved {
            let mut bytes = Vec::new();
            polylogue_model::source_leaf::write_message_reference(resolved, &mut bytes)
                .map_err(|e| Error::Source(e.to_string()))?;
            return Ok(ValueSource::Opaque(
                RawJson::parse(&bytes)
                    .map_err(|e| Error::Source(e.to_string()))?
                    .into_opaque(),
            ));
        }
    }
    let subject = match &fact.field {
        Field::FirstMessage(_)
        | Field::SelectedMessage(_)
        | Field::Usage(_)
        | Field::SelectedExtra { .. } => Subject::Message(&r.message),
        Field::ResultData { local, .. } => {
            let c = c.ok_or(Error::Invariant(
                "Claude structured result lost its contribution",
            ))?;
            let BlockContent::ToolResult(result) = c
                .blocks
                .get(*local as usize)
                .ok_or(Error::Invariant("Claude result block unavailable"))?
            else {
                return Err(Error::Invariant("Claude result field on non-result block"));
            };
            let Slot::Extra { block, .. } = slot else {
                return Err(Error::Invariant("Claude structured data slot changed"));
            };
            Subject::Extras {
                block: block.as_ref(),
                values: result.data.as_ref().ok_or(Error::Unsupported(
                    "Claude structured result data unavailable",
                ))?,
            }
        }
        Field::Extra { local: None, path } => {
            // A top-level extra has an unambiguous attrs key. Nested paths could
            // collide with literal keys or normalized extras when flattened, so
            // they stay unsupported.
            let [key] = path.as_slice() else {
                return Err(Error::UnsupportedField {
                    path: path.clone(),
                    reason: "Claude nested message extra normalization remains pending",
                });
            };
            if r.message
                .attrs
                .as_ref()
                .is_some_and(|attrs| attrs.get(key).is_some())
            {
                return Err(Error::UnsupportedField {
                    path: path.clone(),
                    reason: "Claude immediate extra conflicts with a selected attrs key",
                });
            }
            let c = c.ok_or(Error::Invariant("Claude extra lost its contribution"))?;
            let raw = RawJson::parse(&c.raw).map_err(|e| Error::Source(e.to_string()))?;
            let value = raw
                .get(key)
                .ok_or(Error::Invariant("Claude extra lost its value"))?;
            return Ok(ValueSource::Opaque(value.clone().into_opaque()));
        }
        Field::Extra {
            local: Some(local),
            path,
        } => {
            // Top-level block member only; nested paths would collide in the
            // block's extra-key namespace.
            let [message, content, index, key] = path.as_slice() else {
                return Err(Error::UnsupportedField {
                    path: path.clone(),
                    reason: "Claude nested block extra remains unsupported",
                });
            };
            if message != "message"
                || content != "content"
                || index.parse::<u32>().ok() != Some(*local)
            {
                return Err(Error::UnsupportedField {
                    path: path.clone(),
                    reason: "Claude block extra address is ambiguous",
                });
            }
            let c = c.ok_or(Error::Invariant("Claude block extra lost its contribution"))?;
            if c.projection.facts.iter().any(|fact| matches!(&fact.field,
                Field::ResultData {local: found, key: selected} if found == local && selected == key)) {
                return Err(Error::UnsupportedField { path: path.clone(), reason: "Claude block extra conflicts with selected structured data" });
            }
            let raw = RawJson::parse(&c.raw).map_err(|e| Error::Source(e.to_string()))?;
            let value = raw
                .get("message")
                .and_then(|v| v.get("content"))
                .and_then(RawJson::as_array)
                .and_then(|v| v.get(*local as usize))
                .and_then(|v| v.get(key))
                .ok_or(Error::Invariant("Claude block extra lost its raw value"))?;
            return Ok(ValueSource::Opaque(value.clone().into_opaque()));
        }
        Field::Block { .. } => {
            let Slot::Block { block, .. } = slot else {
                return Err(Error::Invariant("Claude block semantic slot changed"));
            };
            Subject::Block {
                slot: block,
                content: content.ok_or(Error::Invariant("Claude normalized block unavailable"))?,
            }
        }
    };
    Ok(ValueSource::Leaf { subject, slot })
}

// The map is already charged by View. Only adjacent equal ranges merge, so
// correction suppliers A/B/A stay three groups. The stored cursor names the
// last covered path, also when resuming a cursor written before merging.
fn consecutive_origins(
    origins: &source::origins::RecordOrigins,
    after: Option<&[String]>,
    range: polylogue_sources::selection::RecordRange,
    flag: &AtomicBool,
) -> Result<Option<Vec<String>>> {
    let mut last = after.map(<[String]>::to_vec);
    while let Some((key, origin)) = origins.next(last.as_deref()) {
        cancelled(flag)?;
        if origin.record != range {
            break;
        }
        last = Some(key.to_vec());
    }
    Ok(last)
}

// Skips absent slots in the ResponseCell already read, at most eight per
// call, with no provider reads.
fn selected_index(r: &ResponseCell, mut index: u8, flag: &AtomicBool) -> Result<u8> {
    while index < 8 {
        cancelled(flag)?;
        let absent = if index < 6 {
            let i = usize::from(index);
            r.selected[i].is_none()
                && r.selected_origins[i].is_none()
                && r.selected_from[i].is_none()
                && r.selected_facts[i].is_empty()
        } else if index == 6 {
            r.stop_sequence.is_none()
        } else {
            r.stop_details.is_none()
        };
        if !absent {
            break;
        }
        index += 1;
    }
    Ok(index)
}

pub(super) fn compute(
    snapshot: &Snapshot,
    reader: &ReadHandle,
    state: &State,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<Action> {
    let ProviderHead::Claude(head) = &state.handle.head else {
        return Err(Error::Invariant(
            "Claude Message builder has another provider head",
        ));
    };
    let view = View::new(snapshot, b.read)?;
    let r = response(&view, head)?;
    let input = snapshot.message_source_input_id(&state.handle.build, b.metadata)?;
    let retained = reader
        .source(&input)?
        .ok_or(Error::Unsupported("Claude input unavailable"))?;
    if retained.binding != state.handle.request.source_binding
        || retained.input.chain != snapshot.head.token.chain
        || retained.context != head.context
    {
        return Err(Error::Invariant(
            "source binding or context differs from Claude input",
        ));
    }
    if let Some(prepared) = &state.preparation.projection {
        return Ok(action(state, b, Change::Publish(prepared.clone())));
    }
    let mut cursor: super::Cursor = match &state.projection.cursor {
        Some(bytes) => serde_json::from_slice(bytes).map_err(|e| Error::Source(e.to_string()))?,
        None if state.projection.sequence == 0 => super::Cursor {
            source: None,
            stage: Stage::Claude(Cursor::Citations {
                item: 0,
                after: None,
                envelope: false,
            }),
            verified: None,
        },
        _ => return Err(Error::Invariant("Claude materializer lost durable cursor")),
    };
    let Stage::Claude(position) = cursor.stage.clone() else {
        return Err(Error::Invariant(
            "Claude materializer found another provider cursor",
        ));
    };
    let owns_origin = matches!(
        position,
        Cursor::Author { .. }
            | Cursor::MessageFields { .. }
            | Cursor::Selected { .. }
            | Cursor::RecordExtras { .. }
            | Cursor::Block {
                phase: BlockPhase::Fields(_) | BlockPhase::Retained(_) | BlockPhase::Signature,
                ..
            }
    );
    if state.preparation.active_origin_value.is_some()
        && (!owns_origin || cursor.verified.is_none())
    {
        return Err(Error::Unsupported(
            "Claude origin producer has no verified owning field",
        ));
    }
    match position {
        Cursor::Citations {
            item,
            after,
            envelope,
        } => {
            if item >= r.items {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::SelectedCitations { index: 0 },
                    b,
                );
            }
            let c = contribution(snapshot, head, item, b, flag)?;
            if !envelope {
                let range = c
                    .origins
                    .envelope
                    .ok_or(Error::Unsupported("Claude record provenance unavailable"))?;
                let after = consecutive_origins(&c.origins, None, range, flag)?;
                return cite(
                    state,
                    &mut cursor,
                    Cursor::Citations {
                        item,
                        after,
                        envelope: true,
                    },
                    range,
                    b,
                );
            }
            let previous = match after.as_ref() {
                Some(key) => c.origins.fields.get(key).map(|o| o.record),
                None => c.origins.envelope,
            };
            let after = match previous {
                Some(range) => consecutive_origins(&c.origins, after.as_deref(), range, flag)?,
                None => after,
            };
            if let Some((key, origin)) = c.origins.next(after.as_deref()) {
                return cite(
                    state,
                    &mut cursor,
                    Cursor::Citations {
                        item,
                        after: consecutive_origins(&c.origins, Some(key), origin.record, flag)?,
                        envelope: true,
                    },
                    origin.record,
                    b,
                );
            }
            progress(state, &mut cursor, Cursor::CompletionCitation { item }, b)
        }
        Cursor::CompletionCitation { item } => {
            let c = contribution(snapshot, head, item, b, flag)?;
            let next = Cursor::Citations {
                item: item
                    .checked_add(1)
                    .ok_or(Error::Invariant("Claude contribution cursor overflow"))?,
                after: None,
                envelope: false,
            };
            // The completion supplies fields of the earlier launch Message.
            // Cite only that record before evaluating them.
            match &c.completion {
                Some(completion) => cite(state, &mut cursor, next, completion.range, b),
                None => progress(state, &mut cursor, next, b),
            }
        }
        Cursor::SelectedCitations { index } => {
            let index = if state.preparation.active_value.is_none()
                && state.preparation.active_origin_value.is_none()
                && cursor.verified.is_none()
            {
                selected_index(&r, index, flag)?
            } else {
                index
            };
            if index >= 8 {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::ParentCitations { index: 0, donor: 0 },
                    b,
                );
            }
            let next = Cursor::SelectedCitations { index: index + 1 };
            if index >= 6 {
                let extra = if index == 6 {
                    &r.stop_sequence
                } else {
                    &r.stop_details
                };
                return match extra.as_ref().and_then(|fact| fact.selection.as_ref()) {
                    Some(selection) => cite(
                        state,
                        &mut cursor,
                        next,
                        selection
                            .inputs
                            .first()
                            .ok_or(Error::Unsupported("Claude selected extra donor missing"))?
                            .record,
                        b,
                    ),
                    None => progress(state, &mut cursor, next, b),
                };
            }
            match &r.selected_origins[usize::from(index)] {
                Some(origin) => cite(state, &mut cursor, next, origin.record, b),
                None => progress(state, &mut cursor, next, b),
            }
        }
        Cursor::ParentCitations { index, donor } => {
            let first = contribution(snapshot, head, 0, b, flag)?;
            let relations = parent::relations(&first);
            let Some(node) = relations.get(index) else {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::RecordExtraCitations { after: None },
                    b,
                );
            };
            if donor >= 6 {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::ParentCitations {
                        index: index + 1,
                        donor: 0,
                    },
                    b,
                );
            }
            let range = node.citation(donor);
            if let Some(range) = range {
                parent::verify(reader, snapshot, state, node, donor, range, &input, b, flag)?;
            }
            let next = Cursor::ParentCitations {
                index,
                donor: donor + 1,
            };
            match range {
                Some(range) => cite(state, &mut cursor, next, range, b),
                None => progress(state, &mut cursor, next, b),
            }
        }
        Cursor::RecordExtraCitations { after } => {
            use std::ops::Bound::{Excluded, Unbounded};
            let next = match &after {
                Some(key) => r
                    .record_extras
                    .range((Excluded(key.clone()), Unbounded))
                    .next(),
                None => r.record_extras.iter().next(),
            };
            let Some((key, extra)) = next else {
                return progress(state, &mut cursor, Cursor::AuthorCitation, b);
            };
            // A selected supplier may be missing from the effective
            // contribution's current supplier map, so cite its own record.
            let donor = extra
                .fact
                .selection
                .as_ref()
                .and_then(|selection| selection.inputs.first())
                .ok_or(Error::Unsupported(
                    "Claude selected record extra donor missing",
                ))?;
            cite(
                state,
                &mut cursor,
                Cursor::RecordExtraCitations {
                    after: Some(key.clone()),
                },
                donor.record,
                b,
            )
        }
        Cursor::AuthorCitation => {
            let author = r
                .author_selection
                .as_ref()
                .ok_or(Error::Unsupported("Claude author recipe unavailable"))?;
            cite(state, &mut cursor, Cursor::Layout, author.input.record, b)
        }
        Cursor::Layout => {
            stage(&mut cursor, Cursor::Author { index: 0 });
            Ok(action(
                state,
                b,
                Change::Layout {
                    layout: MessageLayout::Flat,
                    next: bounded_json(&cursor, b.metadata)?,
                },
            ))
        }
        Cursor::Author { index } => {
            // written_by is fixed identity metadata; prompt_kind is its own
            // leaf, set by the source classifier and not ranked.
            if index != 0 || r.message.prompt_kind.is_none() {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::MessageFields { item: 0, index: 0 },
                    b,
                );
            }
            let recipe = r
                .author_selection
                .as_ref()
                .ok_or(Error::Unsupported("Claude author recipe unavailable"))?
                .for_leaf(AuthorLeaf::PromptKind);
            let slot = Slot::Message(MessageField::PromptKind);
            field_turn(
                snapshot,
                reader,
                state,
                &mut cursor,
                FieldDescriptor {
                    slot: &slot,
                    time: None,
                    supplied: false,
                    selection: FieldSelector::Author(&recipe),
                },
                ValueSource::Leaf {
                    subject: Subject::Message(&r.message),
                    slot: &slot,
                },
                false,
                Stage::Claude(Cursor::MessageFields { item: 0, index: 0 }),
                &input,
                b,
                flag,
            )
        }
        Cursor::MessageFields { item, index } => {
            if item >= r.items {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::Selected { index: 0, fact: 0 },
                    b,
                );
            }
            let c = contribution(snapshot, head, item, b, flag)?;
            let Some(fact) = c.projection.facts.get(index) else {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::Block {
                        item,
                        local: 0,
                        phase: BlockPhase::Begin,
                    },
                    b,
                );
            };
            let next = Cursor::MessageFields {
                item,
                index: index
                    .checked_add(1)
                    .ok_or(Error::Invariant("Claude fact cursor overflow"))?,
            };
            if !is_message(&fact.field, item) {
                return progress(state, &mut cursor, next, b);
            }
            let slot = slot(&fact.field)?;
            field_turn(
                snapshot,
                reader,
                state,
                &mut cursor,
                field(fact, &slot, true)?,
                leaf(fact, &slot, &r, Some(&c), None)?,
                false,
                Stage::Claude(next),
                &input,
                b,
                flag,
            )
        }
        Cursor::Selected { index, fact } => {
            // An in-flight value/descriptor keeps its original field boundary.
            let original_index = index;
            let index = if state.preparation.active_value.is_none()
                && state.preparation.active_origin_value.is_none()
                && cursor.verified.is_none()
            {
                selected_index(&r, index, flag)?
            } else {
                index
            };
            let fact = if index == original_index { fact } else { 0 };
            if index >= 8 {
                return progress(state, &mut cursor, Cursor::RecordExtras { after: None }, b);
            }
            if index != original_index {
                // BeginValue carries no cursor, so commit the chosen field
                // before opening its lease.
                return progress(state, &mut cursor, Cursor::Selected { index, fact: 0 }, b);
            }
            if index >= 6 {
                let next = Cursor::Selected {
                    index: index + 1,
                    fact: 0,
                };
                let extra = if index == 6 {
                    &r.stop_sequence
                } else {
                    &r.stop_details
                };
                let Some(value) = extra else {
                    return progress(state, &mut cursor, next, b);
                };
                let slot = slot(&value.field)?;
                return field_turn(
                    snapshot,
                    reader,
                    state,
                    &mut cursor,
                    field(value, &slot, true)?,
                    leaf(value, &slot, &r, None, None)?,
                    false,
                    Stage::Claude(next),
                    &input,
                    b,
                    flag,
                );
            }
            let selected = r
                .selected_unit(index)
                .ok_or(Error::Invariant("Claude selected field cursor invalid"))?;
            let Some(value) = selected.facts.get(fact) else {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::Selected {
                        index: index + 1,
                        fact: 0,
                    },
                    b,
                );
            };
            let slot = slot(&value.field)?;
            field_turn(
                snapshot,
                reader,
                state,
                &mut cursor,
                field(value, &slot, true)?,
                leaf(value, &slot, &r, None, None)?,
                false,
                Stage::Claude(Cursor::Selected {
                    index,
                    fact: fact
                        .checked_add(1)
                        .ok_or(Error::Invariant("Claude selected fact overflow"))?,
                }),
                &input,
                b,
                flag,
            )
        }
        Cursor::RecordExtras { after } => {
            use std::ops::Bound::{Excluded, Unbounded};
            let next = match &after {
                Some(key) => r
                    .record_extras
                    .range((Excluded(key.clone()), Unbounded))
                    .next(),
                None => r.record_extras.iter().next(),
            };
            let Some((key, extra)) = next else {
                return progress(state, &mut cursor, Cursor::Finish, b);
            };
            let c = contribution(snapshot, head, extra.item, b, flag)?;
            let slot = slot(&extra.fact.field)?;
            field_turn(
                snapshot,
                reader,
                state,
                &mut cursor,
                field(&extra.fact, &slot, true)?,
                leaf(&extra.fact, &slot, &r, Some(&c), None)?,
                false,
                Stage::Claude(Cursor::RecordExtras {
                    after: Some(key.clone()),
                }),
                &input,
                b,
                flag,
            )
        }
        Cursor::Block {
            item,
            local: block_local,
            phase,
        } => {
            let c = contribution(snapshot, head, item, b, flag)?;
            let Some(original_content) = c.blocks.get(block_local as usize) else {
                return progress(
                    state,
                    &mut cursor,
                    Cursor::MessageFields {
                        item: item
                            .checked_add(1)
                            .ok_or(Error::Invariant("Claude contribution cursor overflow"))?,
                        index: 0,
                    },
                    b,
                );
            };
            let mut content = original_content.clone();
            let body = if let Some(required) = c
                .bodies
                .iter()
                .find(|body| body.content_item == block_local)
            {
                let payload = required
                    .payload
                    .as_ref()
                    .ok_or(Error::Unsupported("Claude required body unavailable"))?;
                let raw = polylogue_model::RawJson::parse(&c.raw)
                    .map_err(|e| Error::Source(e.to_string()))?;
                let path = source::required_body_path(&raw, block_local as usize)
                    .ok_or(Error::Unsupported("Claude required body path unavailable"))?;
                let body = retained
                    .companions
                    .iter()
                    .find(|body| {
                        body.path == path
                            && body.blob == payload.blob
                            && body.length == payload.length
                    })
                    .ok_or(Error::Unsupported(
                        "Claude body differs from pinned manifest",
                    ))?;
                snapshot.charge_selector_record(
                    usize::try_from(body.length).map_err(|_| StoreError::Range)?,
                )?;
                let text = String::from_utf8(reader.claude_body(body)?)
                    .map_err(|_| Error::Unsupported("Claude body is not UTF-8"))?;
                if let BlockContent::ToolResult(result) = &mut content {
                    result.output = Some(ToolOutput::Text(text));
                }
                Some(body)
            } else {
                None
            };
            let content = &content;
            let address = BlockSlot::Item(block_local);
            match phase {
                BlockPhase::Begin => {
                    let (text, kind) = match content {
                        BlockContent::Text { text } => (Some(text.as_str()), OwnedBlockKind::Text),
                        BlockContent::Thinking(thinking) => (
                            thinking.text.as_deref(),
                            OwnedBlockKind::Thinking {
                                encrypted: thinking.encrypted,
                            },
                        ),
                        BlockContent::ToolCall(call) => {
                            if r.message.copy_origin.is_some() || r.message.inherited_from.is_some()
                            {
                                return Err(Error::Unsupported(
                                    "copied Claude call lacks proved original operation scope",
                                ));
                            }
                            let raw_id = call
                                .call_id
                                .as_ref()
                                .filter(|s| !s.as_str().is_empty())
                                .ok_or(Error::Unsupported(
                                "Claude call has no supplied invocation ID",
                            ))?;
                            if !c.projection.facts.iter().any(|f|matches!(f.field,Field::Block{local,field:BlockField::CallId} if local==block_local)&&f.selection.is_some()){return Err(Error::Unsupported("Claude call ID donor unavailable"))}
                            let operation = DeclaredOperationId::from_native(
                                &state.handle.identity.session,
                                raw_id,
                            )
                            .map_err(|_| {
                                Error::Unsupported("Claude declared operation scope unavailable")
                            })?;
                            (
                                None,
                                OwnedBlockKind::ToolCall {
                                    operation: operation.as_str().into(),
                                },
                            )
                        }
                        BlockContent::ToolResult(result) => (
                            None,
                            OwnedBlockKind::ToolResult {
                                outcome: result.outcome.clone(),
                                exit_code: result.exit_code,
                                duration_ms: result.duration_ms,
                            },
                        ),
                        _ => {
                            return Err(Error::Unsupported(
                                "Claude block kind has no canonical projection",
                            ));
                        }
                    };
                    if let Some(text) = text {
                        if let ValueTurn::Action(change) = value_turn(
                            reader,
                            state,
                            ValueSource::Text(text),
                            values::Codec::Utf8,
                            b,
                            flag,
                        )? {
                            return Ok(action(state, b, change));
                        }
                    } else if state.preparation.active_value.is_some() {
                        return Err(Error::Invariant(
                            "textless Claude block has unexpected value producer",
                        ));
                    }
                    let (source_block, source_text) = c
                        .projection
                        .block_counts
                        .get(block_local as usize)
                        .copied()
                        .ok_or(Error::Unsupported("Claude block suppliedness unavailable"))?;
                    stage(
                        &mut cursor,
                        Cursor::Block {
                            item,
                            local: block_local,
                            phase: BlockPhase::Command,
                        },
                    );
                    Ok(action(
                        state,
                        b,
                        Change::BeginBlock {
                            group: item,
                            local: u64::from(block_local),
                            kind,
                            presence: BlockPresence {
                                source_block,
                                source_text,
                            },
                            next: bounded_json(&cursor, b.metadata)?,
                        },
                    ))
                }
                BlockPhase::Command => {
                    let next = Cursor::Block {
                        item,
                        local: block_local,
                        phase: BlockPhase::Fields(0),
                    };
                    let command = match content {
                        BlockContent::ToolCall(call) => call.command.as_deref(),
                        _ => None,
                    };
                    if let Some(command) = command {
                        if let ValueTurn::Action(change) = value_turn(
                            reader,
                            state,
                            ValueSource::Text(command),
                            values::Codec::Utf8,
                            b,
                            flag,
                        )? {
                            return Ok(action(state, b, change));
                        }
                        stage(&mut cursor, next);
                        return Ok(action(
                            state,
                            b,
                            Change::Command(bounded_json(&cursor, b.metadata)?),
                        ));
                    }
                    progress(state, &mut cursor, next, b)
                }
                BlockPhase::Fields(index) | BlockPhase::Retained(index) => {
                    let retained = matches!(phase, BlockPhase::Retained(_));
                    let facts = if retained {
                        &c.projection.derived
                    } else {
                        &c.projection.facts
                    };
                    let Some(fact) = facts.get(index) else {
                        return progress(
                            state,
                            &mut cursor,
                            Cursor::Block {
                                item,
                                local: block_local,
                                phase: if retained {
                                    BlockPhase::Signature
                                } else {
                                    BlockPhase::Retained(0)
                                },
                            },
                            b,
                        );
                    };
                    let next =
                        Cursor::Block {
                            item,
                            local: block_local,
                            phase: if retained {
                                BlockPhase::Retained(
                                    index
                                        .checked_add(1)
                                        .ok_or(Error::Invariant("Claude field cursor overflow"))?,
                                )
                            } else {
                                BlockPhase::Fields(index.checked_add(1).ok_or(Error::Invariant(
                                    "Claude block field cursor overflow",
                                ))?)
                            },
                        };
                    if local(&fact.field) != Some(block_local) {
                        return progress(state, &mut cursor, next, b);
                    }
                    let slot = slot(&fact.field)?;
                    let descriptor = if matches!(
                        fact.field,
                        Field::Block {
                            field: BlockField::Output,
                            ..
                        }
                    ) && body.is_some()
                    {
                        FieldDescriptor {
                            slot: &slot,
                            time: None,
                            supplied: !retained,
                            selection: FieldSelector::ClaudeBody(body.unwrap()),
                        }
                    } else {
                        field(fact, &slot, !retained)?
                    };
                    field_turn(
                        snapshot,
                        reader,
                        state,
                        &mut cursor,
                        descriptor,
                        leaf(fact, &slot, &r, Some(&c), Some(content))?,
                        true,
                        Stage::Claude(next),
                        &input,
                        b,
                        flag,
                    )
                }
                BlockPhase::Signature => {
                    let next = Cursor::Block {
                        item,
                        local: block_local,
                        phase: BlockPhase::Install,
                    };
                    if !matches!(content,BlockContent::Thinking(t) if t.signature.is_some()) {
                        return progress(state, &mut cursor, next, b);
                    }
                    // Take the auxiliary selector from its projection unit,
                    // never by building a raw path.
                    let Some(source::projection::Unit::Signature {
                        selection: Some(recipe),
                        ..
                    }) = c.signature_unit(item, block_local).map_err(Error::Source)?
                    else {
                        return Err(Error::Unsupported("Claude signature selector unavailable"));
                    };
                    let slot = Slot::Block {
                        block: address.clone(),
                        field: BlockField::ThinkingSignature,
                    };
                    field_turn(
                        snapshot,
                        reader,
                        state,
                        &mut cursor,
                        FieldDescriptor {
                            slot: &slot,
                            time: None,
                            supplied: false,
                            selection: FieldSelector::Claude(&recipe),
                        },
                        ValueSource::Leaf {
                            subject: Subject::Block {
                                slot: &address,
                                content,
                            },
                            slot: &slot,
                        },
                        true,
                        Stage::Claude(next),
                        &input,
                        b,
                        flag,
                    )
                }
                BlockPhase::Install => {
                    stage(
                        &mut cursor,
                        Cursor::Block {
                            item,
                            local: block_local
                                .checked_add(1)
                                .ok_or(Error::Invariant("Claude block cursor overflow"))?,
                            phase: BlockPhase::Begin,
                        },
                    );
                    Ok(action(
                        state,
                        b,
                        Change::Install(bounded_json(&cursor, b.metadata)?),
                    ))
                }
            }
        }
        Cursor::Finish => {
            let proof = proof(snapshot, head, state.handle.request.source_binding, b, flag)?;
            let finish = match r.message.in_progress {
                Some(true) => FinishState::InProgress,
                Some(false) => FinishState::Finished,
                None => r.output_header().map_err(Error::Source)?.finish,
            };
            Ok(action(
                state,
                b,
                Change::Finish {
                    proof: EvidenceDigest::Claude(proof),
                    finish,
                    next: bounded_json(&cursor, b.metadata)?,
                },
            ))
        }
    }
}

/// Hashes the selected evidence of one Message with point reads, on the
/// worker. A restart starts over.
fn proof(
    snapshot: &Snapshot,
    head: &Head,
    source_binding: BlobHash,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<Digest> {
    let view = View::new(snapshot, b.read)?;
    let response = response(&view, head)?;
    let mut baseline: [Option<Vec<u8>>; 8] = Default::default();
    // Locators for the selected keys only, not a copy of old values. Scratch
    // encoding is charged after updates.
    let mut extra_baseline = std::collections::BTreeMap::<String, u64>::new();
    let mut bodies = 0u64;
    let mut completions = 0u64;
    for item in 0..response.items {
        cancelled(flag)?;
        let c = contribution(snapshot, head, item, b, flag)?;
        for (previous, current) in baseline
            .iter_mut()
            .zip(c.final_fields().map_err(Error::Source)?)
        {
            if current.is_some() {
                *previous = current
            }
        }
        for fact in &c.projection.facts {
            if let Field::Extra { local: None, path } = &fact.field
                && let [key] = path.as_slice()
                && response.record_extras.contains_key(key)
            {
                extra_baseline.insert(key.clone(), item);
                bounded_json(&extra_baseline, b.metadata)?;
            }
        }
        if c.completion.is_some() {
            completions = completions
                .checked_add(1)
                .ok_or(Error::Invariant("Claude completion count overflow"))?;
        }
        let mut previous = None;
        for body in &c.bodies {
            if previous.is_some_and(|n| n >= body.content_item) {
                return Err(Error::Invariant(
                    "Claude required bodies are not in source item order",
                ));
            }
            previous = Some(body.content_item);
            bodies = bodies
                .checked_add(1)
                .ok_or(Error::Invariant("Claude evidence count overflow"))?;
        }
    }
    let metadata = response.metadata_units(&baseline);
    // Keys and tags only; raw values stay in the response cell.
    let mut extra_corrections = Vec::<(&str, String)>::new();
    for (key, selected) in &response.record_extras {
        cancelled(flag)?;
        let item = extra_baseline.get(key).ok_or(Error::Invariant(
            "Claude selected extra baseline unavailable",
        ))?;
        let c = contribution(snapshot, head, *item, b, flag)?;
        let raw = c.record_extra_raw(key).map_err(Error::Source)?;
        if selected.raw != raw {
            extra_corrections.push((key, source::keyed::SelectedRecordExtra::correction_tag(key)));
            bounded_json(&(&extra_baseline, &extra_corrections), b.metadata)?;
        }
    }
    let first = contribution(snapshot, head, 0, b, flag)?;
    let parent_relations = parent::relations(&first);
    bounded_json(&parent_relations, b.metadata)?;
    let count = response
        .items
        .checked_add(parent_relations.len() as u64)
        .and_then(|n| n.checked_add(metadata.len() as u64))
        .and_then(|n| n.checked_add(extra_corrections.len() as u64))
        .and_then(|n| n.checked_add(bodies))
        .and_then(|n| n.checked_add(completions))
        .ok_or(Error::Invariant("Claude evidence count overflow"))?;
    if count == 0 {
        return Err(Error::Unsupported(
            "Claude Message has no selected raw evidence",
        ));
    }
    let mut framer = if count == 1 && response.items == 1 {
        let c = contribution(snapshot, head, 0, b, flag)?;
        Framer::single_whole_record(c.raw.len() as u64, 0)
    } else {
        Framer::compound(count, Framer::compound_minimum_bytes()).map_err(Error::Evidence)?
    };
    for item in 0..response.items {
        cancelled(flag)?;
        let c = contribution(snapshot, head, item, b, flag)?;
        feed(
            &mut framer,
            c.record_unit(item),
            count == 1 && response.items == 1,
            b,
            flag,
        )?;
    }
    for item in 0..response.items {
        cancelled(flag)?;
        let c = contribution(snapshot, head, item, b, flag)?;
        if let Some(unit) = c.completion_unit(item) {
            feed(&mut framer, unit, false, b, flag)?;
        }
    }
    for (index, node) in parent_relations.iter().enumerate() {
        let bytes = node.evidence().map_err(Error::Source)?;
        bounded_json(node, b.metadata)?;
        feed_parts(
            &mut framer,
            &format!("parent/record/{index}"),
            Some(&bytes),
            None,
            false,
            b,
            flag,
        )?;
    }
    for unit in metadata {
        feed(&mut framer, unit, false, b, flag)?
    }
    for (key, tag) in extra_corrections {
        feed_parts(
            &mut framer,
            &tag,
            Some(&response.record_extras[key].raw),
            None,
            false,
            b,
            flag,
        )?;
    }
    for item in 0..response.items {
        cancelled(flag)?;
        let c = contribution(snapshot, head, item, b, flag)?;
        for index in 0..c.bodies.len() {
            let unit = c
                .body_unit(item, index)
                .map_err(Error::Source)?
                .ok_or(Error::Unsupported("Claude required body evidence missing"))?;
            feed(&mut framer, unit, false, b, flag)?;
        }
    }
    Ok(Digest {
        token: snapshot.head.token.clone(),
        head: head.clone(),
        source_binding,
        digest: framer.finish().map_err(Error::Evidence)?,
    })
}
fn feed(
    f: &mut Framer,
    unit: source::keyed::EvidenceUnit,
    single: bool,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<()> {
    use source::keyed::EvidenceUnit;
    let (tag, json, body) = match unit {
        EvidenceUnit::Json { tag, canonical, .. } => (tag, Some(canonical), None),
        EvidenceUnit::Body { tag, payload } => (tag, None, Some(payload)),
    };
    feed_parts(f, &tag, json.as_deref(), body.as_ref(), single, b, flag)
}
fn feed_parts(
    f: &mut Framer,
    tag: &str,
    json: Option<&[u8]>,
    body: Option<&source::keyed::BodyPayload>,
    single: bool,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<()> {
    let chunk = usize::try_from(b.hash_bytes)
        .unwrap_or(usize::MAX)
        .min(65536);
    if chunk == 0 {
        return Err(Error::Invariant("Claude evidence chunk allowance is zero"));
    }
    if !single {
        let fixed = 17u64
            .checked_add(tag.len() as u64)
            .ok_or(Error::Invariant("Claude evidence tag length overflow"))?;
        f.grant(fixed).map_err(Error::Evidence)?;
        if let Some(json) = &json {
            f.begin_json(tag.len() as u64, json.len() as u64)
                .map_err(Error::Evidence)?
        } else {
            f.begin_body(tag.len() as u64).map_err(Error::Evidence)?
        }
        // Provider tags use fixed ASCII labels, integers and hex-encoded keys.
        if !tag.is_ascii() {
            return Err(Error::Invariant("Claude evidence tag is not ASCII"));
        }
        for piece in tag.as_bytes().chunks(chunk) {
            cancelled(flag)?;
            f.tag(
                std::str::from_utf8(piece)
                    .map_err(|_| Error::Invariant("Claude tag encoding changed"))?,
            )
            .map_err(Error::Evidence)?
        }
    }
    if let Some(json) = json {
        for piece in json.chunks(chunk) {
            cancelled(flag)?;
            f.grant(piece.len() as u64).map_err(Error::Evidence)?;
            f.json(piece).map_err(Error::Evidence)?
        }
    }
    if let Some(body) = body {
        cancelled(flag)?;
        f.grant(40).map_err(Error::Evidence)?;
        f.body(body.length, body.blob).map_err(Error::Evidence)?
    }
    Ok(())
}

#[cfg(test)]
#[path = "claude/coalescing_tests.rs"]
mod coalescing_tests;

#[cfg(test)]
#[path = "claude/bookkeeping_tests.rs"]
mod bookkeeping_tests;
