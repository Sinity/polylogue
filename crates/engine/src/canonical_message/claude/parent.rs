//! Resolve only the first contribution's parent on the same pinned provider view.
//! The view charges cumulative point reads; encoded scratch is admitted separately.
use super::*;
use polylogue_sources::{
    claude_code::{parent::Node, projection::CountingObligation, selection},
    selection::SelectedRaw,
};
use std::collections::BTreeSet;
pub(super) struct Qualified {
    pub contribution: Contribution,
    pub records: Vec<Node>,
}
impl std::ops::Deref for Qualified {
    type Target = Contribution;
    fn deref(&self) -> &Contribution {
        &self.contribution
    }
}
pub(super) fn relations(c: &Qualified) -> &[Node] {
    &c.records
}
fn unresolved(reason: &'static str) -> Error {
    Error::Unsupported(reason)
}
fn native(value: &RawJson) -> Result<Option<polylogue_model::NativeId>> {
    if value.is_null() {
        return Ok(None);
    }
    let n = value
        .native()
        .ok_or_else(|| unresolved("Claude retained parent link has wrong type"))?;
    Ok((!n.as_str().is_empty()).then(|| n.clone()))
}
fn key(n: &polylogue_model::NativeId) -> Result<String> {
    serde_json::to_string(n).map_err(|e| Error::Source(e.to_string()))
}
/// Stop at the nearest Message. Cycles among skipped records and a grouped
/// self-reference refuse here; this is not global tree/active-path validation.
fn walk(
    view: &View<'_>,
    head: &Head,
    mut next: Option<polylogue_model::NativeId>,
    visited: &mut BTreeSet<String>,
    allowance: usize,
    mut records: Option<&mut Vec<Node>>,
    flag: &AtomicBool,
) -> Result<Option<polylogue_sources::claude_code::keyed::RecordAddress>> {
    while let Some(uuid) = next {
        cancelled(flag)?;
        let k = key(&uuid)?;
        if !visited.insert(k.clone()) {
            return Err(unresolved("Claude retained parent cycle"));
        }
        bounded_json(visited, allowance)?;
        let address = callback(view, view.record(&head.pending.session_native, &k))?;
        let node = callback(view, view.parent_node(&head.pending.session_native, &k))?;
        if address.is_some() && node.is_some() {
            return Err(unresolved("Claude retained parent identity conflict"));
        }
        if let Some(address) = address {
            if address.message.session() != head.pending.message.session()
                || address.message == head.pending.message
            {
                return Err(unresolved(
                    "Claude retained parent crosses Session or grouped self",
                ));
            }
            return Ok(Some(address));
        }
        let node = node.ok_or_else(|| unresolved("Claude retained parent record unavailable"))?;
        if node.uuid != uuid {
            return Err(Error::Invariant(
                "Claude retained parent lookup identity changed",
            ));
        }
        next = match node.predecessor().map_err(unresolved)? {
            Some(link) => native(&link.value)?,
            None => None,
        };
        if let Some(records) = records.as_deref_mut() {
            records.push(node);
            bounded_json(&(&*visited, &*records), allowance)?;
        }
    }
    Ok(None)
}
pub(super) fn qualify(
    view: &View<'_>,
    head: &Head,
    c: &mut Contribution,
    allowance: usize,
    flag: &AtomicBool,
) -> Result<Vec<Node>> {
    let Some(index) = c
        .projection
        .facts
        .iter()
        .position(|f| matches!(f.field, Field::FirstMessage(MessageField::Parent)))
    else {
        return Ok(vec![]);
    };
    let raw = RawJson::parse(&c.raw).map_err(|e| Error::Source(e.to_string()))?;
    let supplied = raw
        .get("parentUuid")
        .ok_or(Error::Invariant("Claude parent fact has no value"))?;
    let Some(uuid) = native(supplied)? else {
        return Err(unresolved(
            "Claude supplied parent fact has no native parent",
        ));
    };
    let donor: SelectedRaw = c
        .origins
        .select(&[polylogue_sources::selection::PathPart::Key(
            "parentUuid".into(),
        )])
        .ok_or_else(|| unresolved("Claude parent donor unavailable"))?;
    let mut visited = BTreeSet::new();
    visited.insert(c.uuid.clone());
    let mut records = Vec::new();
    let address = walk(
        view,
        head,
        Some(uuid.clone()),
        &mut visited,
        allowance,
        Some(&mut records),
        flag,
    )?;
    // All other counting obligations retain their existing typed refusal.
    c.projection
        .obligations
        .retain(|o| *o != CountingObligation::ParentResolution);
    if let Some(address) = address {
        c.projection.facts[index].selection = Some(if records.is_empty() {
            // Existing descriptor semantics and direct-history version evidence.
            selection::parent_reference(uuid, address.message, donor)
        } else {
            selection::parent_via(uuid, address.message, donor, records.clone())
        });
        bounded_json(&(&records, &c.projection.facts[index].selection), allowance)?;
    } else {
        // A proved non-Message root supplies no fabricated Message reference.
        c.projection.facts.remove(index);
    }
    Ok(records)
}

/// Verify retained relation evidence before admitting its citation. No provider
/// path is opened and a failed/cancelled turn returns no installable operation.
pub(super) fn verify(
    reader: &ReadHandle,
    snapshot: &Snapshot,
    state: &State,
    node: &Node,
    donor: u8,
    range: polylogue_sources::selection::RecordRange,
    input: &str,
    b: Budgets,
    flag: &AtomicBool,
) -> Result<()> {
    cancelled(flag)?;
    let length = range
        .end
        .checked_sub(range.start)
        .ok_or(Error::Invariant("Claude parent donor range reversed"))?;
    if range.end > snapshot.head.parsed_upto || length == 0 {
        return Err(Error::Unsupported(
            "Claude parent donor is outside complete retained records",
        ));
    }
    if length > b.read.encoded_bytes as u64 {
        return Err(polylogue_store::Error::ParserAdmission {
            resource: "Claude parent source record bytes",
            required: length,
        }
        .into());
    }
    let raw = reader.read_source_range_checked(
        input,
        snapshot.head.token.chain,
        state.handle.request.source_binding,
        range.start,
        length,
        || {
            if flag.load(Ordering::Acquire) {
                Err(polylogue_store::Error::Cancelled)
            } else {
                Ok(())
            }
        },
    )?;
    cancelled(flag)?;
    if !node.verify(donor, &raw).map_err(Error::Source)? {
        return Err(Error::Unsupported(
            "Claude retained parent proof differs from original source",
        ));
    }
    Ok(())
}
