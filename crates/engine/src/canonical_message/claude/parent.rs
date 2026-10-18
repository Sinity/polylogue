//! Resolves the first contribution's parent on the same pinned view. Point
//! reads are charged to the view; scratch encoding has its own limit.
use super::*;
use polylogue_sources::{
    claude_code::{parent::Node, projection::CountingObligation, selection},
    selection::SelectedRaw,
};
use std::collections::BTreeSet;
pub(super) struct ResolvedContribution {
    pub contribution: Contribution,
    pub records: Vec<Node>,
}
impl std::ops::Deref for ResolvedContribution {
    type Target = Contribution;
    fn deref(&self) -> &Contribution {
        &self.contribution
    }
}
pub(super) fn relations(c: &ResolvedContribution) -> &[Node] {
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
        .ok_or_else(|| unresolved("Claude parent link has wrong type"))?;
    Ok((!n.as_str().is_empty()).then(|| n.clone()))
}
fn key(n: &polylogue_model::NativeId) -> Result<String> {
    serde_json::to_string(n).map_err(|e| Error::Source(e.to_string()))
}
/// Stops at the nearest Message. Cycles among skipped records and grouped
/// self-references fail here; the whole tree is not checked.
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
            return Err(unresolved("Claude parent cycle"));
        }
        bounded_json(visited, allowance)?;
        let address = callback(view, view.record(&head.pending.session_native, &k))?;
        let node = callback(view, view.parent_node(&head.pending.session_native, &k))?;
        if address.is_some() && node.is_some() {
            return Err(unresolved("Claude parent identity conflict"));
        }
        if let Some(address) = address {
            if address.message.session() != head.pending.message.session()
                || address.message == head.pending.message
            {
                return Err(unresolved(
                    "Claude parent crosses Session or grouped self",
                ));
            }
            return Ok(Some(address));
        }
        let node = node.ok_or_else(|| unresolved("Claude parent record unavailable"))?;
        if node.uuid != uuid {
            return Err(Error::Invariant(
                "Claude parent lookup identity changed",
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
pub(super) fn resolve(
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
    // Other counting cases stay unsupported.
    c.projection
        .obligations
        .retain(|o| *o != CountingObligation::ParentResolution);
    if let Some(address) = address {
        c.projection.facts[index].selection = Some(if records.is_empty() {
            selection::parent_reference(uuid, address.message, donor)
        } else {
            selection::parent_via(uuid, address.message, donor, records.clone())
        });
        bounded_json(&(&records, &c.projection.facts[index].selection), allowance)?;
    } else {
        // A non-Message root gives no Message reference.
        c.projection.facts.remove(index);
    }
    Ok(records)
}

/// Checks the stored relation records before citing them. Opens no provider
/// file; a failed or cancelled turn returns nothing to install.
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
            "Claude parent supplier is outside the complete records",
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
            "Claude parent proof differs from original source",
        ));
    }
    Ok(())
}
