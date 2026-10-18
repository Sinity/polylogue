//! Follow one declared native leaf through point state before folding responses.
//! The caller already pins current source/selection proofs for every Message.
use super::*;
use polylogue_model::{NativeId, RawJson};
use polylogue_sources::claude_code::{
    keyed::{ActiveLeaf, Contribution, PendingMessage, RecordAddress},
    parent::Node,
};

pub(super) struct ResultPeer {
    pub message: String,
    pub parent: Option<NativeId>,
    pub caller: Option<RawJson>,
    pub record_start: u64,
    pub valid: bool,
    pub calls: BTreeSet<NativeId>,
}

struct Reads<'a> {
    catalog: &'a SessionCatalog,
    dependency: &'a SessionInputDependency,
    allowance: usize,
    used: usize,
    until: Instant,
    cancelled: &'a mut dyn FnMut() -> bool,
}
impl Reads<'_> {
    fn check(&mut self) -> Result<(), Error> {
        deadline(self.until)?;
        if (self.cancelled)() {
            return Err(Error::Cancelled);
        }
        Ok(())
    }
    fn charge(&mut self, bytes: usize) -> Result<(), Error> {
        self.used = self.used.saturating_add(bytes);
        if self.used > self.allowance {
            return Err(Error::ParserAdmission {
                resource: "Claude active leaf proof",
                required: self.used as u64,
            });
        }
        Ok(())
    }
    fn cell<T: serde::de::DeserializeOwned>(
        &mut self,
        namespace: &str,
        key: &impl serde::Serialize,
    ) -> Result<Option<T>, Error> {
        self.check()?;
        let key = serde_json::to_vec(key).map_err(|e| Error::Invariant(e.to_string()))?;
        self.charge(namespace.len().saturating_add(key.len()))?;
        let bytes = self.catalog.dependency_cell(
            self.dependency,
            namespace,
            &key,
            self.allowance - self.used,
        )?;
        bytes
            .map(|bytes| {
                self.charge(bytes.len())?;
                serde_json::from_slice(&bytes)
                    .map_err(|e| Error::Invariant(format!("invalid Claude active leaf state: {e}")))
            })
            .transpose()
    }
}
fn native(value: &RawJson) -> Option<Option<NativeId>> {
    if value.is_null() {
        return Some(None);
    }
    let id = value.native()?;
    (!id.as_str().is_empty()).then(|| Some(id.clone()))
}

/// None means an explicit marker exists but its exact current path is unknown.
/// Absence of a marker is handled separately by the existing linear proof.
pub(super) fn prove(
    catalog: &SessionCatalog,
    dependency: &SessionInputDependency,
    target: &str,
    owner: &str,
    leaf: &ActiveLeaf,
    pending: &BTreeMap<String, PendingMessage>,
    peers: &[ResultPeer],
    until: Instant,
    allowance: usize,
    cancelled: &mut dyn FnMut() -> bool,
) -> Result<Option<BTreeSet<String>>, Error> {
    let head = dependency.head().ok_or(Error::Range)?;
    if !leaf.scoped
        || !matches!(leaf.kind.as_str(), "last-prompt" | "summary")
        || leaf.record.start >= leaf.record.end
        || leaf.record.end > head.parsed_upto
    {
        return Ok(None);
    }
    let Some(Some(mut next)) = native(&leaf.value) else {
        return Ok(None);
    };
    let mut reads = Reads {
        catalog,
        dependency,
        allowance,
        used: 0,
        until,
        cancelled,
    };
    let mut visited = BTreeSet::new();
    let mut active = BTreeSet::new();
    loop {
        reads.check()?;
        let key = serde_json::to_string(&next).map_err(|e| Error::Invariant(e.to_string()))?;
        if !visited.insert(key.clone()) {
            return Ok(None);
        }
        reads.charge(key.len())?;
        let address: Option<RecordAddress> = reads.cell(claude::RECORD, &(owner, &key))?;
        let node: Option<Node> = reads.cell(claude::PARENT_NODE, &(owner, &key))?;
        let predecessor = match (address, node) {
            (Some(address), None) => {
                let Some(p) = pending.get(address.message.as_str()) else {
                    return Ok(None);
                };
                if address.message.session().as_str() != target
                    || p.session_native != owner
                    || p.message != address.message
                    || p.response != address.response
                    || address.item >= p.items
                    || p.incomplete
                {
                    return Ok(None);
                }
                let Some(relation) = &address.relation else {
                    return Ok(None);
                };
                if relation.uuid != next
                    || relation.record.start >= relation.record.end
                    || relation.record.end > head.parsed_upto
                {
                    return Ok(None);
                }
                if active.insert(address.message.as_str().to_owned()) {
                    reads.charge(address.message.as_str().len())?;
                }
                match &relation.parent {
                    Some(link)
                        if link.donor.record.start < link.donor.record.end
                            && link.donor.record.end <= head.parsed_upto =>
                    {
                        native(&link.value)
                    }
                    Some(_) => None,
                    None => Some(None),
                }
            }
            (None, Some(node)) => {
                if node.uuid != next
                    || node.envelope.start >= node.envelope.end
                    || node.envelope.end > head.parsed_upto
                {
                    return Ok(None);
                }
                match node.predecessor() {
                    Ok(Some(link)) => native(&link.value),
                    Ok(None) => Some(None),
                    Err(_) => return Ok(None),
                }
            }
            _ => return Ok(None),
        };
        match predecessor {
            Some(Some(parent)) => next = parent,
            Some(None) => break,
            None => return Ok(None),
        }
    }
    // A call is answered once even if its identical invocation was re-emitted.
    // This is source correspondence only, never result/status inference.
    let mut answered = BTreeSet::new();
    for peer in peers {
        if !peer.valid || peer.calls.is_empty() {
            return Ok(None);
        }
        for id in &peer.calls {
            if !answered.insert(id.clone()) {
                return Ok(None);
            }
            reads.charge(
                serde_json::to_vec(id)
                    .map_err(|e| Error::Invariant(e.to_string()))?
                    .len(),
            )?;
        }
    }
    // Only extra peer membership needs call correspondence. Membership already
    // witnessed by the native record path makes no operation/status assertion.
    for peer in peers {
        if active.contains(&peer.message) {
            continue;
        }
        let direct = if let Some(value) = &peer.caller {
            let Some(Some(uuid)) = native(value) else {
                return Ok(None);
            };
            let key = serde_json::to_string(&uuid).map_err(|e| Error::Invariant(e.to_string()))?;
            let address: Option<RecordAddress> = reads.cell(claude::RECORD, &(owner, &key))?;
            let node: Option<Node> = reads.cell(claude::PARENT_NODE, &(owner, &key))?;
            if node.is_some()
                || address.as_ref().is_none_or(|address| {
                    address.relation.as_ref().is_none_or(|relation| {
                        relation.uuid != uuid
                            || relation.record.start >= relation.record.end
                            || relation.record.end > head.parsed_upto
                    })
                })
            {
                return Ok(None);
            }
            address
        } else {
            None
        };
        let address = if let Some(address) = direct {
            Some(address)
        } else {
            nearest(&mut reads, owner, peer.parent.clone())?
        };
        let Some(address) = address else {
            // No caller proof is not proof of an inactive caller. The result
            // may still be a peer of a call on the active response path.
            return Ok(None);
        };
        let Some(p) = pending.get(address.message.as_str()) else {
            return Ok(None);
        };
        if p.message.session().as_str() != target
            || p.session_native != owner
            || p.response != address.response
            || p.response.role != polylogue_model::Role::Assistant
            || address.item >= p.items
            || p.incomplete
        {
            return Ok(None);
        }
        let Some(_) = call_evidence(
            &mut reads,
            target,
            p,
            peer,
            peer.caller.as_ref().map(|_| address.item),
        )?
        else {
            return Ok(None);
        };
        if peer.caller.is_some() {
            let Some(parent) = nearest(&mut reads, owner, peer.parent.clone())? else {
                return Ok(None);
            };
            // Different native records of one response agree. A different
            // response is contradictory tree authority, even for a re-emitted ID.
            if parent.message != address.message {
                return Ok(None);
            }
        }
        if !active.contains(address.message.as_str()) {
            continue;
        }
        active.insert(peer.message.clone());
        reads.charge(peer.message.len())?;
    }
    reads.check()?;
    Ok((!active.is_empty()).then_some(active))
}

fn nearest(
    reads: &mut Reads<'_>,
    owner: &str,
    mut next: Option<NativeId>,
) -> Result<Option<RecordAddress>, Error> {
    let mut visited = BTreeSet::new();
    while let Some(uuid) = next {
        reads.check()?;
        let key = serde_json::to_string(&uuid).map_err(|e| Error::Invariant(e.to_string()))?;
        if !visited.insert(key.clone()) {
            return Ok(None);
        }
        reads.charge(key.len())?;
        let address: Option<RecordAddress> = reads.cell(claude::RECORD, &(owner, &key))?;
        let node: Option<Node> = reads.cell(claude::PARENT_NODE, &(owner, &key))?;
        match (address, node) {
            (Some(address), None) => {
                let Some(relation) = &address.relation else {
                    return Ok(None);
                };
                if relation.uuid != uuid
                    || relation.record.start >= relation.record.end
                    || relation.record.end
                        > reads.dependency.head().ok_or(Error::Range)?.parsed_upto
                {
                    return Ok(None);
                }
                return Ok(Some(address));
            }
            (None, Some(node))
                if node.uuid == uuid
                    && node.envelope.start < node.envelope.end
                    && node.envelope.end
                        <= reads.dependency.head().ok_or(Error::Range)?.parsed_upto =>
            {
                next = match node.predecessor() {
                    Ok(Some(link)) => match native(&link.value) {
                        Some(next) => next,
                        None => return Ok(None),
                    },
                    Ok(None) => None,
                    Err(_) => return Ok(None),
                };
            }
            _ => return Ok(None),
        }
    }
    Ok(None)
}
fn call_evidence(
    reads: &mut Reads<'_>,
    target: &str,
    p: &PendingMessage,
    peer: &ResultPeer,
    direct: Option<u64>,
) -> Result<Option<()>, Error> {
    let mut calls = BTreeMap::<NativeId, RawJson>::new();
    for item in 0..p.items {
        if direct.is_some_and(|direct| item != direct) {
            continue;
        }
        let Some(c): Option<Contribution> =
            reads.cell(claude::CONTRIBUTION, &(target, p.message.as_str(), item))?
        else {
            return Ok(None);
        };
        let Some(donor) = c
            .origins
            .fields
            .get(&vec!["message".into(), "content".into()])
        else {
            return Ok(None);
        };
        // Corrections/re-emissions after the result cannot supply its call.
        if donor.record.start >= peer.record_start {
            continue;
        }
        let raw = RawJson::parse(&c.raw).map_err(|e| Error::Invariant(e.to_string()))?;
        let Some(content) = raw
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(RawJson::as_array)
        else {
            continue;
        };
        for value in content {
            if value.get("type").and_then(RawJson::as_str) != Some("tool_use") {
                continue;
            }
            let Some(id) = value
                .get("id")
                .and_then(RawJson::native)
                .filter(|id| !id.as_str().is_empty())
            else {
                continue;
            };
            if !peer.calls.contains(id) {
                continue;
            }
            match calls.get(id) {
                Some(previous) if previous != value => return Ok(None),
                Some(_) => {} // exact lawful re-emission is one call
                None => {
                    reads.charge(
                        serde_json::to_vec(value)
                            .map_err(|e| Error::Invariant(e.to_string()))?
                            .len(),
                    )?;
                    calls.insert(id.clone(), value.clone());
                }
            }
        }
    }
    Ok(peer
        .calls
        .iter()
        .all(|id| calls.contains_key(id))
        .then_some(()))
}

#[cfg(test)]
#[path = "claude_active/tests.rs"]
mod tests;
