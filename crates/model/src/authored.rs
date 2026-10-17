//! Immutable authored intent and revision graph semantics. No target lookup occurs here.
use crate::{BlobHash, BlockSlot, ErrorKind, Failure};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const REQUEST_BYTES: usize = 65536;
pub const MAX_TARGETS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoteKind {
    Note,
    Correction,
    Withdrawal,
}
impl NoteKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::Correction => "correction",
            Self::Withdrawal => "withdrawal",
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    #[serde(rename = "ref")]
    pub reference: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seen_hash: Option<BlobHash>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Relation {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(rename = "ref")]
    pub other: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WriteNote {
    pub request_id: String,
    pub author: String,
    pub machine: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<NoteKind>,
    pub value: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<Target>>,
    #[serde(default)]
    pub relations: Vec<Relation>,
}
pub fn invalid(message: impl Into<String>) -> Failure {
    Failure::new(ErrorKind::InvalidQuery, message)
}
pub fn uuid_v7(id: &str) -> bool {
    let b = id.as_bytes();
    b.len() == 36
        && [8, 13, 18, 23].into_iter().all(|i| b[i] == b'-')
        && b[14] == b'7'
        && matches!(b[19], b'8' | b'9' | b'a' | b'b')
        && b.iter()
            .enumerate()
            .all(|(i, v)| [8, 13, 18, 23].contains(&i) || matches!(v,b'0'..=b'9'|b'a'..=b'f'))
}
pub fn target_reference(value: &str) -> bool {
    value.len() <= 16384 && !value.contains('\0') && target_address(value).is_ok()
}
pub fn target_address(
    value: &str,
) -> Result<(std::borrow::Cow<'_, str>, Option<BlockSlot>), Failure> {
    if let Some(id) = value.strip_prefix("message:").filter(|s| !s.is_empty()) {
        return Ok((std::borrow::Cow::Borrowed(id), None));
    }
    if let Some(reference) = value.strip_prefix("block:") {
        let (message, slot) = crate::BlockId::address(reference)
            .ok_or_else(|| invalid("invalid or unsupported exact block reference"))?;
        return Ok((message, Some(slot)));
    }
    Err(invalid("expected exact message: or block: target"))
}

pub fn note_reference(value: &str) -> Result<(&str, Option<&str>), Failure> {
    let value = value
        .strip_prefix("note:")
        .ok_or_else(|| invalid("expected note:<root>[@<revision-id>]"))?;
    let (root, pin) = value
        .split_once('@')
        .map_or((value, None), |(a, b)| (a, Some(b)));
    if !uuid_v7(root) || pin.is_some_and(|p| !uuid_v7(p)) {
        return Err(invalid(
            "note references require full UUIDv7 root/revision IDs; ordinal pins are unsupported",
        ));
    }
    Ok((root, pin))
}
impl WriteNote {
    pub fn validate(&self) -> Result<(), Failure> {
        if !uuid_v7(&self.request_id) {
            return Err(invalid("request_id must be a canonical UUIDv7"));
        }
        if self.author.is_empty()
            || self.author.len() > 1024
            || self.machine.is_empty()
            || self.machine.len() > 1024
            || self.value.len() > 16384
        {
            return Err(invalid("authored identity/value admission exceeded"));
        }
        let required = self.request_id.len()
            + self.author.len()
            + self.machine.len()
            + self.value.len()
            + 4096
            + self
                .targets
                .iter()
                .flatten()
                .map(|t| t.reference.len() + 128)
                .sum::<usize>()
            + self
                .relations
                .iter()
                .map(|r| r.kind.len() + r.other.len() + 64)
                .sum::<usize>();
        if required > REQUEST_BYTES {
            return Err(invalid("authored request metadata exceeds65536 bytes"));
        }
        if self
            .targets
            .as_ref()
            .is_some_and(|t| t.is_empty() || t.len() > MAX_TARGETS)
            || self.relations.len() > 2
        {
            return Err(invalid("authored target/relation admission exceeded"));
        }
        let mut targets = BTreeSet::new();
        for t in self.targets.iter().flatten() {
            if !target_reference(&t.reference) || !targets.insert(&t.reference) {
                return Err(invalid(
                    "targets must be distinct exact Message/block references",
                ));
            }
        }
        let mut kinds = BTreeSet::new();
        for r in &self.relations {
            if !kinds.insert(r.kind.as_str()) {
                return Err(invalid("one relation of each supported type is admitted"));
            }
            match r.kind.as_str() {
                "revises" | "withdraws" => {
                    note_reference(&r.other)?;
                }
                "corrects" if target_reference(&r.other) || note_reference(&r.other).is_ok() => {}
                _ => {
                    return Err(Failure::new(
                        ErrorKind::NotYetAvailable,
                        "only revises, withdraws and corrects relations are supported",
                    ));
                }
            }
        }
        if kinds.contains("withdraws")
            && (kinds.len() != 1
                || self.kind != Some(NoteKind::Withdrawal)
                || self.targets.is_some())
        {
            return Err(invalid(
                "withdrawal requires one withdraws relation and no target override",
            ));
        }
        if kinds.contains("revises") && self.kind == Some(NoteKind::Withdrawal) {
            return Err(invalid(
                "withdrawal revisions are unsupported; withdraw the withdrawal to restore",
            ));
        }
        if !kinds.contains("revises")
            && !kinds.contains("withdraws")
            && (self.kind.is_none()
                || self.kind == Some(NoteKind::Withdrawal)
                || self.targets.is_none())
        {
            return Err(invalid(
                "new note/correction requires kind and exact targets",
            ));
        }
        Ok(())
    }
    /// Hash submitted intent, before inheritance/current-head lookup or clock resolution.
    pub fn intent(&self) -> Result<serde_json::Value, Failure> {
        self.validate()?;
        let mut v = serde_json::to_value(self).map_err(|e| invalid(e.to_string()))?;
        v.as_object_mut().unwrap().remove("request_id");
        v["command"] = serde_json::json!("note");
        // Targets/relations are sets in durable storage, so order is not intent.
        if let Some(a) = v
            .get_mut("targets")
            .and_then(serde_json::Value::as_array_mut)
        {
            a.sort_by_key(|v| v["ref"].as_str().unwrap().to_owned());
        }
        v["relations"].as_array_mut().unwrap().sort_by_key(|v| {
            (
                v["type"].as_str().unwrap().to_owned(),
                v["ref"].as_str().unwrap().to_owned(),
            )
        });
        Ok(v)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Revision {
    pub id: String,
    pub root: String,
    pub kind: NoteKind,
    pub value: String,
    pub author: String,
    pub machine: String,
    pub created_at: i64,
    pub targets: Vec<Target>,
    pub relations: Vec<Relation>,
}
impl Revision {
    pub fn reference(&self) -> String {
        format!("note:{}@{}", self.root, self.id)
    }
}

#[derive(Clone, Debug)]
pub struct GraphNode {
    pub root: String,
    pub relations: Vec<Relation>,
}
pub fn validate_controls(graph: &BTreeMap<String, GraphNode>) -> Result<(), Failure> {
    fn visit(
        id: &str,
        g: &BTreeMap<String, GraphNode>,
        active: &mut BTreeSet<String>,
        done: &mut BTreeSet<String>,
    ) -> Result<(), Failure> {
        if done.contains(id) {
            return Ok(());
        }
        let node = g.get(id).ok_or_else(|| {
            Failure::new(
                ErrorKind::NotYetAvailable,
                "authored control parent is unavailable",
            )
        })?;
        if !active.insert(id.into()) {
            return Err(Failure::new(
                ErrorKind::NotYetAvailable,
                "cyclic authored revision/withdrawal controls",
            ));
        }
        for r in &node.relations {
            if matches!(r.kind.as_str(), "revises" | "withdraws") {
                visit(&r.other, g, active, done)?;
            }
        }
        active.remove(id);
        done.insert(id.into());
        Ok(())
    }
    let mut done = BTreeSet::new();
    for id in graph.keys() {
        visit(id, graph, &mut BTreeSet::new(), &mut done)?;
    }
    Ok(())
}
/// All dependencies must be present. New local writes only point backwards;
/// malformed/cyclic imported graphs refuse rather than choose by a timestamp.
fn withdrawal_states(
    graph: &BTreeMap<String, GraphNode>,
) -> Result<BTreeMap<String, bool>, Failure> {
    fn visit(
        id: &str,
        g: &BTreeMap<String, GraphNode>,
        visiting: &mut BTreeSet<String>,
        memo: &mut BTreeMap<String, bool>,
    ) -> Result<bool, Failure> {
        if let Some(v) = memo.get(id) {
            return Ok(*v);
        }
        if !g.contains_key(id) || !visiting.insert(id.to_owned()) {
            return Err(Failure::new(
                ErrorKind::NotYetAvailable,
                "incomplete or cyclic authored withdrawal graph",
            ));
        }
        let mut result = false;
        for (other, node) in g {
            if node
                .relations
                .iter()
                .any(|r| r.kind == "withdraws" && r.other == id)
                && !visit(other, g, visiting, memo)?
            {
                result = true;
            }
        }
        visiting.remove(id);
        memo.insert(id.to_owned(), result);
        Ok(result)
    }
    let mut memo = BTreeMap::new();
    for id in graph.keys() {
        visit(id, graph, &mut BTreeSet::new(), &mut memo)?;
    }
    Ok(memo)
}
pub fn withdrawn(id: &str, graph: &BTreeMap<String, GraphNode>) -> Result<bool, Failure> {
    withdrawal_states(graph)?.get(id).copied().ok_or_else(|| {
        Failure::new(
            ErrorKind::NotYetAvailable,
            "authored control parent is unavailable",
        )
    })
}
pub fn current(root: &str, graph: &BTreeMap<String, GraphNode>) -> Result<Vec<String>, Failure> {
    validate_controls(graph)?;
    let withdrawn = withdrawal_states(graph)?;
    if *withdrawn.get(root).ok_or_else(|| {
        Failure::new(
            ErrorKind::NotYetAvailable,
            "authored family root is unavailable",
        )
    })? {
        return Ok(vec![]);
    }
    let mut heads = vec![];
    for (id, _) in graph.iter().filter(|(_, n)| n.root == root) {
        if withdrawn[id] {
            continue;
        }
        let mut revised = false;
        for (other, n) in graph.iter().filter(|(_, n)| n.root == root) {
            if n.relations
                .iter()
                .any(|r| r.kind == "revises" && r.other == *id)
                && !withdrawn[other]
            {
                revised = true;
                break;
            }
        }
        if !revised {
            heads.push(id.clone());
        }
    }
    Ok(heads)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn graph(edges: &[(&str, &str, &str)]) -> BTreeMap<String, GraphNode> {
        let mut g = BTreeMap::new();
        for (a, b, kind) in edges {
            for id in [a, b] {
                g.entry((*id).into()).or_insert_with(|| GraphNode {
                    root: (*id).into(),
                    relations: vec![],
                });
            }
            g.get_mut(*a).unwrap().relations.push(Relation {
                kind: (*kind).into(),
                other: (*b).into(),
            });
        }
        g
    }
    #[test]
    fn authored_control_dag_rejects_self_even_odd_and_mixed_cycles() {
        for edges in [
            vec![("a", "a", "withdraws")],
            vec![("a", "b", "withdraws"), ("b", "a", "withdraws")],
            vec![
                ("a", "b", "withdraws"),
                ("b", "c", "withdraws"),
                ("c", "a", "withdraws"),
            ],
            vec![("a", "b", "revises"), ("b", "a", "withdraws")],
        ] {
            let g = graph(&edges);
            assert!(validate_controls(&g).is_err());
            assert!(current("a", &g).is_err());
        }
        let g = graph(&[("w1", "a", "withdraws"), ("w2", "w1", "withdraws")]);
        validate_controls(&g).unwrap();
        assert!(!withdrawn("a", &g).unwrap());
        assert!(withdrawn("w1", &g).unwrap());
        let mut missing = g;
        missing.remove("a");
        assert!(validate_controls(&missing).is_err());
    }
}
