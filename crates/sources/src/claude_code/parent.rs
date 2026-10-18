//! Parent links of Claude records, including non-Message records, so a
//! Message's parent can be found by skipping those records.
use crate::selection::{PathPart, RecordRange, SelectedRaw};
use polylogue_model::{NativeId, RawJson as Value};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub value: Value,
    pub origin: SelectedRaw,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    pub uuid: NativeId,
    pub kind: String,
    pub subtype: Option<String>,
    pub subtype_origin: Option<SelectedRaw>,
    pub parent: Option<Link>,
    pub logical_parent: Option<Link>,
    pub summary: Option<Link>,
    pub progress_kind: Option<Link>,
    pub envelope: RecordRange,
    /// Hash of the overlaid record, to detect an identical repeat.
    pub fingerprint: [u8; 32],
    pub field_fingerprints: BTreeMap<NativeId, [u8; 32]>,
    pub conflicted: bool,
}
impl Node {
    pub(super) fn overlay(
        old: Option<Self>,
        incoming: &Value,
        range: RecordRange,
    ) -> Result<Self, String> {
        let uuid = incoming
            .get("uuid")
            .and_then(Value::native)
            .ok_or("parent node has no native UUID")?
            .clone();
        let kind = incoming
            .get("type")
            .and_then(Value::as_str)
            .ok_or("parent node has no type")?
            .to_owned();
        let subtype = if kind == "system" {
            incoming
                .get("subtype")
                .and_then(Value::as_str)
                .map(str::to_owned)
        } else {
            None
        };
        let mut field_fingerprints = old
            .as_ref()
            .map(|o| o.field_fingerprints.clone())
            .unwrap_or_default();
        for (key, value) in incoming.as_object().ok_or("parent node is not an object")? {
            let raw = serde_json::to_vec(value).map_err(|e| e.to_string())?;
            field_fingerprints.insert(key.clone(), *blake3::hash(&raw).as_bytes());
        }
        let fingerprint =
            *blake3::hash(&serde_json::to_vec(&field_fingerprints).map_err(|e| e.to_string())?)
                .as_bytes();
        if let Some(previous) = &old
            && previous.fingerprint == fingerprint
        {
            return Ok(previous.clone());
        }
        let conflicted = old.as_ref().is_some_and(|o| {
            o.conflicted
                || o.uuid != uuid
                || o.kind != kind
                || (kind == "system" && incoming.get("subtype").is_some() && o.subtype != subtype)
        });
        let link = |key: &str, previous: Option<Link>| {
            incoming
                .get(key)
                .map(|v| Link {
                    value: v.opaque(),
                    origin: SelectedRaw {
                        record: range,
                        path: vec![PathPart::Key(key.into())],
                    },
                })
                .or(previous)
        };
        let subtype_origin = incoming
            .get("subtype")
            .filter(|_| kind == "system")
            .map(|_| SelectedRaw {
                record: range,
                path: vec![PathPart::Key("subtype".into())],
            })
            .or_else(|| old.as_ref().and_then(|o| o.subtype_origin.clone()));
        let summary = if kind == "user" {
            link(
                "isCompactSummary",
                old.as_ref().and_then(|o| o.summary.clone()),
            )
        } else {
            None
        };
        let progress_kind = if kind == "progress" {
            incoming
                .get("data")
                .map(|data| Link {
                    value: data.get("type").cloned().unwrap_or(Value::Null),
                    origin: SelectedRaw {
                        record: range,
                        path: vec![PathPart::Key("data".into()), PathPart::Key("type".into())],
                    },
                })
                .or_else(|| old.as_ref().and_then(|o| o.progress_kind.clone()))
        } else {
            None
        };
        Ok(Self {
            uuid,
            kind,
            subtype: subtype.or_else(|| old.as_ref().and_then(|o| o.subtype.clone())),
            subtype_origin,
            parent: link("parentUuid", old.as_ref().and_then(|o| o.parent.clone())),
            logical_parent: link(
                "logicalParentUuid",
                old.as_ref().and_then(|o| o.logical_parent.clone()),
            ),
            summary,
            progress_kind,
            envelope: range,
            fingerprint,
            field_fingerprints,
            conflicted,
        })
    }
    /// The record this one follows. A compaction boundary uses its
    /// `logicalParentUuid`.
    pub fn predecessor(&self) -> Result<Option<&Link>, &'static str> {
        if self.conflicted {
            return Err("Claude record UUID has conflicting parent records");
        }
        if self.kind == "user"
            && self.summary.as_ref().and_then(|l| l.value.as_bool()) != Some(true)
        {
            return Err("Claude user record is not a compact summary");
        }
        if self.kind == "system" && self.subtype.as_deref() == Some("compact_boundary") {
            return self
                .logical_parent
                .as_ref()
                .map(Some)
                .ok_or("Claude compaction logical parent unavailable");
        }
        Ok(self.parent.as_ref())
    }
    /// The record range of origin `origin` (0 is the envelope).
    pub fn citation(&self, origin: u8) -> Option<RecordRange> {
        match origin {
            0 => Some(self.envelope),
            1 => self.parent.as_ref().map(|l| l.origin.record),
            2 => self.logical_parent.as_ref().map(|l| l.origin.record),
            3 => self.subtype_origin.as_ref().map(|d| d.record),
            4 => self.summary.as_ref().map(|l| l.origin.record),
            5 => self.progress_kind.as_ref().map(|l| l.origin.record),
            _ => None,
        }
    }
    pub fn verify(&self, origin: u8, bytes: &[u8]) -> Result<bool, String> {
        let raw = Value::parse(bytes).map_err(|e| e.to_string())?;
        if raw.get("uuid").and_then(Value::native) != Some(&self.uuid)
            || raw.get("type").and_then(Value::as_str) != Some(self.kind.as_str())
        {
            return Ok(false);
        }
        let selected = match origin {
            1 => self.parent.as_ref().map(|l| &l.origin),
            2 => self.logical_parent.as_ref().map(|l| &l.origin),
            3 => self.subtype_origin.as_ref(),
            4 => self.summary.as_ref().map(|l| &l.origin),
            5 => self.progress_kind.as_ref().map(|l| &l.origin),
            _ => None,
        };
        let path = match origin {
            1 => vec![PathPart::Key("parentUuid".into())],
            2 => vec![PathPart::Key("logicalParentUuid".into())],
            3 => vec![PathPart::Key("subtype".into())],
            4 => vec![PathPart::Key("isCompactSummary".into())],
            5 => vec![PathPart::Key("data".into()), PathPart::Key("type".into())],
            _ => vec![],
        };
        if origin != 0 && selected.is_none_or(|d| d.path != path) {
            return Ok(false);
        }
        Ok(match origin {
            0 => true,
            1 => self
                .parent
                .as_ref()
                .is_some_and(|l| raw.get("parentUuid") == Some(&l.value)),
            2 => self
                .logical_parent
                .as_ref()
                .is_some_and(|l| raw.get("logicalParentUuid") == Some(&l.value)),
            3 => raw.get("subtype").and_then(Value::as_str) == self.subtype.as_deref(),
            4 => self
                .summary
                .as_ref()
                .is_some_and(|l| raw.get("isCompactSummary") == Some(&l.value)),
            5 => self
                .progress_kind
                .as_ref()
                .is_some_and(|l| raw.get("data").and_then(|v| v.get("type")) == Some(&l.value)),
            _ => false,
        })
    }
    /// Version evidence for this relation; positions are recorded as citations.
    pub fn evidence(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(&(
            &self.uuid,
            &self.kind,
            &self.subtype,
            self.parent.as_ref().map(|l| &l.value),
            self.logical_parent.as_ref().map(|l| &l.value),
            self.summary.as_ref().map(|l| &l.value),
            self.progress_kind.as_ref().map(|l| &l.value),
        ))
        .map_err(|e| e.to_string())
    }
}

/// A `Node` without its duplicate-detection hashes, as written into descriptors.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationProof {
    pub uuid: NativeId,
    pub kind: String,
    pub subtype: Option<String>,
    pub subtype_origin: Option<SelectedRaw>,
    pub parent: Option<Link>,
    pub logical_parent: Option<Link>,
    pub summary: Option<Link>,
    pub progress_kind: Option<Link>,
    pub envelope: RecordRange,
}
impl From<&Node> for RelationProof {
    fn from(node: &Node) -> Self {
        Self {
            uuid: node.uuid.clone(),
            kind: node.kind.clone(),
            subtype: node.subtype.clone(),
            subtype_origin: node.subtype_origin.clone(),
            parent: node.parent.clone(),
            logical_parent: node.logical_parent.clone(),
            summary: node.summary.clone(),
            progress_kind: node.progress_kind.clone(),
            envelope: node.envelope,
        }
    }
}
