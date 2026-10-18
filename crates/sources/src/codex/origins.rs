//! Exact original JSONL record contributors. Coordinates are never digest
//! bytes, normalized payload extents, input identities or historical versions.
use super::selection::{PathPart, SelectedRaw};
pub use crate::selection::RecordRange;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Origins {
    #[default]
    Unknown,
    Record(RecordRange),
    /// Exact original record and its original selected subtree, never an overlay root.
    Located(SelectedRaw),
    /// Current envelope plus direct original records for selected immediate
    /// item fields. No chain of earlier overlays or per-Message history Vec.
    Overlay {
        envelope: RecordRange,
        fields: BTreeMap<String, SelectedRaw>,
    },
}
impl Origins {
    pub fn field(&self, name: &str) -> Self {
        match self {
            Self::Record(r) => Self::Record(*r),
            Self::Located(raw) => {
                let mut raw = raw.clone();
                raw.path.push(PathPart::Key(name.into()));
                Self::Located(raw)
            }
            Self::Overlay { fields, .. } => fields
                .get(name)
                .cloned()
                .map(Self::Located)
                .unwrap_or(Self::Unknown),
            Self::Unknown => Self::Unknown,
        }
    }
    pub(super) fn overlay(&self, current: &Self, kept: &[String], provided: &[String]) -> Self {
        let Self::Located(current) = current else {
            return Self::Unknown;
        };
        let envelope = current.record;
        if kept.is_empty() {
            return Self::Located(current.clone());
        }
        let mut fields = BTreeMap::new();
        for key in kept {
            let Self::Located(origin) = self.field(key) else {
                return Self::Unknown;
            };
            fields.insert(key.clone(), origin);
        }
        for key in provided {
            let Self::Located(origin) = Self::Located(current.clone()).field(key) else {
                return Self::Unknown;
            };
            fields.insert(key.clone(), origin);
        }
        Self::Overlay { envelope, fields }
    }
    pub(super) fn at_path(&self, path: &[&str]) -> Self {
        let mut selected = match self {
            Self::Record(record) => SelectedRaw {
                record: *record,
                path: Vec::new(),
            },
            Self::Located(raw) => raw.clone(),
            _ => return Self::Unknown,
        };
        selected
            .path
            .extend(path.iter().map(|key| PathPart::Key((*key).into())));
        Self::Located(selected)
    }
    pub(super) fn selected(&self) -> Option<SelectedRaw> {
        match self {
            Self::Located(raw) => Some(raw.clone()),
            _ => None,
        }
    }
    pub fn cursor(&self) -> Cursor {
        Cursor::default()
    }
    /// One original record per call. Duplicates are intentional: the owning
    /// version-pinned SET deduplicates them without a growing in-memory set.
    pub fn advance(&self, cursor: &mut Cursor) -> Result<Option<RecordRange>, String> {
        match self {
            Self::Unknown => Err("Codex original-record origins are unavailable".into()),
            Self::Record(range) | Self::Located(SelectedRaw { record: range, .. }) => {
                if cursor.envelope {
                    return Ok(None);
                }
                RecordRange::new(range.start, range.end)?;
                cursor.envelope = true;
                Ok(Some(*range))
            }
            Self::Overlay { envelope, fields } => {
                if !cursor.envelope {
                    RecordRange::new(envelope.start, envelope.end)?;
                    cursor.envelope = true;
                    return Ok(Some(*envelope));
                }
                use std::ops::Bound::{Excluded, Unbounded};
                let next = match &cursor.after {
                    None => fields.iter().next(),
                    Some(after) => fields
                        .range::<str, _>((Excluded(after.as_str()), Unbounded))
                        .next(),
                };
                match next {
                    Some((name, selected)) => {
                        let range = &selected.record;
                        RecordRange::new(range.start, range.end)?;
                        cursor.after = Some(name.clone());
                        Ok(Some(*range))
                    }
                    None => Ok(None),
                }
            }
        }
    }
}
/// Scalar/keyset position within one admitted current descriptor. Pin its
/// owning source cell/revision before persisting; mutation invalidates it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Cursor {
    envelope: bool,
    after: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlay_preserves_original_root_paths_instead_of_rebasing_retained_fields() {
        let old = Origins::Record(RecordRange::new(0, 80).unwrap()).at_path(&["payload"]);
        let new = Origins::Record(RecordRange::new(90, 180).unwrap()).at_path(&["payload", "item"]);
        let overlaid = old.overlay(&new, &["content".into()], &["phase".into()]);
        let content = overlaid.field("content").selected().unwrap();
        let phase = overlaid.field("phase").selected().unwrap();
        assert_eq!(content.record, RecordRange { start: 0, end: 80 });
        assert_eq!(
            content.path,
            vec![
                PathPart::Key("payload".into()),
                PathPart::Key("content".into())
            ]
        );
        assert_eq!(
            phase.path,
            vec![
                PathPart::Key("payload".into()),
                PathPart::Key("item".into()),
                PathPart::Key("phase".into())
            ]
        );
    }
}
