//! Original references and UTF-8 body ranges identify presentation overlays.
//! Stored records and their source hashes are never rewritten.
use super::{AlternateMark, ReadError, ReadOptions, Source};
use crate::{AlternateRelation, Reference};
use serde::{Deserialize, Serialize};
use std::{borrow::Cow, collections::BTreeMap};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextAlignment {
    pub source: Source,
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AlternateText {
    pub mark: AlternateMark,
    /// Declared relations, in original alternate part order.
    pub relations: Vec<AlternateRelation>,
    pub text: String,
    /// Original body ranges in source order. Replacement is emitted at the first
    /// range; subsequent covered ranges are removed. Empty means the key names
    /// exactly one complete text body. Multi-body targets need alignment.
    /// A synthesized source-coordinate separator has no bytes in either block;
    /// its replacement uses a zero-width anchor at the preceding body's end,
    /// retaining the exact original span key. View formatting stays structural.
    #[serde(default)]
    pub alignment: Vec<TextAlignment>,
}

/// Checked construction rejects ambiguity. The private index is derived from
/// the one public reference map, so a record with N bodies needs N body lookups.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct PresentationMap(
    BTreeMap<Reference, AlternateText>,
    #[serde(skip)]
    #[schemars(skip)]
    BTreeMap<BodyKey, Vec<Patch>>,
);
type BodyKey = (String, Option<u64>);
#[derive(Debug, Clone, PartialEq, Eq)]
struct Patch {
    original: Reference,
    start: u64,
    end: Option<u64>,
    first: bool,
}
impl PresentationMap {
    pub fn insert(
        &mut self,
        original: Reference,
        alternate: AlternateText,
    ) -> Result<(), ReadError> {
        if self.0.contains_key(&original) {
            return Err(ReadError::invalid("multiple alternates for one original"));
        }
        if alternate.mark.original != original.to_string() {
            return Err(ReadError::invalid(
                "alternate mark and original reference disagree",
            ));
        }
        if alternate.relations.is_empty() {
            return Err(ReadError::invalid(
                "alternate presentation needs declared part relations",
            ));
        }
        let mut pending: BTreeMap<BodyKey, Vec<Patch>> = BTreeMap::new();
        if alternate.alignment.is_empty() {
            let exact = match &original {
                Reference::Block { .. } => true,
                Reference::Annotation(revision)
                | Reference::Alternate {
                    revision,
                    part: Some(_),
                } => revision.revision.is_some(),
                _ => false,
            };
            if !exact {
                return Err(ReadError::invalid(
                    "multi-body or unresolved alternate target needs original body alignment",
                ));
            }
            pending
                .entry(body_key(&Source::record(original.to_string())))
                .or_default()
                .push(Patch {
                    original: original.clone(),
                    start: 0,
                    end: None,
                    first: true,
                });
        } else {
            for (n, range) in alternate.alignment.iter().enumerate() {
                let reference = Reference::parse(&range.source.reference)
                    .map_err(|e| ReadError::invalid(e.to_string()))?;
                if range.start > range.end {
                    return Err(ReadError::invalid("reversed alternate body range"));
                }
                let patch = Patch {
                    original: original.clone(),
                    start: range.start,
                    end: Some(range.end),
                    first: n == 0,
                };
                let ranges = pending
                    .entry(body_key_of(&range.source, &reference))
                    .or_default();
                if ranges.iter().any(|prior| overlap(prior, &patch)) {
                    return Err(ReadError::invalid("overlapping alternate body ranges"));
                }
                ranges.push(patch);
            }
        }
        for (body, patches) in &pending {
            if let Some(prior) = self.1.get(body)
                && patches
                    .iter()
                    .any(|patch| prior.iter().any(|prior| overlap(patch, prior)))
            {
                return Err(ReadError::invalid(
                    "overlapping distinct alternate replacements",
                ));
            }
        }
        self.0.insert(original, alternate);
        for (body, patches) in pending {
            self.1.entry(body).or_default().extend(patches);
        }
        Ok(())
    }
    pub fn for_source<'a>(
        &'a self,
        source: &Source,
    ) -> impl Iterator<Item = (&'a Reference, &'a AlternateText)> {
        self.1
            .get(&body_key(source))
            .into_iter()
            .flatten()
            .map(|patch| (&patch.original, &self.0[&patch.original]))
    }
    pub fn mark_for<'a>(
        &'a self,
        source: &Source,
        options: &ReadOptions,
    ) -> Option<&'a AlternateMark> {
        self.for_source(source)
            .map(|(_, alternate)| alternate)
            .find(|alternate| alternate.is_active(options))
            .map(|alternate| &alternate.mark)
    }
    /// Apply simultaneously against original bytes. Never use offsets in the
    /// output of another replacement; reject non-scalar UTF-8 boundaries.
    pub fn apply<'a>(
        &self,
        source: &Source,
        original: &'a str,
        options: &ReadOptions,
    ) -> Result<Cow<'a, str>, ReadError> {
        let mut patches = Vec::new();
        for patch in self.1.get(&body_key(source)).into_iter().flatten() {
            let alternate = &self.0[&patch.original];
            if !alternate.is_active(options) {
                continue;
            }
            let replacement = if alternate
                .relations
                .iter()
                .all(|relation| *relation == AlternateRelation::Omits)
                || !patch.first
            {
                ""
            } else {
                &alternate.text
            };
            let start = usize::try_from(patch.start)
                .map_err(|_| ReadError::invalid("alternate range exceeds address space"))?;
            let end = patch
                .end
                .map(usize::try_from)
                .transpose()
                .map_err(|_| ReadError::invalid("alternate range exceeds address space"))?
                .unwrap_or(original.len());
            if original.get(start..end).is_none() {
                return Err(ReadError::invalid(
                    "alternate range is outside original UTF-8 body boundaries",
                ));
            }
            patches.push((start, end, replacement));
        }
        if patches.is_empty() {
            return Ok(Cow::Borrowed(original));
        }
        patches.sort_by_key(|p| p.0);
        let mut text = String::new();
        let mut at = 0;
        for (start, end, replacement) in patches {
            if start < at {
                return Err(ReadError::invalid(
                    "overlapping distinct alternate replacements",
                ));
            }
            text.push_str(&original[at..start]);
            text.push_str(replacement);
            at = end;
        }
        text.push_str(&original[at..]);
        Ok(Cow::Owned(text))
    }
}
impl AlternateText {
    /// A stale alternate applies only when the reader names its key.
    pub fn is_active(&self, options: &ReadOptions) -> bool {
        !self.mark.stale || options.alternate.as_deref() == Some(self.mark.key.as_str())
    }
}
fn body_key(source: &Source) -> BodyKey {
    match Reference::parse(&source.reference) {
        Ok(reference) => body_key_of(source, &reference),
        Err(_) => (source.reference.clone(), source.block),
    }
}
fn body_key_of(source: &Source, reference: &Reference) -> BodyKey {
    let block = match reference {
        Reference::Block { n, .. } => Some(*n),
        _ => source.block,
    };
    (source.reference.clone(), block)
}
fn overlap(a: &Patch, b: &Patch) -> bool {
    a.end.is_none()
        || b.end.is_none()
        || (a.start < b.end.unwrap() && b.start < a.end.unwrap())
        || a.start == b.start
}
impl<'de> Deserialize<'de> for PresentationMap {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Checked;
        impl<'de> serde::de::Visitor<'de> for Checked {
            type Value = PresentationMap;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a map of distinct original references to non-overlapping alternates")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let mut result = PresentationMap::default();
                while let Some((key, value)) = map.next_entry()? {
                    result
                        .insert(key, value)
                        .map_err(serde::de::Error::custom)?;
                }
                Ok(result)
            }
        }
        deserializer.deserialize_map(Checked)
    }
}
#[cfg(test)]
mod tests;
