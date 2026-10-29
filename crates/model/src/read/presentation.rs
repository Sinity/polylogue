//! Original references and UTF-8 body ranges identify presentation overlays.
//! Canonical records and their source hashes are never rewritten.
use super::{ReadError, ReadOptions, Source, VariantMark};
use crate::{Reference, VariantRelation};
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
pub struct VariantText {
    pub mark: VariantMark,
    /// Declared relations, in original variant part order.
    pub relations: Vec<VariantRelation>,
    pub text: String,
    /// Original body ranges in source order. Replacement is emitted at the first
    /// range; subsequent covered ranges are removed. Empty means the key names
    /// exactly one complete text body. Multi-body targets need alignment.
    #[serde(default)]
    pub alignment: Vec<TextAlignment>,
}

/// Checked construction rejects ambiguity. The private index is derived from
/// the one public reference map, so a record with N bodies needs N body lookups.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct PresentationMap(
    BTreeMap<Reference, VariantText>,
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
    pub fn insert(&mut self, original: Reference, variant: VariantText) -> Result<(), ReadError> {
        if self.0.contains_key(&original) {
            return Err(invalid("multiple renditions for one original"));
        }
        if variant.mark.original != original.to_string() {
            return Err(invalid("variant mark and original reference disagree"));
        }
        if variant.relations.is_empty() {
            return Err(invalid(
                "variant presentation needs declared part relations",
            ));
        }
        let mut pending: BTreeMap<BodyKey, Vec<Patch>> = BTreeMap::new();
        if variant.alignment.is_empty() {
            let exact = match &original {
                Reference::Block { .. } => true,
                Reference::Assertion(revision)
                | Reference::Variant {
                    revision,
                    part: Some(_),
                } => revision.revision.is_some(),
                _ => false,
            };
            if !exact {
                return Err(invalid(
                    "multi-body or unresolved variant target needs original body alignment",
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
            for (n, range) in variant.alignment.iter().enumerate() {
                Reference::parse(&range.source.reference).map_err(|e| invalid(&e.to_string()))?;
                if range.start > range.end {
                    return Err(invalid("reversed variant body range"));
                }
                let patch = Patch {
                    original: original.clone(),
                    start: range.start,
                    end: Some(range.end),
                    first: n == 0,
                };
                let ranges = pending.entry(body_key(&range.source)).or_default();
                if ranges.iter().any(|prior| overlap(prior, &patch)) {
                    return Err(invalid("overlapping variant body ranges"));
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
                return Err(invalid("overlapping distinct variant replacements"));
            }
        }
        self.0.insert(original, variant);
        for (body, patches) in pending {
            self.1.entry(body).or_default().extend(patches);
        }
        Ok(())
    }
    pub fn get(&self, original: &Reference) -> Option<&VariantText> {
        self.0.get(original)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&Reference, &VariantText)> {
        self.0.iter()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn for_source<'a>(
        &'a self,
        source: &Source,
    ) -> impl Iterator<Item = (&'a Reference, &'a VariantText)> {
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
    ) -> Option<&'a VariantMark> {
        self.for_source(source)
            .map(|(_, variant)| variant)
            .find(|variant| active(variant, options))
            .map(|variant| &variant.mark)
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
            let variant = &self.0[&patch.original];
            if !active(variant, options) {
                continue;
            }
            let replacement = if variant
                .relations
                .iter()
                .all(|relation| *relation == VariantRelation::Omits)
                || !patch.first
            {
                ""
            } else {
                &variant.text
            };
            let start = usize::try_from(patch.start)
                .map_err(|_| invalid("variant range exceeds address space"))?;
            let end = patch
                .end
                .map(usize::try_from)
                .transpose()
                .map_err(|_| invalid("variant range exceeds address space"))?
                .unwrap_or(original.len());
            if original.get(start..end).is_none() {
                return Err(invalid(
                    "variant range is outside original UTF-8 body boundaries",
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
                return Err(invalid("overlapping distinct variant replacements"));
            }
            text.push_str(&original[at..start]);
            text.push_str(replacement);
            at = end;
        }
        text.push_str(&original[at..]);
        Ok(Cow::Owned(text))
    }
}
fn active(variant: &VariantText, options: &ReadOptions) -> bool {
    !variant.mark.stale || options.variants.contains(&variant.mark.key)
}
fn body_key(source: &Source) -> BodyKey {
    let block = match Reference::parse(&source.reference) {
        Ok(Reference::Block { n, .. }) => Some(n),
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
fn invalid(why: &str) -> ReadError {
    ReadError::InvalidEvidence(why.into())
}
impl TryFrom<BTreeMap<Reference, VariantText>> for PresentationMap {
    type Error = ReadError;
    fn try_from(map: BTreeMap<Reference, VariantText>) -> Result<Self, Self::Error> {
        let mut checked = Self::default();
        for (key, value) in map {
            checked.insert(key, value)?;
        }
        Ok(checked)
    }
}
impl From<PresentationMap> for BTreeMap<Reference, VariantText> {
    fn from(value: PresentationMap) -> Self {
        value.0
    }
}
impl<'de> Deserialize<'de> for PresentationMap {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Checked;
        impl<'de> serde::de::Visitor<'de> for Checked {
            type Value = PresentationMap;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a map of distinct original references to non-overlapping renditions")
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
