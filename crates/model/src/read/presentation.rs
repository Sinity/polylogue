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
impl TextAlignment {
    fn same_body(&self, other: &Self) -> bool {
        same_body(&self.source, &other.source)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct VariantText {
    pub mark: VariantMark,
    pub relation: VariantRelation,
    pub text: String,
    /// Original body ranges in source order. The replacement is emitted at the
    /// first range; subsequent covered ranges are removed. Empty means the key
    /// names exactly one complete text body. Multi-body targets need alignment.
    #[serde(default)]
    pub alignment: Vec<TextAlignment>,
}

/// Checked construction and deserialization reject ambiguous overlays. Parts of
/// one rendition for the same original must be grouped before insertion.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, schemars::JsonSchema)]
#[serde(transparent)]
pub struct PresentationMap(BTreeMap<Reference, VariantText>);
impl PresentationMap {
    pub fn insert(&mut self, original: Reference, variant: VariantText) -> Result<(), ReadError> {
        if self.0.contains_key(&original) {
            return Err(invalid("multiple renditions for one original"));
        }
        if variant.mark.original != original.to_string() {
            return Err(invalid("variant mark and original reference disagree"));
        }
        if variant.alignment.is_empty()
            && !matches!(
                original,
                Reference::Block { .. }
                    | Reference::Assertion(_)
                    | Reference::Variant { part: Some(_), .. }
            )
        {
            return Err(invalid(
                "multi-body variant target needs original body alignment",
            ));
        }
        for (n, range) in variant.alignment.iter().enumerate() {
            Reference::parse(&range.source.reference).map_err(|e| invalid(&e.to_string()))?;
            if range.start > range.end {
                return Err(invalid("reversed variant body range"));
            }
            if variant.alignment[..n]
                .iter()
                .any(|other| overlaps(range, other))
            {
                return Err(invalid("overlapping variant body ranges"));
            }
        }
        for (key, prior) in &self.0 {
            if conflicts(key, prior, &original, &variant) {
                return Err(invalid("overlapping distinct variant replacements"));
            }
        }
        self.0.insert(original, variant);
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
    pub fn mark_for<'a>(
        &'a self,
        source: &Source,
        options: &ReadOptions,
    ) -> Option<&'a VariantMark> {
        self.0
            .iter()
            .find(|(key, value)| active(value, options) && targets(key, value, source))
            .map(|(_, value)| &value.mark)
    }
    /// Applies ranges simultaneously against original bytes, never the output
    /// of another replacement. Scalar boundaries are checked before slicing.
    pub fn apply<'a>(
        &self,
        source: &Source,
        original: &'a str,
        options: &ReadOptions,
    ) -> Result<Cow<'a, str>, ReadError> {
        let mut patches = Vec::new();
        for (key, variant) in &self.0 {
            if !active(variant, options) {
                continue;
            }
            let replacement = if variant.relation == VariantRelation::Omits {
                ""
            } else {
                &variant.text
            };
            if variant.alignment.is_empty() {
                if key.to_string() == source.reference {
                    patches.push((0, original.len(), replacement));
                }
            } else {
                for (n, range) in variant
                    .alignment
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| same_body(&r.source, source))
                {
                    let start = usize::try_from(range.start)
                        .map_err(|_| invalid("variant range exceeds address space"))?;
                    let end = usize::try_from(range.end)
                        .map_err(|_| invalid("variant range exceeds address space"))?;
                    if original.get(start..end).is_none() {
                        return Err(invalid(
                            "variant range is outside original UTF-8 body boundaries",
                        ));
                    }
                    patches.push((start, end, if n == 0 { replacement } else { "" }));
                }
            }
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
fn same_body(a: &Source, b: &Source) -> bool {
    a.reference == b.reference && a.block == b.block
}
fn targets(key: &Reference, value: &VariantText, source: &Source) -> bool {
    if value.alignment.is_empty() {
        key.to_string() == source.reference
    } else {
        value.alignment.iter().any(|r| same_body(&r.source, source))
    }
}
fn overlaps(a: &TextAlignment, b: &TextAlignment) -> bool {
    a.same_body(b) && (a.start < b.end && b.start < a.end || a.start == b.start)
}
fn conflicts(ak: &Reference, a: &VariantText, bk: &Reference, b: &VariantText) -> bool {
    if a.alignment.is_empty() {
        return targets(bk, b, &Source::record(ak.to_string()))
            || b.alignment
                .iter()
                .any(|r| r.source.reference == ak.to_string());
    }
    if b.alignment.is_empty() {
        return a
            .alignment
            .iter()
            .any(|r| r.source.reference == bk.to_string());
    }
    a.alignment
        .iter()
        .any(|a| b.alignment.iter().any(|b| overlaps(a, b)))
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
