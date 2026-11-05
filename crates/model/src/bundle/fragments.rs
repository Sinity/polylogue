//! Selected text is a partial source observation, never a shortened canonical
//! Message or Block. Complete-source attestation belongs to the Store reader.
pub mod seek;

use super::SourceClaim;
use crate::{BlobHash, Reference, SpanBounds, SpanUnit, authored::ImportAttribution};
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum TextSourceVersion {
    /// The existing complete Message or Document value identity. A fragment
    /// carrying this hash does not verify the complete value.
    ContentHash {
        record: Reference,
        hash: BlobHash,
    },
    ExactRevision {
        reference: Reference,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextCoverage {
    pub start: u64,
    pub end: u64,
    /// A retained length claim; knowing it does not make outside bytes present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct TextSpanOrigin {
    pub reference: Reference,
    /// Declared UTF-8 coordinates of the original request. For a non-byte
    /// request, the unavailable prefix cannot be verified from a snippet.
    pub start: u64,
    pub end: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextFragmentProvenance {
    /// Requests that contributed to this coverage, not permissions or proof
    /// that any one sender observed the whole union.
    pub spans: Vec<TextSpanOrigin>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<SourceClaim>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub imports: Vec<ImportAttribution>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextFragment {
    /// One existing reference-text recipe, including its declared separators.
    pub container: Reference,
    pub source_version: TextSourceVersion,
    pub coverage: TextCoverage,
    pub text: String,
    pub text_hash: BlobHash,
    pub provenance: TextFragmentProvenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TextFragmentRecord {
    #[serde(rename = "ref")]
    pub reference: String,
    /// Sender selection/order evidence. Import computes receiving admission
    /// from its actual source reading; it never trusts this as a local receipt.
    pub commit: u64,
    pub fragment: TextFragment,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TextFragmentError {
    #[error("text fragment has invalid coverage or a contradictory known total")]
    Coverage,
    #[error("text fragment byte length or selected-text hash does not match")]
    SelectedText,
    #[error("text fragment reference does not name its normalized byte interval")]
    Reference,
    #[error("text fragment version does not identify the exact container owner")]
    SourceVersion,
    #[error("text fragment has invalid original-span provenance")]
    Origin,
    #[error("text fragment source claim is invalid: {0}")]
    SourceClaim(String),
    #[error("same-version text observations conflict")]
    Conflict,
    #[error("text fragment bounds split a UTF-8 character")]
    CharacterBoundary,
    #[error("complete source text disagrees with the fragment or its span claims")]
    CompleteSource,
}

type Result<T> = std::result::Result<T, TextFragmentError>;

impl TextSourceVersion {
    /// Checks exact version/container metadata without attesting source bytes.
    pub fn validate_container(&self, container: &Reference) -> Result<()> {
        if Reference::parse(&container.to_string()).ok().as_ref() != Some(container) {
            return Err(TextFragmentError::SourceVersion);
        }
        let version_reference = match self {
            Self::ContentHash { record, .. } => record,
            Self::ExactRevision { reference } => reference,
        };
        if Reference::parse(&version_reference.to_string())
            .ok()
            .as_ref()
            != Some(version_reference)
        {
            return Err(TextFragmentError::SourceVersion);
        }
        let valid = match (self, container) {
            (
                Self::ContentHash {
                    record: Reference::Message(owner),
                    ..
                },
                Reference::Message(message) | Reference::Block { message, .. },
            ) => owner == message,
            (
                Self::ContentHash {
                    record: Reference::Document(owner),
                    ..
                },
                Reference::Document(document),
            ) => owner == document,
            (Self::ExactRevision { reference }, Reference::Assertion(revision)) => {
                revision.revision.is_some() && reference == container
            }
            (
                Self::ExactRevision { reference },
                Reference::Variant {
                    revision,
                    part: Some(_),
                },
            ) => revision.revision.is_some() && reference == container,
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(TextFragmentError::SourceVersion)
        }
    }
}

impl TextFragmentRecord {
    pub fn validate(&self) -> Result<()> {
        self.fragment.validate()?;
        if self.reference != self.fragment.reference().to_string() {
            return Err(TextFragmentError::Reference);
        }
        Ok(())
    }
}

impl TextFragment {
    /// A staging key for one exact source domain, version and covered value.
    /// Provenance and a newly learned total do not change this key; it is not
    /// proof of the complete source content or a receiving admission receipt.
    pub fn content_key(&self) -> BlobHash {
        let bytes = serde_json::to_vec(&(
            &self.container,
            &self.source_version,
            self.coverage.start,
            self.coverage.end,
            self.text_hash,
        ))
        .expect("text fragment identity is serializable");
        BlobHash::of(&bytes)
    }

    pub fn reference(&self) -> Reference {
        Reference::Span {
            container: Box::new(self.container.clone()),
            unit: SpanUnit::Bytes,
            start: self.coverage.start,
            end: self.coverage.end,
        }
    }

    pub fn validate(&self) -> Result<()> {
        let coverage = self.coverage;
        if coverage.start > coverage.end
            || coverage
                .total_bytes
                .is_some_and(|total| coverage.end > total)
        {
            return Err(TextFragmentError::Coverage);
        }
        if coverage.end - coverage.start != self.text.len() as u64
            || BlobHash::of(self.text.as_bytes()) != self.text_hash
        {
            return Err(TextFragmentError::SelectedText);
        }
        self.source_version.validate_container(&self.container)?;
        let mut ranges = Vec::with_capacity(self.provenance.spans.len());
        for origin in &self.provenance.spans {
            let Reference::Span {
                container,
                unit,
                start,
                end,
            } = &origin.reference
            else {
                return Err(TextFragmentError::Origin);
            };
            if container.as_ref() != &self.container
                || start > end
                || origin.start > origin.end
                || coverage.total_bytes.is_some_and(|total| origin.end > total)
                || origin.end < coverage.start
                || origin.start > coverage.end
                || *unit == SpanUnit::Bytes && (*start != origin.start || *end != origin.end)
            {
                return Err(TextFragmentError::Origin);
            }
            ranges.push((origin.start, origin.end));
        }
        ranges.sort_unstable();
        let mut covered = coverage.start;
        let mut has_origin = false;
        for (start, end) in ranges {
            if start > covered {
                break;
            }
            if end >= covered {
                has_origin = true;
                covered = covered.max(end.min(coverage.end));
            }
        }
        if !has_origin || covered != coverage.end {
            return Err(TextFragmentError::Origin);
        }
        for source in &self.provenance.sources {
            source
                .validate()
                .map_err(|error| TextFragmentError::SourceClaim(error.to_string()))?;
        }
        Ok(())
    }

    /// Compare touching or overlapping observations without copying text.
    /// Different domains/versions and disjoint coverage return None.
    pub fn validate_overlap(&self, other: &Self) -> Result<Option<()>> {
        self.overlap(other).map(|overlap| overlap.map(|_| ()))
    }

    fn overlap<'a>(&'a self, other: &'a Self) -> Result<Option<(&'a Self, &'a Self, usize)>> {
        self.validate()?;
        other.validate()?;
        if self.container != other.container || self.source_version != other.source_version {
            return Ok(None);
        }
        if self
            .coverage
            .total_bytes
            .zip(other.coverage.total_bytes)
            .is_some_and(|(left, right)| left != right)
        {
            return Err(TextFragmentError::Conflict);
        }
        if self.coverage.start > other.coverage.end || other.coverage.start > self.coverage.end {
            return Ok(None);
        }
        let (left, right) = if self.coverage.start <= other.coverage.start {
            (self, other)
        } else {
            (other, self)
        };
        let overlap_end = left.coverage.end.min(right.coverage.end);
        let left_start = usize::try_from(right.coverage.start - left.coverage.start)
            .map_err(|_| TextFragmentError::Coverage)?;
        let left_end = usize::try_from(overlap_end - left.coverage.start)
            .map_err(|_| TextFragmentError::Coverage)?;
        let right_end = usize::try_from(overlap_end - right.coverage.start)
            .map_err(|_| TextFragmentError::Coverage)?;
        let left_overlap = left
            .text
            .get(left_start..left_end)
            .ok_or(TextFragmentError::CharacterBoundary)?;
        let right_overlap = right
            .text
            .get(..right_end)
            .ok_or(TextFragmentError::CharacterBoundary)?;
        if left_overlap != right_overlap {
            return Err(TextFragmentError::Conflict);
        }
        Ok(Some((left, right, right_end)))
    }

    /// Merge data only. Wire commits remain foreign claims retained by each
    /// receiving source reading, never a clock used to choose an authority.
    pub fn merge(&self, other: &Self) -> Result<Option<Self>> {
        let Some((left, right, right_end)) = self.overlap(other)? else {
            return Ok(None);
        };
        let mut text = left.text.clone();
        if right.coverage.end > left.coverage.end {
            text.push_str(
                right
                    .text
                    .get(right_end..)
                    .ok_or(TextFragmentError::CharacterBoundary)?,
            );
        }
        let mut provenance = self.provenance.clone();
        provenance
            .spans
            .extend(other.provenance.spans.iter().cloned());
        provenance.spans.sort();
        provenance.spans.dedup();
        provenance
            .sources
            .extend(other.provenance.sources.iter().cloned());
        provenance.sources.sort();
        provenance.sources.dedup();
        provenance
            .imports
            .extend(other.provenance.imports.iter().cloned());
        provenance.imports.sort_by_cached_key(|claim| {
            (
                claim.machine.clone(),
                claim.input.clone(),
                claim.chain,
                claim.actor.to_string(),
                claim.at.unix_ns(),
            )
        });
        provenance.imports.dedup();
        let merged = Self {
            container: self.container.clone(),
            source_version: self.source_version.clone(),
            coverage: TextCoverage {
                start: left.coverage.start,
                end: left.coverage.end.max(right.coverage.end),
                total_bytes: self.coverage.total_bytes.or(other.coverage.total_bytes),
            },
            text_hash: BlobHash::of(text.as_bytes()),
            text,
            provenance,
        };
        merged.validate()?;
        Ok(Some(merged))
    }

    /// Checks the reference-text stream against a caller-supplied exact version.
    /// This does not prove a complete Message/Document hash from text: Store
    /// must first validate the complete canonical source identity itself.
    pub fn check_source<'a>(
        &'a self,
        version: &TextSourceVersion,
    ) -> Result<TextFragmentSourceCheck<'a>> {
        self.validate()?;
        if version != &self.source_version {
            return Err(TextFragmentError::SourceVersion);
        }
        let origins = self
            .provenance
            .spans
            .iter()
            .map(|origin| {
                let Reference::Span {
                    unit, start, end, ..
                } = origin.reference
                else {
                    return Err(TextFragmentError::Origin);
                };
                SpanBounds::new(unit, start, end).map_err(|_| TextFragmentError::Origin)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TextFragmentSourceCheck {
            fragment: self,
            position: 0,
            origins,
        })
    }
}

pub struct TextFragmentSourceCheck<'a> {
    fragment: &'a TextFragment,
    position: u64,
    origins: Vec<SpanBounds>,
}
impl TextFragmentSourceCheck<'_> {
    pub fn push(&mut self, text: &str) -> Result<()> {
        let end = self
            .position
            .checked_add(text.len() as u64)
            .ok_or(TextFragmentError::Coverage)?;
        let from = self.position.max(self.fragment.coverage.start);
        let until = end.min(self.fragment.coverage.end);
        if from <= until && from >= self.position && until <= end {
            let source = text
                .get((from - self.position) as usize..(until - self.position) as usize)
                .ok_or(TextFragmentError::CharacterBoundary)?;
            let expected = self
                .fragment
                .text
                .get(
                    (from - self.fragment.coverage.start) as usize
                        ..(until - self.fragment.coverage.start) as usize,
                )
                .ok_or(TextFragmentError::CharacterBoundary)?;
            if source != expected {
                return Err(TextFragmentError::CompleteSource);
            }
        }
        for origin in &mut self.origins {
            origin
                .push(text)
                .map_err(|_| TextFragmentError::CharacterBoundary)?;
        }
        self.position = end;
        Ok(())
    }
    pub fn finish(self) -> Result<()> {
        if self.fragment.coverage.end > self.position
            || self
                .fragment
                .coverage
                .total_bytes
                .is_some_and(|total| total != self.position)
        {
            return Err(TextFragmentError::CompleteSource);
        }
        for (origin, declared) in self
            .origins
            .into_iter()
            .zip(&self.fragment.provenance.spans)
        {
            let normalized = origin
                .finish()
                .map_err(|_| TextFragmentError::CompleteSource)?;
            if normalized.start != declared.start || normalized.end != declared.end {
                return Err(TextFragmentError::CompleteSource);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
