//! Partial text availability is decided here, against actual canonical rows.
//! Foreign envelopes remain exportable even when a complete source refutes them.
use crate::{Reader, Result, StoreError};
use polylogue_model::bundle::fragments::{
    TextFragmentError, TextFragmentRecord, TextSourceVersion,
};
use polylogue_model::{BlobHash, Reference, SpanUnit};
use rusqlite::{OptionalExtension, params};
use std::ops::ControlFlow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentConflict {
    Overlap,
    Total,
    CanonicalSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FragmentClaims {
    Absent,
    Compatible,
    Conflict(FragmentConflict),
}

/// Canonical authority is obtained from complete Store rows, never from a
/// serialized attestation flag or from complete coverage of partial text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FragmentAvailability {
    Canonical {
        version: TextSourceVersion,
        claims: FragmentClaims,
    },
    Declared {
        version: TextSourceVersion,
    },
    Conflict {
        version: TextSourceVersion,
        reason: FragmentConflict,
    },
    Ambiguous,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentClaim {
    pub record: TextFragmentRecord,
    pub input: String,
    pub chain: i64,
    pub start: u64,
    pub end: u64,
    /// Receiving admission. `record.commit` is the sender's independent claim.
    pub admitted: Option<u64>,
}

/// Snapshot evidence for the exact normalized selection. Missing admission
/// remains unknown; neither sender receipts nor the read position supply it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentSelection {
    pub availability: FragmentAvailability,
    pub coverage: polylogue_model::bundle::fragments::TextCoverage,
    pub admitted: Option<u64>,
}

fn coordinate(bytes: Vec<u8>) -> Result<u64> {
    Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| {
        StoreError::Invalid("invalid stored fragment coordinate".into())
    })?))
}
fn decode(bytes: &[u8]) -> Result<TextFragmentRecord> {
    let record: TextFragmentRecord = crate::encoding::unpack_json(bytes)?;
    record.validate().map_err(crate::fragment::invalid)?;
    Ok(record)
}
fn hash(bytes: Vec<u8>) -> Result<BlobHash> {
    Ok(BlobHash(bytes.try_into().map_err(|_| {
        StoreError::Invalid("invalid canonical source hash".into())
    })?))
}

impl Reader<'_> {
    pub fn select_fragment_span(
        &self,
        container: &Reference,
        version: Option<&TextSourceVersion>,
        start: u64,
        end: u64,
    ) -> Result<FragmentSelection> {
        use polylogue_model::bundle::fragments::TextCoverage;
        let availability = self.fragment_availability(container, version, start, end)?;
        let (total_bytes, admitted) = match &availability {
            FragmentAvailability::Canonical { .. } => {
                let total = self.visit_reference_text(container, |_| Ok(()))?;
                let admitted = match container {
                    Reference::Message(id) | Reference::Block { message: id, .. } => {
                        self.message_admission(id)?
                    }
                    Reference::Document(id) => self.document_commit(id)?,
                    // The canonical source owner must supply exact revision
                    // evidence; a variant header cannot date a later part.
                    _ => None,
                };
                (total, admitted)
            }
            FragmentAvailability::Declared { version } => {
                let key = crate::fragment::version_key(version)?;
                let from = start.to_be_bytes();
                let until = end.to_be_bytes();
                let total: Option<Vec<u8>> = self.conn()?.query_row(
                    "SELECT f.total FROM text_fragment f WHERE f.container=?1 AND f.version=?2 AND f.total IS NOT NULL
                     AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash) LIMIT 1",
                    params![container.to_string(),key], |row| row.get(0),
                ).optional()?.flatten();
                let mut statement = self.conn()?.prepare_cached(
                    "SELECT i.id,s.chain,s.end,s.admitted FROM text_fragment f JOIN text_fragment_source s ON s.hash=f.hash
                     JOIN input_ref i ON i.pk=s.input_pk WHERE f.container=?1 AND f.version=?2 AND f.start<=?4 AND f.end>=?3",
                )?;
                let mut rows = statement.query(params![
                    container.to_string(),
                    key,
                    from.as_slice(),
                    until.as_slice()
                ])?;
                let mut admitted = None;
                let mut complete = true;
                while let Some(row) = rows.next()? {
                    let receipt = match row.get::<_, Option<i64>>(3)? {
                        Some(n) => {
                            Some(u64::try_from(n).ok().filter(|n| *n > 0).ok_or_else(|| {
                                StoreError::Invalid("invalid fragment source admission".into())
                            })?)
                        }
                        None => {
                            let through = u64::try_from(row.get::<_, i64>(2)?).map_err(|_| {
                                StoreError::Invalid("negative fragment source endpoint".into())
                            })?;
                            self.source_commit(
                                &row.get::<_, String>(0)?,
                                row.get(1)?,
                                Some(through),
                            )?
                        }
                    };
                    if let Some(receipt) = receipt {
                        admitted = Some(admitted.map_or(receipt, |prior: u64| prior.max(receipt)));
                    } else {
                        complete = false;
                    }
                }
                (
                    total.map(coordinate).transpose()?,
                    complete.then_some(admitted).flatten(),
                )
            }
            _ => (None, None),
        };
        Ok(FragmentSelection {
            availability,
            coverage: TextCoverage {
                start,
                end,
                total_bytes,
            },
            admitted,
        })
    }

    /// Streams retained foreign claims, including conflicts and every actual
    /// receiving source. It does not require the complete container to exist.
    pub fn visit_fragment_claims(
        &self,
        container: &Reference,
        version: Option<&TextSourceVersion>,
        visit: impl FnMut(FragmentClaim) -> ControlFlow<()>,
    ) -> Result<()> {
        self.visit_fragment_claims_from_ordinal(container, version, 0, visit)
    }

    /// Ordinals count actual source memberships in the same stable snapshot
    /// order as `visit_fragment_claims`; SQL skips before payload decoding.
    pub fn visit_fragment_claims_from_ordinal(
        &self,
        container: &Reference,
        version: Option<&TextSourceVersion>,
        ordinal: u64,
        mut visit: impl FnMut(FragmentClaim) -> ControlFlow<()>,
    ) -> Result<()> {
        let Ok(ordinal) = i64::try_from(ordinal) else {
            return Ok(());
        };
        let version = version.map(crate::fragment::version_key).transpose()?;
        let mut statement = self.conn()?.prepare_cached(
            "SELECT f.record,i.id,s.chain,s.start,s.end,s.admitted
             FROM text_fragment f JOIN text_fragment_source s ON s.hash=f.hash
             JOIN input_ref i ON i.pk=s.input_pk WHERE f.container=?1
             AND (?2 IS NULL OR f.version=?2)
             ORDER BY f.version,f.start,f.end,f.hash,i.id,s.chain,s.start,s.end LIMIT -1 OFFSET ?3",
        )?;
        let mut rows = statement.query(params![container.to_string(), version, ordinal])?;
        while let Some(row) = rows.next()? {
            let input: String = row.get(1)?;
            let chain: i64 = row.get(2)?;
            let checked = |n: i64| {
                u64::try_from(n)
                    .map_err(|_| StoreError::Invalid("negative fragment source coordinate".into()))
            };
            let start = checked(row.get(3)?)?;
            let end = checked(row.get(4)?)?;
            let admitted = match row.get::<_, Option<i64>>(5)? {
                Some(n) => Some(u64::try_from(n).ok().filter(|n| *n > 0).ok_or_else(|| {
                    StoreError::Invalid("invalid fragment source admission".into())
                })?),
                None => self.source_commit(&input, chain, Some(end))?,
            };
            if visit(FragmentClaim {
                record: decode(&row.get::<_, Vec<u8>>(0)?)?,
                input,
                chain,
                start,
                end,
                admitted,
            })
            .is_break()
            {
                break;
            }
        }
        Ok(())
    }

    /// Availability for normalized UTF-8 bounds. An explicit foreign version
    /// never substitutes the currently selected complete source's version.
    pub fn fragment_availability(
        &self,
        container: &Reference,
        version: Option<&TextSourceVersion>,
        start: u64,
        end: u64,
    ) -> Result<FragmentAvailability> {
        if start > end {
            return Err(crate::fragment::invalid(TextFragmentError::Coverage));
        }
        let canonical = self.canonical_fragment_version(container)?;
        if let Some(actual) =
            canonical.filter(|actual| version.is_none_or(|requested| requested == actual))
        {
            // The existing reference reader validates full bounds and character
            // boundaries before emitting a span. A sparse owner cannot enter here.
            let span = Reference::Span {
                container: Box::new(container.clone()),
                unit: SpanUnit::Bytes,
                start,
                end,
            };
            match self.visit_reference_text(&span, |_| Ok(())) {
                Ok(Some(_)) => {
                    let claims = self.check_fragment_claims(container, &actual, start, end)?;
                    return Ok(FragmentAvailability::Canonical {
                        version: actual,
                        claims,
                    });
                }
                Ok(None) | Err(StoreError::ReferenceUnavailable { .. }) => {}
                Err(error) => return Err(error),
            }
        }
        let selected = if let Some(version) = version {
            version.clone()
        } else {
            let mut statement = self.conn()?.prepare_cached(
                "SELECT DISTINCT f.version FROM text_fragment f WHERE f.container=?1
                 AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash) LIMIT 2",
            )?;
            let mut rows = statement.query([container.to_string()])?;
            let Some(row) = rows.next()? else {
                return Ok(FragmentAvailability::Unknown);
            };
            let version: TextSourceVersion = serde_json::from_str(&row.get::<_, String>(0)?)?;
            if rows.next()?.is_some() {
                return Ok(FragmentAvailability::Ambiguous);
            }
            version
        };
        let key = crate::fragment::version_key(&selected)?;
        let conn = self.conn()?;
        let mut totals=conn.prepare_cached(
            "SELECT DISTINCT f.total FROM text_fragment f WHERE f.container=?1 AND f.version=?2 AND f.total IS NOT NULL
             AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash) LIMIT 2",
        )?;
        let mut rows = totals.query(params![container.to_string(), key])?;
        let total = rows
            .next()?
            .map(|row| coordinate(row.get(0)?))
            .transpose()?;
        if rows.next()?.is_some() {
            return Ok(FragmentAvailability::Conflict {
                version: selected,
                reason: FragmentConflict::Total,
            });
        }
        if let Some(total) = total {
            let beyond:bool=conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM text_fragment f WHERE f.container=?1 AND f.version=?2 AND f.end>?3
                 AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash))",
                params![container.to_string(),key,total.to_be_bytes().as_slice()],|row|row.get(0),
            )?;
            if beyond {
                return Ok(FragmentAvailability::Conflict {
                    version: selected,
                    reason: FragmentConflict::Total,
                });
            }
            if end > total {
                return Ok(FragmentAvailability::Unknown);
            }
        }
        let from = start.to_be_bytes();
        let until = end.to_be_bytes();
        let mut pairs=conn.prepare_cached(
            "SELECT l.record,r.record FROM text_fragment l JOIN text_fragment r
             ON l.container=r.container AND l.version=r.version AND l.hash<r.hash
             AND l.start<=r.end AND r.start<=l.end
             WHERE l.container=?1 AND l.version=?2 AND l.start<=?4 AND l.end>=?3 AND r.start<=?4 AND r.end>=?3
             AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=l.hash)
             AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=r.hash)",
        )?;
        let mut pairs = pairs.query(params![
            container.to_string(),
            key,
            from.as_slice(),
            until.as_slice()
        ])?;
        while let Some(row) = pairs.next()? {
            let left = decode(&row.get::<_, Vec<u8>>(0)?)?;
            let right = decode(&row.get::<_, Vec<u8>>(1)?)?;
            if left.fragment.validate_overlap(&right.fragment).is_err() {
                return Ok(FragmentAvailability::Conflict {
                    version: selected,
                    reason: FragmentConflict::Overlap,
                });
            }
        }
        let mut statement=conn.prepare_cached(
            "SELECT f.record FROM text_fragment f WHERE f.container=?1 AND f.version=?2 AND f.start<=?4 AND f.end>=?3
             AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash) ORDER BY f.start,f.end,f.hash",
        )?;
        let mut rows = statement.query(params![
            container.to_string(),
            key,
            from.as_slice(),
            until.as_slice()
        ])?;
        let mut cursor = start;
        let mut present = false;
        while let Some(row) = rows.next()? {
            let record = decode(&row.get::<_, Vec<u8>>(0)?)?;
            let f = &record.fragment;
            if f.coverage.start > cursor {
                break;
            }
            if f.coverage.end < cursor {
                continue;
            }
            let next = f.coverage.end.min(end);
            if f.text
                .get((cursor - f.coverage.start) as usize..(next - f.coverage.start) as usize)
                .is_none()
            {
                return Err(crate::fragment::invalid(
                    TextFragmentError::CharacterBoundary,
                ));
            }
            present = true;
            cursor = next;
        }
        if present && cursor == end {
            Ok(FragmentAvailability::Declared { version: selected })
        } else {
            Ok(FragmentAvailability::Unknown)
        }
    }

    /// Emits only an available range, after validating coverage and conflicts.
    /// Memory holds at most two captured records; adjacent text is not collected.
    /// Break stops delivery successfully. Availability validation covers the
    /// requested range first; callers requesting a prefix pass its exact bounds.
    pub fn visit_fragment_text(
        &self,
        container: &Reference,
        version: Option<&TextSourceVersion>,
        start: u64,
        end: u64,
        mut visit: impl FnMut(&str) -> Result<ControlFlow<()>>,
    ) -> Result<FragmentAvailability> {
        let availability = self.fragment_availability(container, version, start, end)?;
        match &availability {
            FragmentAvailability::Canonical { .. } => {
                self.visit_reference_text_while(
                    &Reference::Span {
                        container: Box::new(container.clone()),
                        unit: SpanUnit::Bytes,
                        start,
                        end,
                    },
                    visit,
                )?;
            }
            FragmentAvailability::Declared { version } => {
                let key = crate::fragment::version_key(version)?;
                let from = start.to_be_bytes();
                let until = end.to_be_bytes();
                let mut statement=self.conn()?.prepare_cached(
                    "SELECT f.record FROM text_fragment f WHERE f.container=?1 AND f.version=?2 AND f.start<=?4 AND f.end>=?3
                     AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash) ORDER BY f.start,f.end,f.hash",
                )?;
                let mut rows = statement.query(params![
                    container.to_string(),
                    key,
                    from.as_slice(),
                    until.as_slice()
                ])?;
                let mut cursor = start;
                while let Some(row) = rows.next()? {
                    let record = decode(&row.get::<_, Vec<u8>>(0)?)?;
                    let f = &record.fragment;
                    if f.coverage.end <= cursor {
                        continue;
                    }
                    let next = f.coverage.end.min(end);
                    let text = f
                        .text
                        .get(
                            (cursor - f.coverage.start) as usize
                                ..(next - f.coverage.start) as usize,
                        )
                        .ok_or_else(|| {
                            crate::fragment::invalid(TextFragmentError::CharacterBoundary)
                        })?;
                    let flow = visit(text)?;
                    cursor = next;
                    if flow.is_break() {
                        break;
                    }
                    if cursor == end {
                        break;
                    }
                }
            }
            _ => {}
        }
        Ok(availability)
    }

    fn canonical_fragment_version(
        &self,
        container: &Reference,
    ) -> Result<Option<TextSourceVersion>> {
        match container {
            Reference::Message(message) | Reference::Block { message, .. } => {
                let stored:Option<(Vec<u8>,Option<Vec<u8>>)>=self.conn()?.query_row(
                    "SELECT (SELECT version_hash FROM message_version WHERE pk=s.version_pk),p.layout FROM message m JOIN message_source s ON s.message_pk=m.pk AND s.input_pk=m.input_pk AND s.chain=m.source_chain
                     JOIN message_projection p ON p.version_pk=s.version_pk WHERE m.id=?1",
                    [message.to_string()],|row|Ok((row.get(0)?,row.get(1)?)),
                ).optional()?;
                let Some((version, layout)) = stored else {
                    return Ok(None);
                };
                // A missing layout is historical unknown, not a dense full value.
                let Some(layout) = layout else {
                    return Ok(None);
                };
                let (count, ordinals): (u64, Vec<u64>) = serde_json::from_slice(&layout)?;
                if count != ordinals.len() as u64 || ordinals.iter().copied().ne(0..count) {
                    return Ok(None);
                }
                Ok(Some(TextSourceVersion::ContentHash {
                    record: Reference::Message(message.clone()),
                    hash: hash(version)?,
                }))
            }
            Reference::Document(id) => {
                let version: Option<Vec<u8>> = self
                    .conn()?
                    .query_row(
                        "SELECT version_hash FROM document WHERE id=?1",
                        [id.to_string()],
                        |row| row.get(0),
                    )
                    .optional()?;
                version
                    .map(|version| {
                        Ok(TextSourceVersion::ContentHash {
                            record: container.clone(),
                            hash: hash(version)?,
                        })
                    })
                    .transpose()
            }
            Reference::Variant {
                revision,
                part: Some(part),
            } if revision.revision.is_some() => {
                let Some(variant) = self
                    .variant(revision.revision.as_deref().expect("checked revision"))?
                    .filter(|variant| variant.root == revision.root)
                else {
                    return Ok(None);
                };
                let Ok(part) = i64::try_from(*part) else {
                    return Ok(None);
                };
                let conn = self.conn()?;
                let source = crate::source::Source::reading(conn)?;
                let present: bool = conn.query_row(
                    &source.sql("SELECT EXISTS(SELECT 1 FROM @source.variant_part WHERE variant=?1 AND n=?2)"),
                    params![variant.id, part], |row| row.get(0),
                )?;
                Ok(present.then(|| TextSourceVersion::ExactRevision {
                    reference: container.clone(),
                }))
            }
            _ => Ok(None),
        }
    }

    fn check_fragment_claims(
        &self,
        container: &Reference,
        version: &TextSourceVersion,
        start: u64,
        end: u64,
    ) -> Result<FragmentClaims> {
        let key = crate::fragment::version_key(version)?;
        let from = start.to_be_bytes();
        let until = end.to_be_bytes();
        let mut statement=self.conn()?.prepare_cached(
            "SELECT f.record FROM text_fragment f WHERE f.container=?1 AND f.version=?2 AND f.start<=?4 AND f.end>=?3
             AND EXISTS(SELECT 1 FROM text_fragment_source s WHERE s.hash=f.hash) ORDER BY f.hash",
        )?;
        let mut rows = statement.query(params![
            container.to_string(),
            key,
            from.as_slice(),
            until.as_slice()
        ])?;
        let mut status = FragmentClaims::Absent;
        while let Some(row) = rows.next()? {
            let record = decode(&row.get::<_, Vec<u8>>(0)?)?;
            let mut checker = record
                .fragment
                .check_source(version)
                .map_err(crate::fragment::invalid)?;
            let mut conflict = false;
            let source = self.visit_reference_text(container, |text| {
                if let Err(error) = checker.push(text) {
                    conflict = true;
                    return Err(crate::fragment::invalid(error));
                }
                Ok(())
            });
            if conflict {
                return Ok(FragmentClaims::Conflict(FragmentConflict::CanonicalSource));
            }
            if source?.is_none() {
                return Err(StoreError::Invalid(
                    "canonical fragment source disappeared".into(),
                ));
            }
            if checker.finish().is_err() {
                return Ok(FragmentClaims::Conflict(FragmentConflict::CanonicalSource));
            }
            status = FragmentClaims::Compatible;
        }
        Ok(status)
    }
}

#[cfg(test)]
mod tests;
