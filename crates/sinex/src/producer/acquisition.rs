//! Acquisition compares the latest durable source evidence under the same
//! custody lock that publishes the complete group. No network operation enters
//! this path and no tentative input coordinate becomes source authority.
use super::*;
use polylogue_store::primary::{
    CaptureDisposition, CaptureGroup, CaptureMembers, CaptureMode, CapturePlan, CaptureReceipt,
    PendingSource, RawCustodyReceipt, SourceSelector, SourceStatus,
};
use std::collections::BTreeMap;

fn pending(queue: &crate::spool::Spool, entry: crate::spool::Prepared) -> Result<PendingCapture> {
    let captured = entry
        .capture()
        .cloned()
        .ok_or_else(|| Error::Invalid("source operation has no capture header".into()))?;
    let acknowledged = if entry.phase.is_refused() {
        None
    } else {
        Sinex::capture_material(queue, &entry)?
    };
    let anchor = acknowledged.as_ref().map(|material| SourceLocation {
        material: material.id.clone(),
        anchor: 0,
    });
    let material = if matches!(&captured, Capture::Input(input) if input.raw_range.is_none()) {
        None
    } else {
        acknowledged
    };
    Ok(PendingCapture {
        identity: entry.record.identity,
        captured,
        material,
        anchor,
    })
}

fn semantic(input: &CapturedInput, mode: CaptureMode) -> Result<serde_json::Value> {
    // Finalized cache coordinates and observer clocks are frozen by the first
    // acceptance. Byte identity still matches exactly.
    Ok(serde_json::json!({
        "hash":input.hash, "length":input.length, "raw_range":input.raw_range,
        "generation":input.generation, "raw_total":input.raw_total, "raw_predecessor":input.raw_predecessor,
        "index":input.index, "route":input.route,
        "origin":input.input.origin,
        "source":(mode == CaptureMode::Mutable).then(|| input.basis.as_ref().map(|basis| &basis.source)),
    }))
}

impl Sinex {
    pub(crate) fn prepare_input_group(
        &self,
        plan: &CapturePlan,
        members: &mut CaptureMembers<'_>,
    ) -> Result<CaptureReceipt> {
        plan.validate()?;
        let queue = self.queue()?;
        let mut heads = BTreeMap::new();
        for basis in &plan.bases {
            if let Some(entry) = queue.source_heads(&basis.source, "")?.head {
                let Some(Capture::Input(input)) = entry.capture() else {
                    return Err(Error::Invalid("pending source head has no input".into()));
                };
                heads.insert(
                    basis.source.clone(),
                    (entry.identity.clone(), input.group.clone()),
                );
            }
        }
        let retry_group = plan
            .bases
            .iter()
            .map(|basis| heads.get(&basis.source))
            .map(|head| head.and_then(|(_, group)| group.as_ref()))
            .try_fold(None::<String>, |same, group| {
                let group = group
                    .filter(|group| group.intent == plan.intent && group.count == plan.count)?;
                if same.as_ref().is_some_and(|id| *id != group.identity) {
                    return None;
                }
                Some(Some(group.identity.clone()))
            })
            .flatten();
        let identity = if let Some(identity) = retry_group {
            identity
        } else {
            let bases = (plan.mode == CaptureMode::Mutable).then_some(&plan.bases);
            format!(
                "acquisition:{}",
                blake3::hash(&serde_json::to_vec(&(
                    "polylogue-acquisition-1",
                    &self.config.producer,
                    plan.bases
                        .iter()
                        .map(|basis| &basis.source.machine)
                        .collect::<std::collections::BTreeSet<_>>(),
                    &plan.intent,
                    bases
                ))?)
                .to_hex()
            )
        };
        let staged = queue.stage_acquisition(&identity, &plan.intent, plan.count)?;
        drop(queue);
        let mut frozen = blake3::Hasher::new();
        frozen.update(b"polylogue-unresolved-custody-1\0");
        serde_json::to_writer(
            &mut frozen,
            &(&self.config.producer, &identity, &plan.intent, plan.mode),
        )?;
        let staging = staged.queue();
        let mut ordinal = 0u64;
        let mut raw_siblings = BTreeMap::<Vec<u8>, (u64, String)>::new();
        {
            members(&mut |mut input, body| {
                if ordinal >= plan.count {
                    return Err(polylogue_store::StoreError::Invalid(
                        "acquisition emitted more members than declared".into(),
                    ));
                }
                let named = match body.as_slice() {
                    [one] => {
                        one.descriptor.hash.0 == input.hash.0
                            && one.descriptor.decoded_len == input.length
                    }
                    several => {
                        several
                            .iter()
                            .map(|part| part.descriptor.decoded_len)
                            .sum::<u64>()
                            == input.length
                    }
                };
                if !named {
                    return Err(polylogue_store::StoreError::Invalid(
                        "acquired body differs from its declared identity".into(),
                    ));
                }
                if input
                    .raw_range
                    .is_some_and(|range| range.length().ok() != Some(input.length))
                {
                    return Err(polylogue_store::StoreError::Invalid(
                        "acquired body differs from its original raw range".into(),
                    ));
                }
                if let Some(basis) = &input.basis
                    && !plan.bases.iter().any(|expected| expected == basis)
                {
                    return Err(polylogue_store::StoreError::Invalid(
                        "acquisition member has no declared source basis".into(),
                    ));
                }
                input.group = Some(CaptureGroup {
                    identity: identity.clone(),
                    intent: plan.intent.clone(),
                    ordinal,
                    count: plan.count,
                });
                let member = format!("{identity}:{ordinal}");
                ordinal += 1;
                if let (Some(generation), Some(range)) = (&input.generation, input.raw_range) {
                    generation.validate()?;
                    let key = serde_json::to_vec(generation)?;
                    if input.raw_predecessor.is_none()
                        && let Some((end, previous)) = raw_siblings.get(&key)
                        && *end == range.start
                    {
                        input.raw_predecessor = Some(previous.clone());
                    }
                    raw_siblings.insert(key, (range.end, member.clone()));
                }
                let retained = {
                    let queue = self.queue()?;
                    queue
                        .get(&member)?
                        .map(|_| queue.capture(&member))
                        .transpose()?
                };
                if let Some(retained) = retained {
                    let Capture::Input(original) = retained else {
                        return Err(polylogue_store::StoreError::Invalid(
                            "acquisition retry replaced a source capture".into(),
                        ));
                    };
                    // The original observation and predecessor were frozen by
                    // the accepted acquisition, rather than this retry's clock.
                    let same_basis = input.basis == original.basis;
                    let retry_self = input
                        .basis
                        .as_ref()
                        .and_then(|basis| basis.pending.as_ref())
                        .is_some_and(|pending| {
                            heads.values().any(|(head, group)| {
                                head == pending
                                    && group
                                        .as_ref()
                                        .is_some_and(|group| group.identity == identity)
                            })
                        });
                    if semantic(&input, plan.mode)? == semantic(&original, plan.mode)?
                        && (same_basis || retry_self || plan.mode == CaptureMode::Immutable)
                    {
                        input = *original;
                    }
                }
                serde_json::to_writer(
                    &mut frozen,
                    &(
                        semantic(&input, plan.mode)?,
                        (plan.mode == CaptureMode::Mutable).then_some(&input.basis),
                    ),
                )?;
                let captured = Capture::Input(Box::new(input));
                let chunks = super::chunks_of(body.iter().map(|part| part.descriptor.clone()));
                let frame = self.capture_frame(&captured, &member, chunks)?;
                staging.prepare_encoded_capture(frame, body, &captured)?;
                Ok(())
            })?;
            if ordinal != plan.count {
                return Err(polylogue_store::StoreError::Invalid(
                    "acquisition emitted fewer members than declared".into(),
                )
                .into());
            }
        }
        staged.finish()?;
        let custody_identity = format!("raw-custody:{}", frozen.finalize().to_hex());
        let queue = self.queue()?;
        // Existing immutable acceptance is returned before looking at today's
        // head. Retrying it cannot create or advance another observation.
        if let Some(disposition) = queue.matching_acquisition(&staged)? {
            return Ok(match disposition {
                CaptureDisposition::Prepared => CaptureReceipt::Accepted {
                    write: PreparedWrite {
                        identity,
                        pending: true,
                    },
                },
                CaptureDisposition::UnresolvedBasisConflict => {
                    CaptureReceipt::UnresolvedBasisConflict {
                        custody: RawCustodyReceipt {
                            identity,
                            count: plan.count,
                        },
                    }
                }
            });
        }
        let offered_exists = queue.acquisition_disposition(&identity)?.is_some();
        let unresolved_exists = queue.acquisition_disposition(&custody_identity)?.is_some();
        let mut moved = false;
        if plan.mode == CaptureMode::Mutable {
            for basis in &plan.bases {
                if queue
                    .source_heads(&basis.source, "")?
                    .head
                    .as_ref()
                    .map(|head| &head.identity)
                    != basis.pending.as_ref()
                {
                    moved = true;
                }
            }
        }
        if !offered_exists && !unresolved_exists && !moved && plan.required_disposition.is_none() {
            queue.publish_acquisition(staged, CaptureDisposition::Prepared)?;
            return Ok(CaptureReceipt::Accepted {
                write: PreparedWrite {
                    identity,
                    pending: true,
                },
            });
        }
        // The body remains durably owned even when it cannot establish source
        // progress. This namespace is bound to the frozen observation, never
        // a subsequently changing head or a random revision.
        let retained = queue.stage_acquisition(&custody_identity, &plan.intent, plan.count)?;
        drop(queue);
        let mut after = None;
        loop {
            let page = staging.pending_after(after, 64)?;
            if page.is_empty() {
                break;
            }
            for entry in page {
                after = Some(entry.order);
                let Capture::Input(mut input) = staging.capture(&entry.identity)? else {
                    return Err(Error::Invalid(
                        "acquisition contains a non-input member".into(),
                    ));
                };
                let group = input
                    .group
                    .as_mut()
                    .ok_or_else(|| Error::Invalid("acquisition member has no ordinal".into()))?;
                let member = format!("{custody_identity}:{}", group.ordinal);
                group.identity = custody_identity.clone();
                if let Some(previous) = input.raw_predecessor.as_ref()
                    && let Some(preceding) = staging.get(previous)?
                    && preceding.order < entry.order
                    && let Some(Capture::Input(prior)) = preceding.capture().cloned()
                    && prior.generation == input.generation
                    && prior
                        .raw_range
                        .zip(input.raw_range)
                        .is_some_and(|(a, b)| a.end == b.start)
                    && let Some(group) = prior.group
                {
                    input.raw_predecessor = Some(format!("{custody_identity}:{}", group.ordinal));
                }
                let original = {
                    let queue = self.queue()?;
                    queue
                        .get(&member)?
                        .map(|_| queue.capture(&member))
                        .transpose()?
                };
                if let Some(original) = original {
                    let Capture::Input(original) = original else {
                        return Err(Error::Invalid("raw custody has a non-input member".into()));
                    };
                    if semantic(&input, plan.mode)? != semantic(&original, plan.mode)?
                        || (plan.mode == CaptureMode::Mutable && input.basis != original.basis)
                    {
                        return Err(Error::Invalid(
                            "raw custody identity changed its frozen observation".into(),
                        ));
                    }
                    input = original;
                }
                let parts = staging
                    .encoded_body(&entry.identity)?
                    .into_iter()
                    .map(|(path, descriptor)| {
                        Ok(PreparedBody {
                            file: std::fs::File::open(&path)?,
                            descriptor,
                            path: Some(path),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                let captured = Capture::Input(input);
                let chunks = super::chunks_of(parts.iter().map(|part| part.descriptor.clone()));
                let frame = self.capture_frame(&captured, &member, chunks)?;
                retained
                    .queue()
                    .prepare_encoded_capture(frame, parts, &captured)?;
            }
        }
        retained.finish()?;
        self.queue()?
            .publish_acquisition(retained, CaptureDisposition::UnresolvedBasisConflict)?;
        Ok(CaptureReceipt::UnresolvedBasisConflict {
            custody: RawCustodyReceipt {
                identity: custody_identity,
                count: plan.count,
            },
        })
    }

    pub(crate) fn pending_group_headers(
        &self,
        group: &str,
        after: Option<u64>,
        limit: usize,
    ) -> Result<polylogue_store::primary::PendingGroupPage> {
        let queue = self.queue()?;
        let (disposition, entries, next) = queue.acquisition_headers(group, after, limit)?;
        let headers = entries
            .into_iter()
            .map(|entry| pending(&queue, entry))
            .collect::<Result<_>>()?;
        Ok(polylogue_store::primary::PendingGroupPage {
            disposition,
            headers,
            next,
        })
    }

    pub(crate) fn pending_source(
        &self,
        source: &SourceSelector,
        reading_version: &str,
    ) -> Result<PendingSource> {
        let queue = self.queue()?;
        let heads = queue.source_heads(source, reading_version)?;
        let mut result = PendingSource {
            status: SourceStatus::Ready,
            ..Default::default()
        };
        if let Some(head) = heads.head {
            if let Some(Capture::Input(input)) = head.capture() {
                result.status = input
                    .basis
                    .as_ref()
                    .map_or(SourceStatus::MissingEvidence, |basis| basis.status);
            }
            result.head = Some(pending(&queue, head)?);
        }
        if let Some(raw) = heads.raw {
            result.raw = if result
                .head
                .as_ref()
                .is_some_and(|head| head.identity == raw.identity)
            {
                result.head.clone()
            } else {
                Some(pending(&queue, raw)?)
            };
        }
        if heads.unattributed {
            result.status = SourceStatus::MissingEvidence;
        }
        Ok(result)
    }

    pub(crate) fn capture_body(&self, identity: &str) -> Result<Vec<PreparedBody>> {
        let queue = self.queue()?;
        Ok(queue
            .prepared_body(identity)?
            .into_iter()
            .map(|prepared| PreparedBody {
                descriptor: prepared.descriptor(),
                file: prepared.into_file(),
                path: None,
            })
            .collect())
    }
}

mod continuation;

#[cfg(test)]
mod raw_tests;
#[cfg(test)]
mod tests;
