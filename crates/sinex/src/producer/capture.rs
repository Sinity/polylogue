//! A capture read compares the source's latest spooled operation under the
//! same spool lock that commits the complete group. No network operation
//! enters this path, and no input ID that may still change decides a source's state.
use super::*;
use polylogue_store::primary::{
    CaptureGroup, CaptureMembers, CaptureMode, CaptureOutcome, CapturePlan, CaptureResult,
    PendingSource, SourceSelector, SourceStatus, SpooledGroup,
};
use std::collections::BTreeMap;

fn pending(queue: &crate::spool::Spool, entry: crate::spool::Prepared) -> Result<PendingCapture> {
    let captured = entry
        .capture()
        .cloned()
        .ok_or_else(|| Error::invalid("source operation has no capture header"))?;
    let acknowledged = if entry.phase.is_refused() {
        None
    } else {
        Sinex::capture_material(queue, &entry)?
    };
    let location = acknowledged.as_ref().map(|material| SourceLocation {
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
        anchor: location,
    })
}

fn compared_fields(input: &CapturedInput, mode: CaptureMode) -> Result<serde_json::Value> {
    // The first accepted capture fixes cache coordinates and observer clocks.
    // Byte identity must still match.
    let mut fields = serde_json::json!({
        "hash":input.hash, "length":input.length, "raw_range":input.raw_range,
        "generation":input.generation, "raw_total":input.raw_total, "raw_predecessor":input.raw_predecessor,
        "index":input.index, "route":input.route,
        "origin":input.input.origin,
        "source":(mode == CaptureMode::Mutable).then(|| input.basis.as_ref().map(|base| &base.source)),
    });
    if input.provenance == polylogue_store::primary::InputProvenance::Restored {
        fields["provenance"] = serde_json::to_value(input.provenance)?;
        fields["restored"] = serde_json::to_value(&input.rows)?;
        // The accepted group fixes its material predecessors. Retrying exact
        // history cannot replace them with a newer pending group position.
        fields["index"] = serde_json::Value::Null;
    }
    Ok(fields)
}

impl Sinex {
    pub(crate) fn prepare_input_group(
        &self,
        plan: &CapturePlan,
        members: &mut CaptureMembers<'_>,
    ) -> Result<CaptureResult> {
        plan.validate()?;
        let queue = self.queue()?;
        let mut latest = BTreeMap::new();
        for base in &plan.bases {
            if let Some(entry) = queue.latest_pending(&base.source)?.latest {
                let Some(Capture::Input(input)) = entry.capture() else {
                    return Err(Error::invalid(
                        "latest pending source operation has no input",
                    ));
                };
                latest.insert(
                    base.source.clone(),
                    (entry.identity.clone(), input.group.clone()),
                );
            }
        }
        let retry_group = plan
            .bases
            .iter()
            .map(|base| latest.get(&base.source))
            .map(|operation| operation.and_then(|(_, group)| group.as_ref()))
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
                        .map(|base| &base.source.machine)
                        .collect::<std::collections::BTreeSet<_>>(),
                    &plan.intent,
                    bases
                ))?)
                .to_hex()
            )
        };
        let staged = queue.stage_capture_group(&identity, &plan.intent, plan.count)?;
        drop(queue);
        let mut spooled_hash = blake3::Hasher::new();
        spooled_hash.update(b"polylogue-unresolved-custody-1\0");
        serde_json::to_writer(
            &mut spooled_hash,
            &(&self.config.producer, &identity, &plan.intent, plan.mode),
        )?;
        let staging = staged.queue();
        let mut ordinal = 0u64;
        let mut restored_chains = BTreeMap::<(String, i64), (u32, String)>::new();
        let mut adjacent_members = BTreeMap::<Vec<u8>, (u64, String)>::new();
        {
            members(&mut |mut input, body| {
                if ordinal >= plan.count {
                    return Err(polylogue_store::StoreError::invalid(
                        "capture read emitted more members than declared",
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
                    return Err(polylogue_store::StoreError::invalid(
                        "captured body differs from its declared identity",
                    ));
                }
                if input
                    .raw_range
                    .is_some_and(|range| range.length().ok() != Some(input.length))
                {
                    return Err(polylogue_store::StoreError::invalid(
                        "captured body differs from its original byte range",
                    ));
                }
                if let Some(base) = &input.basis
                    && !plan.bases.iter().any(|expected| expected == base)
                {
                    return Err(polylogue_store::StoreError::invalid(
                        "capture read member has no declared source base",
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
                        && let Some((end, previous)) = adjacent_members.get(&key)
                        && *end == range.start
                    {
                        input.raw_predecessor = Some(previous.clone());
                    }
                    adjacent_members.insert(key, (range.end, member.clone()));
                }
                if input.provenance == polylogue_store::primary::InputProvenance::Restored
                    && let Some(extent) = input.rows.extents.first()
                {
                    let key = (input.rows.input.clone(), extent.chain);
                    if let Some((index, previous)) = restored_chains.get(&key) {
                        input.index = *index;
                        input.previous =
                            Some(crate::material_id(&self.config.producer, previous).to_string());
                    }
                    restored_chains.insert(
                        key,
                        (
                            input.index.checked_add(1).ok_or_else(|| {
                                Error::invalid("restored chain has too many material parts")
                            })?,
                            member.clone(),
                        ),
                    );
                }
                let kept = {
                    let queue = self.queue()?;
                    queue
                        .get(&member)?
                        .map(|_| queue.capture(&member))
                        .transpose()?
                };
                if let Some(kept) = kept {
                    let Capture::Input(original) = kept else {
                        return Err(polylogue_store::StoreError::invalid(
                            "capture read retry replaced a source capture",
                        ));
                    };
                    // The accepted capture read fixed the original observation and
                    // previous revision, not this retry's clock.
                    let same_base = input.basis == original.basis;
                    let retry_self = input
                        .basis
                        .as_ref()
                        .and_then(|base| base.pending.as_ref())
                        .is_some_and(|pending| {
                            latest.values().any(|(operation, group)| {
                                operation == pending
                                    && group
                                        .as_ref()
                                        .is_some_and(|group| group.identity == identity)
                            })
                        });
                    let mut offered = compared_fields(&input, plan.mode)?;
                    let original_fields = compared_fields(&original, plan.mode)?;
                    if plan.mode == CaptureMode::Mutable
                        && matches!(
                            (&input.generation, &original.generation),
                            (
                                Some(polylogue_store::primary::SourceGeneration::Database { .. }),
                                Some(polylogue_store::primary::SourceGeneration::Database { .. })
                            )
                        )
                    {
                        // A consistent read after restart has a fresh custody nonce.
                        // An authorized retry keeps the original read, including
                        // its nonce; stream, window and byte identity still match.
                        offered["generation"]["snapshot"] =
                            original_fields["generation"]["snapshot"].clone();
                    }
                    if offered == original_fields
                        && (same_base || retry_self || plan.mode == CaptureMode::Immutable)
                    {
                        input = *original;
                    }
                }
                serde_json::to_writer(
                    &mut spooled_hash,
                    &(
                        compared_fields(&input, plan.mode)?,
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
                return Err(polylogue_store::StoreError::invalid(
                    "capture read emitted fewer members than declared",
                )
                .into());
            }
        }
        staged.finish()?;
        let spooled_identity = format!("raw-custody:{}", spooled_hash.finalize().to_hex());
        let queue = self.queue()?;
        // An already accepted group is returned before looking at the source's
        // latest operation. Retrying it cannot create or advance another observation.
        if let Some(outcome) = queue.matching_capture_group(&staged)? {
            return Ok(match outcome {
                CaptureOutcome::Prepared => CaptureResult::Accepted {
                    write: PreparedWrite {
                        identity,
                        pending: true,
                    },
                },
                CaptureOutcome::UnresolvedBasisConflict => CaptureResult::UnresolvedBasisConflict {
                    spooled: SpooledGroup {
                        identity,
                        count: plan.count,
                    },
                },
            });
        }
        let offered_exists = queue.capture_group_outcome(&identity)?.is_some();
        let unresolved_exists = queue.capture_group_outcome(&spooled_identity)?.is_some();
        let mut moved = false;
        if plan.mode == CaptureMode::Mutable {
            for base in &plan.bases {
                if queue
                    .latest_pending(&base.source)?
                    .latest
                    .as_ref()
                    .map(|operation| &operation.identity)
                    != base.pending.as_ref()
                {
                    moved = true;
                }
            }
        }
        if !offered_exists && !unresolved_exists && !moved && plan.required_outcome.is_none() {
            queue.publish_capture_group(staged, CaptureOutcome::Prepared)?;
            return Ok(CaptureResult::Accepted {
                write: PreparedWrite {
                    identity,
                    pending: true,
                },
            });
        }
        // The body stays in the spool even when it cannot advance the source.
        // This namespace is bound to the recorded observation, never a later
        // latest operation or a random revision.
        let kept = queue.stage_capture_group(&spooled_identity, &plan.intent, plan.count)?;
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
                    return Err(Error::invalid("capture read contains a non-input member"));
                };
                let group = input
                    .group
                    .as_mut()
                    .ok_or_else(|| Error::invalid("capture read member has no ordinal"))?;
                let member = format!("{spooled_identity}:{}", group.ordinal);
                group.identity = spooled_identity.clone();
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
                    input.raw_predecessor = Some(format!("{spooled_identity}:{}", group.ordinal));
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
                        return Err(Error::invalid("the spooled group has a non-input member"));
                    };
                    if compared_fields(&input, plan.mode)? != compared_fields(&original, plan.mode)?
                        || (plan.mode == CaptureMode::Mutable && input.basis != original.basis)
                    {
                        return Err(Error::invalid(
                            "spooled group identity changed its recorded observation",
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
                kept.queue()
                    .prepare_encoded_capture(frame, parts, &captured)?;
            }
        }
        kept.finish()?;
        self.queue()?
            .publish_capture_group(kept, CaptureOutcome::UnresolvedBasisConflict)?;
        Ok(CaptureResult::UnresolvedBasisConflict {
            spooled: SpooledGroup {
                identity: spooled_identity,
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
        let (outcome, entries, next) = queue.capture_group_headers(group, after, limit)?;
        let headers = entries
            .into_iter()
            .map(|entry| pending(&queue, entry))
            .collect::<Result<_>>()?;
        Ok(polylogue_store::primary::PendingGroupPage {
            outcome,
            headers,
            next,
        })
    }

    pub(crate) fn pending_source(&self, source: &SourceSelector) -> Result<PendingSource> {
        let queue = self.queue()?;
        let pending_operations = queue.latest_pending(source)?;
        let mut result = PendingSource {
            status: SourceStatus::Ready,
            ..Default::default()
        };
        if let Some(latest) = pending_operations.latest {
            if let Some(Capture::Input(input)) = latest.capture() {
                result.status = input
                    .basis
                    .as_ref()
                    .map_or(SourceStatus::MissingEvidence, |base| base.status);
            }
            result.head = Some(pending(&queue, latest)?);
        }
        if let Some(raw) = pending_operations.raw {
            result.raw = if result
                .head
                .as_ref()
                .is_some_and(|latest| latest.identity == raw.identity)
            {
                result.head.clone()
            } else {
                Some(pending(&queue, raw)?)
            };
        }
        if pending_operations.unattributed {
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
mod byte_range_tests;
#[cfg(test)]
mod tests;

#[cfg(test)]
mod database_retry_tests;
