//! Acquisition compares the latest durable source evidence under the same
//! custody lock that publishes the complete group. No network operation enters
//! this path and no tentative input coordinate becomes source authority.
use super::*;
use polylogue_store::primary::{
    CaptureGroup, CaptureMembers, CaptureMode, CapturePlan, PendingCheckpoint, PendingSource,
    SourceSelector, SourceStatus,
};
use std::collections::BTreeMap;

fn visit(
    queue: &crate::spool::Spool,
    mut callback: impl FnMut(crate::spool::Prepared) -> Result<()>,
) -> Result<()> {
    let mut after = None;
    loop {
        let page = queue.pending_after(after, 64)?;
        if page.is_empty() {
            return Ok(());
        }
        for entry in page {
            after = Some(entry.order);
            callback(entry)?;
        }
    }
}

fn pending(queue: &crate::spool::Spool, entry: crate::spool::Prepared) -> Result<PendingCapture> {
    let captured = entry
        .capture
        .clone()
        .ok_or_else(|| Error::Invalid("source operation has no capture header".into()))?;
    let acknowledged = if entry.refused {
        None
    } else {
        Sinex::capture_material(queue, &entry)?
    };
    let anchor = acknowledged.as_ref().map(|material| SourceLocation {
        material: material.id.clone(),
        anchor: 0,
    });
    let material = if matches!(&captured, Capture::Input(input) if input.rows.extent.is_none()) {
        None
    } else {
        acknowledged
    };
    Ok(PendingCapture {
        identity: entry.identity,
        captured,
        material,
        anchor,
    })
}

fn semantic(input: &CapturedInput) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(input)?;
    if let Some(observation) = value
        .pointer_mut("/rows/observation")
        .and_then(serde_json::Value::as_object_mut)
    {
        observation.remove("id");
        observation.remove("seen_at");
    }
    Ok(value)
}

impl Sinex {
    pub(crate) fn prepare_input_group(
        &self,
        plan: &CapturePlan,
        members: &mut CaptureMembers<'_>,
    ) -> Result<PreparedWrite> {
        plan.validate()?;
        let queue = self.queue()?;
        let mut heads = BTreeMap::new();
        visit(&queue, |entry| {
            if let Some(Capture::Input(input)) = entry.capture.as_ref()
                && let Some(basis) = &input.basis
                && plan
                    .bases
                    .iter()
                    .any(|expected| expected.source == basis.source)
            {
                heads.insert(
                    basis.source.clone(),
                    (entry.identity.clone(), input.group.clone()),
                );
            }
            Ok(())
        })?;
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
            if plan.mode == CaptureMode::Mutable {
                for basis in &plan.bases {
                    if heads.get(&basis.source).map(|(id, _)| id) != basis.pending.as_ref() {
                        return Err(polylogue_store::StoreError::Conflict(
                            "source observation changed during acquisition".into(),
                        )
                        .into());
                    }
                }
            }
            let bases = (plan.mode == CaptureMode::Mutable).then_some(&plan.bases);
            format!(
                "acquisition:{}",
                blake3::hash(&serde_json::to_vec(&(
                    "polylogue-acquisition-1",
                    &plan.intent,
                    bases
                ))?)
                .to_hex()
            )
        };
        let mut ordinal = 0u64;
        queue.prepare_acquisition(&identity, &plan.intent, plan.count, |staging| {
            members(&mut |mut input, body| {
                if ordinal >= plan.count {
                    return Err(polylogue_store::StoreError::Invalid(
                        "acquisition emitted more members than declared".into(),
                    ));
                }
                if body.descriptor.hash.0 != input.hash.0
                    || body.descriptor.decoded_len != input.length
                {
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
                if staging.get(&member)?.is_some() {
                    let Capture::Input(original) = staging.capture(&member)? else {
                        return Err(polylogue_store::StoreError::Invalid(
                            "acquisition retry replaced a source capture".into(),
                        ));
                    };
                    // The original observation and predecessor were frozen by
                    // the accepted acquisition, rather than this retry's clock.
                    input.basis = original.basis.clone();
                    if semantic(&input)? != semantic(&original)? {
                        return Err(polylogue_store::StoreError::Conflict(member).into());
                    }
                    input = *original;
                }
                let captured = Capture::Input(Box::new(input));
                let frame = self.capture_frame(&captured, &member)?;
                staging.prepare_encoded_capture(frame, body, &captured)?;
                Ok(())
            })?;
            if ordinal != plan.count {
                return Err(polylogue_store::StoreError::Invalid(
                    "acquisition emitted fewer members than declared".into(),
                )
                .into());
            }
            Ok(())
        })?;
        Ok(PreparedWrite {
            identity,
            pending: true,
        })
    }

    pub(crate) fn pending_source(
        &self,
        source: &SourceSelector,
        reading_version: &str,
    ) -> Result<PendingSource> {
        let queue = self.queue()?;
        let mut result = PendingSource::default();
        let mut control = None;
        let mut unknown = false;
        visit(&queue, |entry| {
            let Some(Capture::Input(input)) = entry.capture.as_ref() else {
                return Ok(());
            };
            let Some(basis) = &input.basis else {
                unknown = true;
                return Ok(());
            };
            if &basis.source != source {
                return Ok(());
            }
            result.status = basis.status;
            if input.raw_range.is_some() {
                result.raw = Some(pending(&queue, entry.clone())?);
            }
            if entry
                .control
                .as_ref()
                .is_some_and(|state| state.reading_version == reading_version)
            {
                control = Some((
                    entry.identity.clone(),
                    input.as_ref().clone(),
                    entry.control.clone().expect("matched control"),
                ));
            }
            result.head = Some(pending(&queue, entry)?);
            Ok(())
        })?;
        if unknown {
            result.status = SourceStatus::MissingEvidence;
        }
        if let Some((identity, header, index)) = control {
            let Capture::Input(full) = queue.capture(&identity)? else {
                return Err(Error::Invalid(
                    "parser control has no source capture".into(),
                ));
            };
            let checkpoint = full
                .parsed
                .iter()
                .rev()
                .filter_map(|reading| reading.checkpoint.as_ref())
                .find(|checkpoint| checkpoint.matches(&index.reading_version, index.parsed_upto))
                .ok_or_else(|| {
                    Error::Invalid("parser control index differs from its immutable capture".into())
                })?;
            result.continuation = Some(PendingCheckpoint {
                capture: identity,
                header,
                checkpoint: checkpoint.clone(),
            });
        }
        Ok(result)
    }

    pub(crate) fn capture_body(&self, identity: &str) -> Result<PreparedBody> {
        let queue = self.queue()?;
        let prepared = queue.prepared_body(identity)?;
        Ok(PreparedBody {
            descriptor: prepared.descriptor(),
            file: prepared.into_file(),
            path: None,
        })
    }
}

#[cfg(test)]
mod tests;
