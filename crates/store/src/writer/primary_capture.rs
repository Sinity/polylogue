//! Raw acquisition is durable in the SDK spool before its temporary bytes are
//! released. Only a confirmed Input frame changes the source cache.
use super::*;
use crate::input_rows::{InputRows, Revision};
use crate::primary::{
    Capture, CaptureGroup, CapturedInput, CapturedReading, Material, PendingCapture, Product,
    ProductWrite,
};
use crate::staging::{PreparedBatch, StoredInput};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};

fn latest(index: &Connection, input: &str) -> Result<Option<Revision>> {
    index.query_row("SELECT number,produced,chain,length FROM source_revision WHERE input=?1 ORDER BY number DESC LIMIT 1",[input],|row|Ok(Revision{number:row.get(0)?,produced:row.get(1)?,chain:row.get(2)?,length:row.get(3)?})).optional().map_err(StoreError::from)
}

fn failed(count: usize, error: StoreError) -> Vec<Result<StoredInput>> {
    let reason = error.to_string();
    (0..count)
        .map(|_| {
            Err(match &error {
                StoreError::Conflict(identity) => StoreError::Conflict(identity.clone()),
                StoreError::PrimaryPending { identity } => StoreError::PrimaryPending {
                    identity: identity.clone(),
                },
                StoreError::UnresolvedBasisConflict { custody } => {
                    StoreError::UnresolvedBasisConflict {
                        custody: custody.clone(),
                    }
                }
                _ => StoreError::Invalid(reason.clone()),
            })
        })
        .collect()
}

fn source_name(
    index: &Connection,
    batch: &Batch,
    pending: &[PendingCapture],
) -> Result<crate::primary::InputName> {
    if let Some(observation) = &batch.observation {
        return Ok(crate::primary::InputName {
            origin: batch.input.format.clone(),
            path: observation.path.clone(),
            host: observation.host.clone(),
        });
    }
    if let Some(name) = pending
        .iter()
        .rev()
        .find_map(|pending| match &pending.captured {
            Capture::Input(input) if input.rows.input == batch.input.id => {
                Some(input.input.clone())
            }
            _ => None,
        })
    {
        return Ok(name);
    }
    index.query_row("SELECT path,host FROM source_observation o JOIN source_machine m ON m.id=o.machine WHERE input=?1 ORDER BY o.rowid DESC LIMIT 1",[&batch.input.id],|row|Ok(crate::primary::InputName{origin:batch.input.format.clone(),path:row.get(0)?,host:row.get(1)?})).optional()?.ok_or_else(||StoreError::Invalid("a continuing primary capture has no source observation".into()))
}

fn material_reader(archive: &Archive, material: &Material) -> Result<hashed_blobs::BlobReader> {
    let [bytes] = material.extents.as_slice() else {
        return Err(StoreError::Invalid(
            "a captured material needs one physical extent".into(),
        ));
    };
    let extent = hashed_blobs::Extent::decode(bytes)?;
    if extent.hash.0 != material.hash.0 || extent.decoded_len != material.length || extent.len == 0
    {
        return Err(StoreError::Invalid(
            "captured material descriptor differs from its bytes".into(),
        ));
    }
    Ok(hashed_blobs::Reader::new(archive.byte_root()?).open(&[extent], None)?)
}

// The acquired body keeps its own coordinates after logical extents are trimmed.
fn captured_location(
    input: &CapturedInput,
    material: Option<&Material>,
    offset: u64,
) -> Result<Option<crate::primary::SourceLocation>> {
    let (Some(range), Some(material)) = (input.raw_range, material) else {
        return Ok(None);
    };
    if material.hash != input.hash
        || material.length != input.length
        || range.length()? != input.length
    {
        return Err(StoreError::Invalid(
            "ready provider material differs from its acquired body".into(),
        ));
    }
    Ok(
        (offset >= range.start && offset <= range.end).then(|| crate::primary::SourceLocation {
            material: material.id.clone(),
            anchor: offset - range.start,
        }),
    )
}

// The acquired body may be retained without creating a logical source extent.
// A receipt identifies the physical preparation, never a whole-chain blob guess.
fn acquired_body(item: &PreparedBatch) -> Result<Option<&crate::staging::PreparedBytes>> {
    let batch = &item.batch;
    let expected = if let Some(receipt) = &batch.acquisition {
        Some((receipt.hash, receipt.range.length()?))
    } else {
        batch.new_blob.as_ref().map(|blob| (blob.hash, blob.size))
    };
    expected
        .map(|(hash, length)| {
            item.files
                .iter()
                .find(|file| file.hash() == hash && file.len() == length)
                .ok_or_else(|| {
                    StoreError::Invalid(
                        "acquired provider body has no matching private encoding".into(),
                    )
                })
        })
        .transpose()
}

fn validate(item: &PreparedBatch, archive: &Archive) -> Result<()> {
    let batch = &item.batch;
    acquired_body(item)?;
    if batch.input.id.is_empty() || i64::try_from(batch.length).is_err() {
        return Err(StoreError::Invalid("invalid primary capture input".into()));
    }
    for blob in batch.new_blob.iter().chain(&batch.blobs) {
        if !item.files.iter().any(|file| {
            file.owner == archive.root && file.hash() == blob.hash && file.len() == blob.size
        }) {
            return Err(StoreError::Invalid(
                "primary capture has no matching private source bytes".into(),
            ));
        }
    }
    for file in &item.files {
        if file.owner != archive.root
            || !batch
                .new_blob
                .iter()
                .chain(&batch.blobs)
                .any(|blob| blob.hash == file.hash() && blob.size == file.len())
        {
            return Err(StoreError::Invalid(
                "primary capture carries unrelated prepared bytes".into(),
            ));
        }
    }
    if let Some(blob) = &batch.new_blob
        && blob.size > batch.length
    {
        return Err(StoreError::Invalid(
            "captured bytes exceed their source endpoint".into(),
        ));
    }
    Ok(())
}

// An acquisition intent is independent of observer time and generated observation IDs.
// Ordered source content, source facts and reading versions distinguish actual new work.
fn capture_group_identity(group: &[PreparedBatch], immutable: bool) -> Result<String> {
    let mut hash = blake3::Hasher::new();
    hash.update(b"polylogue-capture-group-v1");
    for item in group {
        let b = &item.batch;
        let change = match b.change {
            Change::First => ("first", 0, 0),
            Change::Unchanged => ("unchanged", 0, 0),
            Change::Observed { revision, chain } => ("observed", revision, chain),
            Change::Reparse { chain } => ("reparse", 0, chain),
            Change::Appended { from } => (
                "appended",
                i64::try_from(from)
                    .map_err(|_| StoreError::Invalid("capture offset overflow".into()))?,
                0,
            ),
            Change::Rewritten => ("rewritten", 0, 0),
            Change::Disappeared => ("disappeared", 0, 0),
        };
        let observation = b.observation.as_ref().filter(|_| !immutable).map(|o| {
            (
                &o.machine,
                &o.host,
                &o.path,
                &o.path_bytes,
                o.facts
                    .as_ref()
                    .map(|f| (f.device, f.inode, f.size, f.mtime_ns, f.ctime_ns)),
                o.import.clone().map(|mut member| {
                    member.import_id.clear();
                    member
                }),
                &o.container,
            )
        });
        let encoded = serde_json::to_vec(&(
            &b.input,
            (!immutable).then_some(change),
            b.continues,
            b.length,
            b.new_blob.as_ref().map(|v| (v.hash, v.size)),
            b.blobs.iter().map(|v| (v.hash, v.size)).collect::<Vec<_>>(),
            observation,
            b.parse
                .iter()
                .map(|step| {
                    (
                        &step.reading_version,
                        step.start,
                        step.parsed_upto,
                        &step.progress,
                        step.reset,
                        step.delivery,
                    )
                })
                .collect::<Vec<_>>(),
        ))?;
        hash.update(&(encoded.len() as u64).to_le_bytes());
        hash.update(&encoded);
    }
    Ok(format!("capture:{}", hash.finalize().to_hex()))
}

impl Inner {
    pub(super) fn primary_store_group(
        &mut self,
        group: Vec<PreparedBatch>,
    ) -> Vec<Result<StoredInput>> {
        if super::staging::independent_files(&group) {
            let mut outcomes = Vec::with_capacity(group.len());
            for item in group {
                match self.capture_primary_group(std::slice::from_ref(&item)) {
                    Ok(result) => outcomes.extend(result),
                    Err(error) => outcomes.extend(failed(1, error)),
                }
            }
            return outcomes;
        }
        let count = group.len();
        match self.capture_primary_group(&group) {
            Ok(outcomes) => outcomes,
            Err(error) => failed(count, error),
        }
    }

    fn capture_primary_group(
        &mut self,
        group: &[PreparedBatch],
    ) -> Result<Vec<Result<StoredInput>>> {
        if group.is_empty() {
            return Ok(Vec::new());
        }
        for item in group {
            validate(item, &self.location)?;
        }
        let sinex = Arc::clone(self.sinex.as_ref().ok_or_else(|| {
            StoreError::Invalid("primary acquisition requires a producer".into())
        })?);
        let immutable = group.iter().all(|item| {
            item.batch.acquisition.as_ref().is_some_and(|receipt| {
                receipt.route == crate::primary::AcquisitionRoute::ExplicitImport
            })
        });
        let intent = capture_group_identity(group, immutable)?;
        let pending = sinex.pending_captures()?;
        let mut bases = BTreeMap::new();
        let mut required_disposition = None;
        for item in group {
            match item.batch.required_disposition {
                Some(crate::primary::CaptureDisposition::Prepared) => {
                    return Err(StoreError::Invalid(
                        "an acquisition constraint cannot require source acceptance".into(),
                    ));
                }
                Some(crate::primary::CaptureDisposition::UnresolvedBasisConflict) => {
                    required_disposition = item.batch.required_disposition
                }
                None => {}
            }
        }
        for item in group {
            let source = self.capture_source_selector(&item.batch, &pending)?;
            if let Some(existing) = bases.get(&source) {
                if item
                    .batch
                    .basis
                    .as_ref()
                    .is_some_and(|basis| basis != existing)
                {
                    required_disposition =
                        Some(crate::primary::CaptureDisposition::UnresolvedBasisConflict);
                }
            } else {
                let version = item
                    .batch
                    .parse
                    .last()
                    .map_or("", |step| step.reading_version.as_str());
                let basis = if let Some(sampled) = &item.batch.basis {
                    if !self.sampled_basis_matches(sampled, &source, &item.batch.input.id)? {
                        required_disposition =
                            Some(crate::primary::CaptureDisposition::UnresolvedBasisConflict);
                    }
                    // Pending predecessor is compared under the SDK custody
                    // lock, after the serialized writer checks confirmed state.
                    sampled.clone()
                } else {
                    self.source_snapshot(source.clone(), version)?.basis
                };
                bases.insert(source, basis);
            }
        }
        let mut revisions = BTreeMap::new();
        let mut known: BTreeSet<String> = group
            .iter()
            .map(|item| item.batch.input.id.clone())
            .collect();
        for pending in &pending {
            if let Capture::Input(input) = &pending.captured {
                known.insert(input.rows.input.clone());
                if let Some(revision) = &input.rows.revision {
                    revisions.insert(input.rows.input.clone(), revision.clone());
                }
            }
        }
        let mut preview = revisions.clone();
        let mut prefixes = BTreeMap::new();
        let mut export_apply = Vec::with_capacity(group.len());
        for item in group {
            let batch = &item.batch;
            let previous = preview
                .get(&batch.input.id)
                .cloned()
                .or(latest(&self.index, &batch.input.id)?);
            let mut metadata = batch.clone();
            metadata.new_blob = None;
            metadata.blobs.clear();
            let (chain, rows) =
                bytes::capture_rows(&self.index, &metadata, previous.clone(), &known)?;
            let actual_chain = chain.or_else(|| previous.as_ref().map(|revision| revision.chain));
            if let Some(revision) = rows.revision {
                preview.insert(batch.input.id.clone(), revision);
            }
            let mut decisions = Vec::with_capacity(batch.parse.len());
            for step in &batch.parse {
                if step.delivery.is_none() {
                    decisions.push(true);
                    continue;
                }
                let admitted = (|| {
                    let chain = actual_chain.ok_or_else(|| {
                        StoreError::Invalid("export capture has no actual source chain".into())
                    })?;
                    let key = crate::primary::ExportReadingKey {
                        input: batch.input.clone(),
                        chain,
                        reading_version: step.reading_version.clone(),
                        length: batch.length,
                    };
                    let identity = key.identity().to_string();
                    if !prefixes.contains_key(&identity) {
                        prefixes.insert(identity.clone(), self.pending_export_prefix(&key)?);
                    }
                    crate::wire_shapes::advance_prefix(
                        prefixes.get_mut(&identity).expect("inserted export prefix"),
                        batch.length,
                        step,
                    )
                })();
                match admitted {
                    Ok(apply) => decisions.push(apply),
                    Err(_) => {
                        required_disposition =
                            Some(crate::primary::CaptureDisposition::UnresolvedBasisConflict);
                        decisions.push(false);
                    }
                }
            }
            export_apply.push(decisions);
        }
        let plan = crate::primary::CapturePlan {
            intent: intent.clone(),
            bases: bases.values().cloned().collect(),
            count: group.len() as u64,
            required_disposition,
            mode: if immutable {
                crate::primary::CaptureMode::Immutable
            } else {
                crate::primary::CaptureMode::Mutable
            },
        };
        plan.validate()?;
        // Provider-output blobs have their own durable capture. Group callbacks
        // stage encodings outside the SDK lock; publication rechecks the basis.
        for item in group {
            for blob in &item.batch.blobs {
                // The acquired provider body belongs to this atomic group, even
                // when it creates no logical extent. Generic blob replay must
                // not publish bytes from an unresolved acquisition separately.
                if item.batch.acquisition.as_ref().is_some_and(|receipt| {
                    receipt.hash == blob.hash && receipt.range.length().ok() == Some(blob.size)
                }) {
                    continue;
                }
                let file = item
                    .files
                    .iter()
                    .find(|file| file.hash() == blob.hash)
                    .expect("validated blob");
                sinex.prepare_blob(blob.hash, blob.size, file.body()?)?;
            }
        }
        let mut accepted: Vec<_> = group
            .iter()
            .map(|item| {
                (
                    item.batch.input.id.clone(),
                    item.batch.length,
                    None::<InputRows>,
                    item.batch.new_blob.is_none(),
                )
            })
            .collect();
        let mut members = |emit: &mut crate::primary::CaptureEmitter<'_>| -> Result<()> {
            for (ordinal, item) in group.iter().enumerate() {
                let batch = &item.batch;
                let previous = match revisions.get(&batch.input.id) {
                    Some(revision) => Some(revision.clone()),
                    None => latest(&self.index, &batch.input.id)?,
                };
                // Metadata describes the original acquired source range. Exact overlap
                // with a later confirmed prefix is resolved when this group is ready.
                let mut metadata = batch.clone();
                metadata.new_blob = None;
                metadata.blobs.clear();
                let (chain, mut rows) =
                    bytes::capture_rows(&self.index, &metadata, previous.clone(), &known)?;
                let chain = chain
                    .or_else(|| previous.as_ref().map(|r| r.chain))
                    .unwrap_or(1);
                if let Some(blob) = &batch.new_blob
                    && blob.size > 0
                {
                    rows.extent = Some(crate::input_rows::Extent {
                        chain,
                        offset: (batch.length - blob.size) as i64,
                        length: blob.size as i64,
                        blob: blob.hash,
                        blob_offset: 0,
                    });
                }
                if let Some(revision) = &rows.revision {
                    revisions.insert(batch.input.id.clone(), revision.clone());
                }
                let name = source_name(&self.index, batch, &pending)?;
                let (index, previous_material) =
                    self.capture_predecessor(&batch.input.id, chain, &pending)?;
                let metadata_bytes = serde_json::to_vec(&rows)?;
                let raw_body = acquired_body(item)?;
                let (hash, length) = raw_body
                    .map(|body| (body.hash(), body.len()))
                    .unwrap_or_else(|| {
                        (BlobHash::of(&metadata_bytes), metadata_bytes.len() as u64)
                    });
                let captured = CapturedInput {
                    generation: batch
                        .acquisition
                        .as_ref()
                        .and_then(|receipt| receipt.generation.clone()),
                    raw_total: batch
                        .acquisition
                        .as_ref()
                        .and_then(|receipt| receipt.raw_total),
                    raw_predecessor: batch
                        .acquisition
                        .as_ref()
                        .and_then(|receipt| receipt.raw_predecessor.clone()),
                    raw_range: batch
                        .acquisition
                        .as_ref()
                        .map(|receipt| receipt.range)
                        .or_else(|| {
                            raw_body.map(|body| crate::primary::RawRange {
                                start: batch.length - body.len(),
                                end: batch.length,
                            })
                        }),
                    basis: Some(bases[&self.capture_source_selector(batch, &pending)?].clone()),
                    input: name,
                    arrival_rank: batch.input.arrival_rank,
                    parsed: batch
                        .parse
                        .iter()
                        .enumerate()
                        .filter(|(step_index, _)| export_apply[ordinal][*step_index])
                        .map(|(_, step)| {
                            CapturedReading::from_step(
                                step,
                                step.delivery.map(|_| crate::primary::ExportReadingKey {
                                    input: batch.input.clone(),
                                    chain,
                                    reading_version: step.reading_version.clone(),
                                    length: batch.length,
                                }),
                            )
                        })
                        .collect::<Result<_>>()?,
                    rows,
                    index,
                    previous: previous_material,
                    hash,
                    length,
                    group: Some(CaptureGroup {
                        identity: intent.clone(),
                        intent: intent.clone(),
                        ordinal: ordinal as u64,
                        count: group.len() as u64,
                    }),
                };
                let body = if let Some(body) = raw_body {
                    body.body()?
                } else {
                    self.location
                        .prepare_bytes(metadata_bytes.as_slice())?
                        .body()?
                };
                // The SDK assigns the original frozen occurrence to all members at
                // one publication boundary. No per-member durable acceptance exists.
                accepted[ordinal].2 = Some(captured.rows.clone());
                emit(captured, body)?;
            }
            Ok(())
        };
        let prepared = match sinex.prepare_input_group(&plan, &mut members)? {
            crate::primary::CaptureReceipt::Accepted { write } => write,
            crate::primary::CaptureReceipt::UnresolvedBasisConflict { custody } => {
                return Ok(failed(
                    group.len(),
                    StoreError::UnresolvedBasisConflict { custody },
                ));
            }
        };
        match self.drain_primary_captures() {
            Ok(()) | Err(StoreError::Sinex(_)) => {}
            Err(error) => return Err(error),
        }
        let mut outcomes = Vec::new();
        for (input, length, rows, already_stored) in accepted {
            let pending = prepared.identity.clone();
            // An idempotent SDK retry can return its frozen group without
            // calling the visitor. Its original metadata is not replaced by
            // newly generated observation IDs.
            let Some(rows) = rows else {
                outcomes.push(Err(StoreError::PrimaryPending { identity: pending }));
                continue;
            };
            let chain = if let Some(observation) = &rows.observation {
                self.index.query_row("SELECT coalesce(o.chain,r.chain) FROM source_observation o LEFT JOIN source_revision r ON r.input=o.input AND r.number=o.revision WHERE o.id=?1",[&observation.id],|row|row.get::<_,Option<i64>>(0)).optional()?.map(|chain|chain.unwrap_or(1))
            } else if let Some(extent) = &rows.extent {
                self.index.query_row("SELECT e.chain FROM source_extent e JOIN source_revision r ON r.input=e.input AND r.chain=e.chain WHERE e.input=?1 AND e.blob=?2 AND e.offset-e.blob_offset=?3 AND r.length>=?4 ORDER BY r.number DESC LIMIT 1",params![input,&extent.blob.0[..],extent.offset-extent.blob_offset,length as i64],|row|row.get(0)).optional()?
            } else {
                None
            };
            outcomes.push(match chain {
                Some(chain) => Ok(StoredInput {
                    input,
                    chain,
                    length,
                    already_stored,
                }),
                None => Err(StoreError::PrimaryPending { identity: pending }),
            });
        }
        Ok(outcomes)
    }

    fn capture_source_selector(
        &self,
        batch: &Batch,
        pending: &[PendingCapture],
    ) -> Result<crate::primary::SourceSelector> {
        if let Some(receipt) = &batch.acquisition {
            return Ok(receipt.source.clone());
        }
        if let Some(observation) = &batch.observation {
            return Ok(crate::primary::SourceSelector {
                origin: batch.input.format.clone(),
                machine: observation.machine.clone(),
                path_bytes: observation.path_bytes.clone(),
            });
        }
        if let Some(source) = pending
            .iter()
            .rev()
            .find_map(|pending| match &pending.captured {
                Capture::Input(input) if input.rows.input == batch.input.id => {
                    input.basis.as_ref().map(|basis| basis.source.clone())
                }
                _ => None,
            })
        {
            return Ok(source);
        }
        let value:Option<(String, Option<Vec<u8>>)> = self.index.query_row(
            "SELECT machine,path_bytes FROM source_observation WHERE input=?1 ORDER BY rowid DESC LIMIT 1",
            [&batch.input.id], |row| Ok((row.get(0)?,row.get(1)?)),
        ).optional()?;
        match value {
            Some((machine, Some(path_bytes))) => Ok(crate::primary::SourceSelector {
                origin: batch.input.format.clone(),
                machine,
                path_bytes,
            }),
            _ => Err(StoreError::Invalid(
                "continuing capture has no exact source location evidence".into(),
            )),
        }
    }

    fn capture_predecessor(
        &self,
        input: &str,
        chain: i64,
        pending: &[PendingCapture],
    ) -> Result<(u32, Option<String>)> {
        if let Some((captured, material)) = pending.iter().rev().find_map(|pending| match &pending
            .captured
        {
            Capture::Input(captured)
                if captured.rows.input == input
                    && captured
                        .rows
                        .extent
                        .as_ref()
                        .is_some_and(|extent| extent.chain == chain) =>
            {
                Some((captured, pending.material.as_ref()))
            }
            _ => None,
        }) {
            return Ok((
                captured.index.checked_add(1).ok_or_else(|| {
                    StoreError::Invalid("capture part index exceeds source address space".into())
                })?,
                material
                    .map(|m| m.id.clone())
                    .or_else(|| captured.previous.clone()),
            ));
        }
        let count: i64 = self.index.query_row(
            "SELECT count(*) FROM source_material WHERE input=?1 AND chain=?2",
            params![input, chain],
            |row| row.get(0),
        )?;
        let previous=self.index.query_row("SELECT id FROM source_material WHERE input=?1 AND chain=?2 ORDER BY offset DESC LIMIT 1",params![input,chain],|row|row.get(0)).optional()?;
        Ok((
            u32::try_from(count).map_err(|_| {
                StoreError::Invalid("capture part index exceeds source address space".into())
            })?,
            previous,
        ))
    }

    pub(super) fn drain_primary_captures(&mut self) -> Result<()> {
        self.complete_ready_captures()
    }

    /// Prepares the parsed facts of captures whose materials Sinex has
    /// confirmed. It reads only the spool and the index.
    pub(super) fn complete_ready_captures(&mut self) -> Result<()> {
        let sinex = Arc::clone(
            self.sinex
                .as_ref()
                .ok_or_else(|| StoreError::Invalid("primary drain requires a producer".into()))?,
        );
        let pending = sinex.pending_captures()?;
        let Some(first) = pending.first() else {
            return Ok(());
        };
        let grouped = match &first.captured {
            Capture::Input(input) => input.group.as_ref(),
            Capture::Blob { .. } => None,
        };
        let mut members:Vec<&PendingCapture>=match grouped{Some(group)=>pending.iter().filter(|pending|matches!(&pending.captured,Capture::Input(input) if input.group.as_ref().is_some_and(|other|other.identity==group.identity))).collect(),None=>vec![first]};
        if let Some(group) = grouped {
            members.sort_by_key(|member| match &member.captured {
                Capture::Input(input) => input.group.as_ref().map_or(0, |group| group.ordinal),
                _ => 0,
            });
            if members.len() as u64 != group.count {
                return Ok(());
            }
            let ordinals: BTreeSet<u64> = members
                .iter()
                .filter_map(|member| match &member.captured {
                    Capture::Input(input) => input.group.as_ref().map(|g| g.ordinal),
                    _ => None,
                })
                .collect();
            if ordinals.iter().copied().ne(0..group.count) {
                return Err(StoreError::Invalid(
                    "capture group has duplicate or missing ordinals".into(),
                ));
            }
        }
        if members.iter().any(|member| {
            member.anchor.is_none()
                || match &member.captured {
                    Capture::Input(input) => input.raw_range.is_some() && member.material.is_none(),
                    Capture::Blob { .. } => member.material.is_none(),
                }
        }) {
            return Ok(());
        }
        let mut writes = Vec::new();
        let mut ready_revisions = BTreeMap::<String, Revision>::new();
        let mut ready_extents = Vec::<(String, crate::input_rows::Extent, Material)>::new();
        for member in &members {
            let captured = sinex.capture(&member.identity)?;
            match &captured {
                Capture::Blob { hash, length } => {
                    let material = member.material.as_ref().expect("ready material");
                    if material.hash != *hash || material.length != *length {
                        return Err(StoreError::Invalid(
                            "ready blob differs from its durable capture".into(),
                        ));
                    }
                    writes.push(ProductWrite {
                        product: Product::Blob(material.clone()),
                        location: member.anchor.clone(),
                        parents: Vec::new(),
                        machine: None,
                        output_index: writes.len() as u64,
                    });
                }
                Capture::Input(input) => {
                    if let Some(range) = input.raw_range {
                        let material = member.material.as_ref().ok_or_else(|| {
                            StoreError::Invalid(
                                "ready provider capture has no actual material".into(),
                            )
                        })?;
                        if material.hash != input.hash
                            || material.length != input.length
                            || range.length()? != input.length
                        {
                            return Err(StoreError::Invalid(
                                "ready provider material differs from its acquired body".into(),
                            ));
                        }
                    }
                    let Some(rows) = self.ready_capture_rows(
                        input,
                        member.material.as_ref(),
                        &ready_revisions,
                        &ready_extents,
                    )?
                    else {
                        return Ok(());
                    };
                    let chain = if let Some(chain) = rows.observation.as_ref().and_then(|o| o.chain)
                    {
                        chain
                    } else if let Some(extent) = &rows.extent {
                        extent.chain
                    } else if let Some(revision) = &rows.revision {
                        revision.chain
                    } else if let Some(revision) =
                        rows.observation.as_ref().and_then(|o| o.revision)
                    {
                        self.index
                            .query_row(
                                "SELECT chain FROM source_revision WHERE input=?1 AND number=?2",
                                params![rows.input, revision],
                                |row| row.get(0),
                            )
                            .optional()?
                            .or_else(|| {
                                ready_revisions
                                    .get(&rows.input)
                                    .filter(|r| r.number == revision)
                                    .map(|r| r.chain)
                            })
                            .ok_or_else(|| {
                                StoreError::Invalid(
                                    "capture observation references an unavailable revision".into(),
                                )
                            })?
                    } else {
                        latest(&self.index, &rows.input)?.map_or(1, |revision| revision.chain)
                    };
                    let key = InputKey {
                        id: rows.input.clone(),
                        format: rows.format.clone(),
                        arrival_rank: input.arrival_rank,
                    };
                    let machine = rows.observation.as_ref().map(|o| o.machine.clone());
                    if let Some(revision) = &rows.revision {
                        ready_revisions.insert(rows.input.clone(), revision.clone());
                    }
                    if let (Some(extent), Some(material)) = (&rows.extent, &member.material) {
                        ready_extents.push((rows.input.clone(), extent.clone(), material.clone()));
                    }
                    let material = if rows.extent.is_some() {
                        member.material.clone()
                    } else {
                        None
                    };
                    if rows.extent.is_none()
                        && let Some(material) = &member.material
                    {
                        writes.push(ProductWrite {
                            product: Product::Blob(material.clone()),
                            location: member.anchor.clone(),
                            parents: Vec::new(),
                            machine: rows.observation.as_ref().map(|o| o.machine.clone()),
                            output_index: writes.len() as u64,
                        });
                    }
                    writes.push(ProductWrite {
                        machine: rows.observation.as_ref().map(|o| o.machine.clone()),
                        product: Product::Input {
                            rows: Box::new(rows),
                            material,
                        },
                        location: member.anchor.clone(),
                        parents: Vec::new(),
                        output_index: writes.len() as u64,
                    });
                    for captured in &input.parsed {
                        let mut step = captured.clone().into_step();
                        step.prepare();
                        for product in crate::primary::products::parse(
                            &key,
                            chain,
                            captured.export_key.as_ref().map(|key| key.length),
                            &step,
                        )? {
                            let reading = product.reading().expect("parsed product");
                            let start = match &product {
                                Product::Message { facts, .. } => facts.offset,
                                Product::Document { facts, .. } => facts.offset,
                                Product::Fragment { facts, .. } => facts.offset,
                                Product::Event { facts, .. } => {
                                    facts.range.map_or(reading.start, |range| range.0)
                                }
                                _ => reading.start,
                            };
                            let location =
                                match captured_location(input, member.material.as_ref(), start)? {
                                    Some(location) => Some(location),
                                    None => {
                                        match ready_extents.iter().rev().find(|(id, extent, _)| {
                                            id == &key.id
                                                && extent.chain == chain
                                                && start >= extent.offset as u64
                                                && start <= (extent.offset + extent.length) as u64
                                        }) {
                                            Some((_, extent, material)) => {
                                                Some(crate::primary::SourceLocation {
                                                    material: material.id.clone(),
                                                    anchor: start - extent.offset as u64
                                                        + extent.blob_offset as u64,
                                                })
                                            }
                                            None => crate::primary::projection::material_at(
                                                &self.index,
                                                &key.id,
                                                chain,
                                                start,
                                            )?,
                                        }
                                    }
                                };
                            if location.is_none() {
                                return Err(StoreError::Invalid(
                                    "parsed fact has no captured source anchor".into(),
                                ));
                            }
                            writes.push(ProductWrite {
                                product,
                                location,
                                machine: machine.clone(),
                                parents: Vec::new(),
                                output_index: writes.len() as u64,
                            });
                        }
                    }
                }
            }
        }
        let identities = members
            .iter()
            .map(|member| member.identity.clone())
            .collect::<Vec<_>>();
        sinex.complete_captures(&identities, &writes)?;
        Ok(())
    }

    fn ready_capture_rows(
        &mut self,
        input: &CapturedInput,
        material: Option<&Material>,
        ready_revisions: &BTreeMap<String, Revision>,
        ready_extents: &[(String, crate::input_rows::Extent, Material)],
    ) -> Result<Option<InputRows>> {
        let source = input
            .basis
            .as_ref()
            .map(|basis| basis.source.clone())
            .or_else(|| {
                let observation = input.rows.observation.as_ref()?;
                Some(crate::primary::SourceSelector {
                    origin: input.rows.format.clone(),
                    machine: observation.machine.clone(),
                    path_bytes: observation.path_bytes.clone()?,
                })
            });
        let Some(source) = source else {
            return Ok(None);
        };
        // Current source coordinates require a complete scoped prefix; a known
        // exact row alone cannot exclude an unreadable newer input. Frozen
        // pre-I/O uncertainty continues to govern the pending capture's alias claims.
        self.primary_cache_ready()?;
        let (status, confirmed) = crate::source::Source::confirmed_source(&self.index, &source)?;
        if status != crate::primary::SourceStatus::Ready {
            return Ok(None);
        }
        if let Some(confirmed) = confirmed
            && confirmed.input != input.rows.input
        {
            return Ok(None);
        }
        if crate::source::SourceBoundary::read(&self.index, &self.index, &self.location)?.is_none()
        {
            return Ok(None);
        }
        let replay = crate::primary::projection::replay_state(&self.index)?;
        if replay
            .head
            .is_some_and(|head| replay.applied_position.is_none_or(|applied| applied < head))
        {
            return Ok(None);
        }
        let mut rows = input.rows.clone();
        let Some(original) = &input.rows.extent else {
            return Ok(Some(rows));
        };
        let material = material
            .ok_or_else(|| StoreError::Invalid("ready source extent has no material".into()))?;
        if material.hash != original.blob
            || original.length < 0
            || original.blob_offset < 0
            || (original.blob_offset as u64)
                .checked_add(original.length as u64)
                .is_none_or(|end| end > material.length)
        {
            return Err(StoreError::Invalid(
                "source extent differs from ready capture material".into(),
            ));
        }
        let current = match ready_revisions.get(&rows.input) {
            Some(revision) => Some(revision.clone()),
            None => latest(&self.index, &rows.input)?,
        };
        let Some(current) = current else {
            if original.offset != 0 {
                return Ok(None);
            }
            return Ok(Some(rows));
        };
        let start = u64::try_from(original.offset)
            .map_err(|_| StoreError::Invalid("negative capture offset".into()))?;
        let end = start
            .checked_add(original.length as u64)
            .ok_or_else(|| StoreError::Invalid("capture endpoint overflow".into()))?;
        let stored = u64::try_from(current.length)
            .map_err(|_| StoreError::Invalid("negative confirmed source length".into()))?;
        if start > stored {
            return Ok(None);
        }
        let overlap = stored.min(end) - start;
        let mut captured = material_reader(&self.location, material)?;
        captured.seek(SeekFrom::Start(original.blob_offset as u64))?;
        let mut cursor = start;
        let mut equal = true;
        while cursor < start + overlap {
            let pending = ready_extents.iter().rev().find(|(id, extent, _)| {
                id == &rows.input
                    && extent.chain == current.chain
                    && cursor >= extent.offset as u64
                    && cursor < (extent.offset + extent.length) as u64
            });
            let (mut prefix, end): (Box<dyn Read>, u64) =
                if let Some((_, extent, material)) = pending {
                    let end = (start + overlap).min((extent.offset + extent.length) as u64);
                    let mut reader = material_reader(&self.location, material)?;
                    reader.seek(SeekFrom::Start(
                        cursor - extent.offset as u64 + extent.blob_offset as u64,
                    ))?;
                    (Box::new(reader), end)
                } else {
                    let end = ready_extents
                        .iter()
                        .filter(|(id, extent, _)| {
                            id == &rows.input
                                && extent.chain == current.chain
                                && extent.offset as u64 > cursor
                        })
                        .map(|(_, extent, _)| extent.offset as u64)
                        .min()
                        .unwrap_or(start + overlap)
                        .min(start + overlap);
                    let chain = crate::blobs::StoredChain::load_range(
                        &self.index,
                        "source_",
                        &rows.input,
                        current.chain,
                        cursor,
                        end,
                    )?;
                    (
                        Box::new(chain.reader(Arc::new(self.location.blob_reader()?))),
                        end,
                    )
                };
            let mut left = [0; 65536];
            let mut right = [0; 65536];
            while cursor < end {
                let n = (end - cursor).min(left.len() as u64) as usize;
                prefix.read_exact(&mut left[..n])?;
                captured.read_exact(&mut right[..n])?;
                if left[..n] != right[..n] {
                    equal = false;
                    break;
                }
                cursor += n as u64;
            }
            if !equal {
                break;
            }
        }
        if !equal {
            if start != 0 {
                return Err(StoreError::Conflict(
                    "an appended capture disagrees with its confirmed source prefix".into(),
                ));
            }
            let number = current
                .number
                .checked_add(1)
                .ok_or_else(|| StoreError::Invalid("revision number overflow".into()))?;
            if let Some(revision) = &mut rows.revision {
                revision.number = number;
                revision.chain = number;
                revision.produced = "rewritten".into();
            }
            if let Some(extent) = &mut rows.extent {
                extent.chain = number;
            }
            if let Some(observation) = &mut rows.observation {
                observation.revision = Some(number);
            }
            return Ok(Some(rows));
        }
        if end <= stored {
            rows.revision = None;
            rows.extent = None;
            if let Some(observation) = &mut rows.observation {
                let exact=self.index.query_row("SELECT number FROM source_revision WHERE input=?1 AND chain=?2 AND length=?3 ORDER BY number DESC LIMIT 1",params![rows.input,current.chain,end as i64],|row|row.get(0)).optional()?;
                observation.revision = exact.or_else(|| (end == stored).then_some(current.number));
                observation.chain = observation.revision.is_none().then_some(current.chain);
                observation.length = Some(end as i64);
                observation.class = "back".into();
            }
        } else {
            let continues = rows.observation.is_none()
                && rows.revision.as_ref().is_some_and(|revision| {
                    revision.number == current.number && revision.chain == current.chain
                });
            let number = if continues {
                current.number
            } else {
                current
                    .number
                    .checked_add(1)
                    .ok_or_else(|| StoreError::Invalid("revision number overflow".into()))?
            };
            rows.revision = Some(Revision {
                number,
                produced: if continues {
                    current.produced
                } else {
                    "appended".into()
                },
                chain: current.chain,
                length: end as i64,
            });
            let extent = rows.extent.as_mut().expect("source extent");
            extent.chain = current.chain;
            extent.offset = stored as i64;
            extent.length = (end - stored) as i64;
            extent.blob_offset += overlap as i64;
            if let Some(observation) = &mut rows.observation {
                observation.revision = Some(number);
                observation.length = Some(end as i64);
                observation.class = "appended".into();
            }
        }
        Ok(Some(rows))
    }
}

#[cfg(test)]
mod tests;
