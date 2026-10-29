//! Raw acquisition is durable in the SDK spool before its temporary bytes are
//! released. Only a confirmed Input frame changes the source cache.
use super::*;
use crate::input_rows::{InputRows, Revision};
use crate::primary::{
    Capture, CaptureGroup, CapturedInput, CapturedReading, Material, PendingCapture, PreparedInput,
    Product, ProductWrite,
};
use crate::staging::{Prepared, PreparedBatch, StoredInput};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Seek, SeekFrom};

fn latest(index: &Connection, input: &str) -> Result<Option<Revision>> {
    index.query_row("SELECT number,produced,chain,length FROM source_revision WHERE input=?1 ORDER BY number DESC LIMIT 1",[input],|row|Ok(Revision{number:row.get(0)?,produced:row.get(1)?,chain:row.get(2)?,length:row.get(3)?})).optional().map_err(StoreError::from)
}

fn failed(count: usize, error: StoreError) -> Vec<Result<StoredInput>> {
    let reason = error.to_string();
    (0..count)
        .map(|_| Err(StoreError::Invalid(reason.clone())))
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

fn validate(item: &PreparedBatch, archive: &Archive) -> Result<()> {
    let batch = &item.batch;
    if batch.input.id.is_empty() || i64::try_from(batch.length).is_err() {
        return Err(StoreError::Invalid("invalid primary capture input".into()));
    }
    for blob in batch.new_blob.iter().chain(&batch.blobs) {
        if !item.files.iter().any(|file| {
            file.owner == archive.root
                && file.hash() == blob.hash
                && file.len() == blob.size
                && matches!(file.file, Prepared::Part { .. })
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
    if let Some(blob) = &batch.new_blob {
        if blob.size > batch.length {
            return Err(StoreError::Invalid(
                "captured bytes exceed their source endpoint".into(),
            ));
        }
    }
    Ok(())
}

// An acquisition intent is independent of observer time and generated observation IDs.
// Ordered source content, source facts and reading versions distinguish actual new work.
fn capture_group_identity(group: &[PreparedBatch]) -> Result<String> {
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
        let observation = b.observation.as_ref().map(|o| {
            (
                &o.machine,
                &o.host,
                &o.path,
                o.facts
                    .as_ref()
                    .map(|f| (f.device, f.inode, f.size, f.mtime_ns, f.ctime_ns)),
                &o.import,
                &o.container,
            )
        });
        let encoded = serde_json::to_vec(&(
            &b.input,
            change,
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
                        step.at_end,
                        step.reset,
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
        let pending = sinex.pending_captures()?;
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
        let group_id = capture_group_identity(group)?;
        let mut accepted = Vec::new();
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
            if let Some(blob) = &batch.new_blob {
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
            let (hash, length) = batch
                .new_blob
                .as_ref()
                .map(|blob| (blob.hash, blob.size))
                .unwrap_or_else(|| (BlobHash::of(&metadata_bytes), metadata_bytes.len() as u64));
            let captured = CapturedInput {
                input: name,
                arrival_rank: batch.input.arrival_rank,
                parsed: batch
                    .parse
                    .iter()
                    .map(CapturedReading::from_step)
                    .collect::<Result<_>>()?,
                rows,
                index,
                previous: previous_material,
                hash,
                length,
                group: Some(CaptureGroup {
                    identity: group_id.clone(),
                    ordinal: ordinal as u64,
                    count: group.len() as u64,
                }),
            };
            for blob in &batch.blobs {
                let file = item
                    .files
                    .iter()
                    .find(|file| file.hash() == blob.hash)
                    .expect("validated blob");
                let Prepared::Part { file, .. } = &file.file else {
                    unreachable!()
                };
                sinex.prepare_blob(blob.hash, blob.size, &mut file.reopen()?)?;
            }
            let prepared = if let Some(blob) = &batch.new_blob {
                let file = item
                    .files
                    .iter()
                    .find(|file| file.hash() == blob.hash)
                    .expect("validated source");
                let Prepared::Part { file, .. } = &file.file else {
                    unreachable!()
                };
                sinex.prepare_input(&captured, &mut file.reopen()?)?
            } else {
                sinex.prepare_input(&captured, &mut metadata_bytes.as_slice())?
            };
            accepted.push((
                batch.input.id.clone(),
                batch.length,
                captured.rows.clone(),
                batch.new_blob.is_none(),
                prepared,
            ));
        }
        self.drain_primary_captures()?;
        self.poll_primary()?;
        let mut outcomes = Vec::new();
        for (input, length, rows, already_stored, prepared) in accepted {
            let pending = match prepared {
                PreparedInput::Confirmed { stored, .. } => {
                    outcomes.push(Ok(stored));
                    continue;
                }
                PreparedInput::Pending { identity } => identity,
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
        let sinex = Arc::clone(
            self.sinex
                .as_ref()
                .ok_or_else(|| StoreError::Invalid("primary drain requires a producer".into()))?,
        );
        self.poll_primary()?;
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
                    Capture::Input(input) => {
                        input.rows.extent.is_some() && member.material.is_none()
                    }
                    Capture::Blob { .. } => member.material.is_none(),
                }
        }) {
            return Ok(());
        }
        let mut writes = Vec::new();
        for member in &members {
            match &member.captured {
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
                    let Some(rows) = self.ready_capture_rows(input, member.material.as_ref())?
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
                        self.index.query_row(
                            "SELECT chain FROM source_revision WHERE input=?1 AND number=?2",
                            params![rows.input, revision],
                            |row| row.get(0),
                        )?
                    } else {
                        latest(&self.index, &rows.input)?.map_or(1, |revision| revision.chain)
                    };
                    let key = InputKey {
                        id: rows.input.clone(),
                        format: rows.format.clone(),
                        arrival_rank: input.arrival_rank,
                    };
                    let machine = rows.observation.as_ref().map(|o| o.machine.clone());
                    let material = if rows.extent.is_some() {
                        member.material.clone()
                    } else {
                        None
                    };
                    if rows.extent.is_none() {
                        if let Some(material) = &member.material {
                            writes.push(ProductWrite {
                                product: Product::Blob(material.clone()),
                                location: member.anchor.clone(),
                                parents: Vec::new(),
                                machine: rows.observation.as_ref().map(|o| o.machine.clone()),
                                output_index: writes.len() as u64,
                            });
                        }
                    }
                    writes.push(ProductWrite {
                        machine: rows.observation.as_ref().map(|o| o.machine.clone()),
                        product: Product::Input { rows, material },
                        location: member.anchor.clone(),
                        parents: Vec::new(),
                        output_index: writes.len() as u64,
                    });
                    for captured in &input.parsed {
                        let mut step = captured.clone().into_step();
                        step.prepare();
                        for product in crate::primary::products::parse(&key, chain, &step)? {
                            let reading = product.reading().expect("parsed product");
                            let start = match &product {
                                Product::Message { facts, .. } => facts.offset,
                                Product::Document { facts, .. } => facts.offset,
                                Product::Event { facts, .. } => {
                                    facts.range.map_or(reading.start, |range| range.0)
                                }
                                _ => reading.start,
                            };
                            let location = match (&input.rows.extent, &member.material) {
                                (Some(original), Some(material))
                                    if start >= original.offset as u64
                                        && start
                                            < (original.offset as u64 + original.length as u64)
                                        || original.length == 0
                                            && start == original.offset as u64 =>
                                {
                                    Some(crate::primary::SourceLocation {
                                        material: material.id.clone(),
                                        anchor: start - original.offset as u64
                                            + original.blob_offset as u64,
                                    })
                                }
                                _ => crate::primary::projection::material_at(
                                    &self.index,
                                    &key.id,
                                    chain,
                                    start,
                                )?,
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
        self.poll_primary()?;
        Ok(())
    }

    fn ready_capture_rows(
        &self,
        input: &CapturedInput,
        material: Option<&Material>,
    ) -> Result<Option<InputRows>> {
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
        let current = latest(&self.index, &rows.input)?;
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
        let chain = crate::blobs::StoredChain::load_range(
            &self.index,
            "source_",
            &rows.input,
            current.chain,
            start,
            start + overlap,
        )?;
        let mut prefix = chain.reader(Arc::new(self.location.blob_reader()?));
        let mut left = [0; 65536];
        let mut right = [0; 65536];
        let mut remaining = overlap;
        let mut equal = true;
        while remaining > 0 {
            let n = remaining.min(left.len() as u64) as usize;
            prefix.read_exact(&mut left[..n])?;
            captured.read_exact(&mut right[..n])?;
            if left[..n] != right[..n] {
                equal = false;
                break;
            }
            remaining -= n as u64;
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
