//! Stores the inputs of an archive in the older format as that archive held
//! them: every revision, extent and observation with its original identity,
//! class, location, file facts, version and time. The bytes go into the
//! segment log first, then one transaction writes their extents and the
//! rows that cite them, as for any captured input.

use super::*;
use crate::input_rows::InputRows;
use crate::older_archive::{Input, Machine};
use crate::staging::PreparedBytes;
use std::collections::BTreeSet;

/// One group of an older archive's inputs, with the prepared bytes of the
/// blobs they cite that this archive does not hold yet.
#[derive(Default)]
pub struct OlderGroup {
    pub files: Vec<PreparedBytes>,
    pub machines: Vec<Machine>,
    pub inputs: Vec<Input>,
}

/// What became of a group's inputs and machines, in their order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GroupOutcome {
    pub inputs: Vec<Restored>,
    pub machines: Vec<Restored>,
}

/// What became of one row of a group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Restored {
    /// Its rows were written.
    Stored,
    /// This archive already holds every row of it, from an earlier import.
    Present,
    /// Not written, for the reason given.
    Refused(String),
}

pub(super) fn restore_via(queue: &WriterQueue, group: OlderGroup) -> Result<GroupOutcome> {
    let (reply, wait) = sync_channel(1);
    queue
        .send(Msg::RestoreOlder(Box::new(group), reply))
        .map_err(|_| StoreError::WriterGone)?;
    wait.recv().map_err(|_| StoreError::WriterGone)?
}

/// An observation's recorded facts, compared field by field.
#[derive(PartialEq)]
struct ObservationFacts {
    input: String,
    machine: String,
    path: String,
    class: String,
    numbers: [Option<i64>; 7],
    details_json: Option<String>,
    polylogue_version: String,
    seen_at: i64,
    path_bytes: Option<Vec<u8>>,
}

fn facts(input: &str, o: &crate::input_rows::ObservationRow) -> ObservationFacts {
    ObservationFacts {
        input: input.to_owned(),
        machine: o.machine.clone(),
        path: o.path.clone(),
        class: o.class.clone(),
        numbers: [
            o.revision, o.length, o.device, o.inode, o.size, o.mtime_ns, o.ctime_ns,
        ],
        details_json: o.details_json.clone(),
        polylogue_version: o.polylogue_version.clone(),
        seen_at: o.seen_at,
        path_bytes: o.path_bytes.clone(),
    }
}

/// Whether this archive holds every row of `input` (`Some(None)`), holds an
/// input of the same ID with other rows (`Some(Some(reason))`), or does not
/// hold it (`None`). Rows this archive added after an earlier import, such as
/// later observations of a watched file, do not count as a difference.
fn held(conn: &Connection, input: &Input) -> Result<Option<Option<String>>> {
    let format: Option<String> = conn
        .query_row("SELECT format FROM input WHERE id = ?1", [&input.id], |r| {
            r.get(0)
        })
        .optional()?;
    let Some(format) = format else {
        return Ok(None);
    };
    if format != input.format {
        return Ok(Some(Some(format!(
            "this archive holds it with format {format}, not {}",
            input.format
        ))));
    }
    let mut revision = conn.prepare_cached(
        "SELECT produced, chain, length FROM revision WHERE input = ?1 AND number = ?2",
    )?;
    for r in &input.revisions {
        let found: Option<(String, i64, i64)> = revision
            .query_row(params![input.id, r.number], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .optional()?;
        if found != Some((r.produced.clone(), r.chain, r.length)) {
            return Ok(Some(Some(format!(
                "this archive holds its revision {} differently",
                r.number
            ))));
        }
    }
    let mut extent = conn.prepare_cached(
        "SELECT length, blob, blob_offset FROM extent WHERE input = ?1 AND chain = ?2 AND offset = ?3",
    )?;
    for e in &input.extents {
        let found: Option<(i64, Vec<u8>, i64)> = extent
            .query_row(params![input.id, e.chain, e.offset], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .optional()?;
        if found != Some((e.length, e.blob.0.to_vec(), e.blob_offset)) {
            return Ok(Some(Some(format!(
                "this archive holds its bytes at chain {} offset {} differently",
                e.chain, e.offset
            ))));
        }
    }
    let mut observation = conn.prepare_cached(
        "SELECT input, machine, path, class, revision, length, device, inode, size, mtime_ns,
                ctime_ns, details_json, polylogue_version, seen_at, path_bytes
         FROM observation WHERE id = ?1",
    )?;
    for o in &input.observations {
        let found: Option<ObservationFacts> = observation
            .query_row([&o.id], |row| {
                Ok(ObservationFacts {
                    input: row.get(0)?,
                    machine: row.get(1)?,
                    path: row.get(2)?,
                    class: row.get(3)?,
                    numbers: [
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                        row.get(8)?,
                        row.get(9)?,
                        row.get(10)?,
                    ],
                    details_json: row.get(11)?,
                    polylogue_version: row.get(12)?,
                    seen_at: row.get(13)?,
                    path_bytes: row.get(14)?,
                })
            })
            .optional()?;
        if found.as_ref() != Some(&facts(&input.id, o)) {
            return Ok(Some(Some(format!(
                "this archive holds its observation {} differently",
                o.id
            ))));
        }
    }
    Ok(Some(None))
}

/// For an input this archive does not hold: an observation ID it already
/// uses for another input.
fn taken_observation(conn: &Connection, input: &Input) -> Result<Option<String>> {
    let mut statement = conn.prepare_cached("SELECT input FROM observation WHERE id = ?1")?;
    for o in &input.observations {
        let owner: Option<String> = statement.query_row([&o.id], |r| r.get(0)).optional()?;
        if let Some(owner) = owner {
            return Ok(Some(format!(
                "its observation {} is already the observation of input {owner} here",
                o.id
            )));
        }
    }
    Ok(None)
}

impl Inner {
    /// Writes a group of an older archive's inputs. Inputs already held
    /// unchanged are left alone; an input held differently is refused; the
    /// rest are written in one transaction with the extents of their bytes.
    pub(super) fn restore_older(&mut self, mut group: OlderGroup) -> Result<GroupOutcome> {
        if self.sinex.is_some() {
            return Err(StoreError::Invalid(
                "an archive in the older format is imported by a standalone archive".into(),
            ));
        }
        let mut outcomes = Vec::with_capacity(group.inputs.len());
        let mut write = Vec::new();
        let mut cited = BTreeSet::new();
        for (index, input) in group.inputs.iter().enumerate() {
            let outcome = if let Some(reason) = &input.refusal {
                Restored::Refused(reason.clone())
            } else {
                match held(&self.archive, input)? {
                    Some(None) => Restored::Present,
                    Some(Some(reason)) => Restored::Refused(reason),
                    None => match taken_observation(&self.archive, input)? {
                        Some(reason) => Restored::Refused(reason),
                        None => {
                            write.push(index);
                            cited.extend(input.blobs());
                            Restored::Stored
                        }
                    },
                }
            };
            outcomes.push(outcome);
        }
        let mut machines = Vec::with_capacity(group.machines.len());
        for machine in &group.machines {
            let known: bool = self
                .archive
                .prepare_cached("SELECT 1 FROM machine WHERE id = ?1 AND host = ?2")?
                .exists(params![machine.id, machine.host])?;
            machines.push(if known {
                Restored::Present
            } else {
                Restored::Stored
            });
        }
        let files: Vec<PreparedBytes> = std::mem::take(&mut group.files);
        for file in &files {
            if file.owner != self.location.root {
                return Err(StoreError::Invalid(
                    "prepared bytes belong to another archive".into(),
                ));
            }
        }
        if write.is_empty() && files.is_empty() && !machines.contains(&Restored::Stored) {
            return Ok(GroupOutcome {
                inputs: outcomes,
                machines,
            });
        }
        let synced = if files.is_empty() {
            None
        } else {
            if self.segment.is_none() {
                self.segment = Some(crate::blobs::open_writer(
                    &self.archive,
                    &self.location.root,
                )?);
            }
            let writer = self.segment.as_mut().expect("opened segment writer");
            let prepared = files.into_iter().map(PreparedBytes::into_prepared);
            let appended = writer.append_group(prepared, |hash| {
                crate::blobs::load_extent(&self.archive, "", hash)
                    .map_err(|error| hashed_blobs::Error::Io(std::io::Error::other(error)))
            });
            let pending = match appended {
                Ok(pending) => pending,
                Err(error) => {
                    self.segment.take();
                    return Err(error.into());
                }
            };
            match pending.wait() {
                Ok(synced) => Some(synced),
                Err(error) => {
                    self.segment.take();
                    return Err(error.into());
                }
            }
        };
        let committed = (|| {
            let tx = self.archive.transaction()?;
            tx.execute_batch("PRAGMA defer_foreign_keys=ON")?;
            let seq = bump_counter(&tx)?;
            if let Some(synced) = &synced {
                for extent in &synced.extents {
                    crate::blobs::insert_extent(&tx, extent)?;
                }
                tx.execute(
                    "INSERT INTO segment(name,committed_end) VALUES (?1,?2) ON CONFLICT(name) DO UPDATE SET committed_end=excluded.committed_end",
                    params!["segments/current.hb", synced.end as i64],
                )?;
            }
            let mut missing =
                tx.prepare_cached("SELECT NOT EXISTS(SELECT 1 FROM blob WHERE hash = ?1)")?;
            for hash in &cited {
                if missing.query_row([&hash.0[..]], |r| r.get::<_, bool>(0))? {
                    return Err(StoreError::Invalid(format!(
                        "blob {hash} an older input cites was not prepared"
                    )));
                }
            }
            drop(missing);
            for machine in &group.machines {
                tx.prepare_cached(
                    "INSERT OR IGNORE INTO machine(id, host, first_seen) VALUES (?1, ?2, ?3)",
                )?
                .execute(params![machine.id, machine.host, machine.first_seen])?;
            }
            for &index in &write {
                let input = &group.inputs[index];
                let rows =
                    |revision: Option<crate::input_rows::Revision>,
                     extents: Vec<crate::input_rows::Extent>,
                     observation: Option<crate::input_rows::ObservationRow>| {
                        InputRows {
                            acquisition: None,
                            input: input.id.clone(),
                            format: input.format.clone(),
                            revision,
                            extents,
                            observation,
                        }
                    };
                for revision in &input.revisions {
                    rows(Some(revision.clone()), Vec::new(), None).insert(&tx, seq)?;
                }
                for extent in &input.extents {
                    rows(None, vec![extent.clone()], None).insert(&tx, seq)?;
                }
                for observation in &input.observations {
                    rows(None, Vec::new(), Some(observation.clone())).insert(&tx, seq)?;
                }
            }
            crate::write_lock::commit(tx)?;
            Ok(())
        })();
        if let Err(error) = committed {
            if synced.is_some() {
                self.segment.take();
            }
            return Err(error);
        }
        self.commits.committed(crate::completion::Committed {
            content: true,
            authored: true,
            query_records: false,
        });
        if let Some(synced) = synced {
            self.report.segment_syncs += synced.metrics.syncs;
            self.report.segment_bytes_written += synced.metrics.appended_bytes;
            self.report.blobs_appended += synced.metrics.appended_materials;
            self.report.blobs_deduplicated += synced.metrics.deduplicated_materials;
            if let Err(error) = self
                .segment
                .as_mut()
                .expect("appended writer")
                .acknowledge(&synced)
            {
                self.segment.take();
                return Err(error.into());
            }
        }
        self.refresh_chain_progress()?;
        Ok(GroupOutcome {
            inputs: outcomes,
            machines,
        })
    }
}
