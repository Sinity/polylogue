//! The existing manifest hash input, with object fields in sorted order.
use super::{ExcisionPublication, Replacement};
use crate::Result;
use serde::ser::SerializeSeq;
use serde::{Serialize, Serializer};
use std::io::{BufWriter, Write};

pub(super) fn compute(publication: &ExcisionPublication) -> Result<String> {
    let mut hash = blake3::Hasher::new();
    let mut writer = BufWriter::new(&mut hash);
    serde_json::to_writer(
        &mut writer,
        &Manifest {
            replacements: Replacements(&publication.replacements),
            retired_batches: &publication.retired_batches,
            version: publication.version,
        },
    )?;
    writer.flush()?;
    drop(writer);
    Ok(format!("blake3:{}", hash.finalize()))
}

// Field order is part of the existing digest codec, independent of feature flags.
#[derive(Serialize)]
struct Manifest<'a> {
    replacements: Replacements<'a>,
    retired_batches: &'a [String],
    version: u32,
}
struct Replacements<'a>(&'a [Replacement]);
impl Serialize for Replacements<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(Some(self.0.len()))?;
        for replacement in self.0 {
            seq.serialize_element(&SortedReplacement {
                batch: &replacement.batch,
                blobs: &replacement.blobs,
                bytes: replacement.bytes,
                rows: replacement.rows,
                unit: SortedUnit {
                    commit_seq: replacement.unit.commit_seq,
                    key: &replacement.unit.key,
                    kind: &replacement.unit.kind,
                },
            })?;
        }
        seq.end()
    }
}
#[derive(Serialize)]
struct SortedReplacement<'a> {
    batch: &'a Option<String>,
    blobs: &'a [String],
    bytes: u64,
    rows: u64,
    unit: SortedUnit<'a>,
}
#[derive(Serialize)]
struct SortedUnit<'a> {
    commit_seq: i64,
    key: &'a str,
    kind: &'a str,
}

#[cfg(test)]
mod tests;
