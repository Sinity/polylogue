use super::{Plan, Ranges};
use serde_json::value::RawValue;
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom};
use std::os::unix::fs::FileExt;

#[derive(Debug)]
pub(in crate::excise) enum Error {
    Io(io::Error),
    Records(String),
    Store(crate::StoreError),
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The file must be a private, verified plaintext copy. This moves its file
/// position; all trimming and gap checks use positioned reads.
pub(in crate::excise) fn plan_file(
    file: &File,
    ranges: &mut impl Ranges,
) -> Result<(usize, usize), Error> {
    let mut reader = file.try_clone()?;
    reader.seek(SeekFrom::Start(0))?;
    let mut records =
        serde_json::Deserializer::from_reader(BufReader::new(reader)).into_iter::<Box<RawValue>>();
    let mut documents = 0;
    let mut erased_documents = 0;
    while let Some(raw) = records.next() {
        let raw = raw.map_err(|error| {
            if let Some(kind) = error.io_error_kind() {
                Error::Io(io::Error::new(kind, error))
            } else {
                Error::Records(format!(
                    "source piece is not a complete JSON record stream: {error}"
                ))
            }
        })?;
        let end = records.byte_offset();
        let start = end
            .checked_sub(raw.get().len())
            .ok_or_else(|| Error::Records("invalid JSON stream offset".into()))?;
        let mut plan = Plan {
            bytes: raw.get().as_bytes(),
            base: start,
            ranges: &mut *ranges,
        };
        let range = (start, end);
        documents += 1;
        if plan.ranges.covered(range).map_err(Error::Records)? {
            plan.ranges.mark(range);
            plan.ranges.push(range).map_err(Error::Records)?;
            erased_documents += 1;
        } else if plan.ranges.touched(range).map_err(Error::Records)? {
            plan.descend(&raw).map_err(Error::Records)?;
        }
    }
    Ok((documents, erased_documents))
}

/// First/last non-whitespace byte in [start,end), using bounded reads.
pub(in crate::excise) fn edge(
    file: &File,
    mut start: usize,
    mut end: usize,
    reverse: bool,
) -> io::Result<Option<usize>> {
    let mut bytes = [0; 64 * 1024];
    while start < end {
        let count = (end - start).min(bytes.len());
        let at = if reverse { end - count } else { start };
        file.read_exact_at(&mut bytes[..count], at as u64)?;
        let found = if reverse {
            bytes[..count]
                .iter()
                .rposition(|b| !b.is_ascii_whitespace())
        } else {
            bytes[..count].iter().position(|b| !b.is_ascii_whitespace())
        };
        if let Some(offset) = found {
            return Ok(Some(at + offset));
        }
        if reverse {
            end -= count;
        } else {
            start += count;
        }
    }
    Ok(None)
}
