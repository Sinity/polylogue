use super::{Erasure, Plan, Range};
use serde_json::value::RawValue;
use std::fs::File;
use std::io::{self, BufReader, Seek, SeekFrom};
use std::os::unix::fs::FileExt;

#[derive(Debug)]
pub(in crate::excise) enum Error {
    Io(io::Error),
    Records(String),
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

/// The file must be a private, verified plaintext copy. This moves its file
/// position; all trimming and gap checks use positioned reads.
pub(in crate::excise) fn plan_file(file: &File, requested: &[Range]) -> Result<Erasure, Error> {
    let length = usize::try_from(file.metadata()?.len())
        .map_err(|_| Error::Records("source blob exceeds addressable offsets".into()))?;
    let mut ranges = Vec::new();
    for &(start, end) in requested {
        if start > end || end > length {
            return Err(Error::Records("source range is outside its blob".into()));
        }
        let start = edge(file, start, end, false)?
            .ok_or_else(|| Error::Records("source range contains no JSON record".into()))?;
        let end = edge(file, start, end, true)?.expect("nonempty range") + 1;
        ranges.push((start, end));
    }
    ranges.sort_unstable();
    let mut requested: Vec<Range> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = requested.last_mut().filter(|last| last.1 >= start) {
            last.1 = last.1.max(end);
        } else {
            requested.push((start, end));
        }
    }
    let mut seen = vec![(false, false); requested.len()];
    let mut output = Vec::new();
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
            requested,
            seen,
            output,
        };
        let range = (start, end);
        documents += 1;
        if let Some(index) = plan.covered(range) {
            plan.mark(index, range);
            plan.push(range);
            erased_documents += 1;
        } else if plan.touched(range) {
            plan.descend(&raw).map_err(Error::Records)?;
        }
        requested = plan.requested;
        seen = plan.seen;
        output = plan.output;
    }
    if seen.iter().any(|&(start, end)| !start || !end) {
        return Err(Error::Records(
            "source citation does not follow complete JSON value boundaries".into(),
        ));
    }
    // Local descent already coalesces within roots. Match the same behavior
    // between roots without retaining the whitespace between them.
    let mut ranges: Vec<Range> = Vec::new();
    for range in output {
        if let Some(last) = ranges.last_mut()
            && last.1 <= range.0
            && edge(file, last.1, range.0, false)?.is_none()
        {
            last.1 = range.1;
        } else {
            ranges.push(range);
        }
    }
    Ok(Erasure {
        ranges,
        documents,
        erased_documents,
    })
}

/// First/last non-whitespace byte in [start,end), using bounded reads.
fn edge(file: &File, mut start: usize, mut end: usize, reverse: bool) -> io::Result<Option<usize>> {
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

#[cfg(test)]
mod tests;
