//! Frame the outer-cell delimiter used by this Takeout format, preserving bytes.
use super::CELL;
use crate::export::{Input, ScanError, ScanStats};
use std::ops::Range;

pub(super) fn scan(
    reader: &mut dyn Input,
    mut item: impl FnMut(Range<u64>, &[u8]) -> Result<(), String>,
) -> Result<ScanStats, ScanError> {
    let delimiter = CELL.as_bytes();
    let mut buffer = Vec::new();
    let mut base = 0u64;
    let mut search = 0;
    let mut active = false;
    let mut stats = ScanStats::default();
    loop {
        let chunk = reader
            .fill_buf()
            .map_err(|e| ScanError::Input(serde_json::Error::io(e)))?;
        if chunk.is_empty() {
            break;
        }
        let n = chunk.len().min(8192);
        buffer.extend_from_slice(&chunk[..n]);
        reader.consume(n);
        let mut previous = 0;
        for at in buffer[search..]
            .windows(delimiter.len())
            .enumerate()
            .filter_map(|(i, bytes)| (bytes == delimiter).then_some(search + i))
        {
            if active {
                let raw = &buffer[previous..at];
                stats.elements += 1;
                stats.peak_element_bytes = stats.peak_element_bytes.max(raw.len());
                item(base + previous as u64..base + at as u64, raw).map_err(ScanError::Callback)?;
            }
            active = true;
            previous = at;
        }
        let discard = if active {
            previous
        } else {
            buffer.len().saturating_sub(delimiter.len() - 1)
        };
        buffer.drain(..discard);
        base += discard as u64;
        search = buffer.len().saturating_sub(delimiter.len() - 1);
    }
    if active {
        stats.elements += 1;
        stats.peak_element_bytes = stats.peak_element_bytes.max(buffer.len());
        item(base..base + buffer.len() as u64, &buffer).map_err(ScanError::Callback)?;
    }
    Ok(stats)
}

#[cfg(test)]
mod tests;
