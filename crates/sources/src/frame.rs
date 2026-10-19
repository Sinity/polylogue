//! Record framing. A record is one complete line; a trailing line without
//! its newline is not a record yet and is left for the next append.

/// One framed record: its bytes without the line ending, and where they
/// start in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}

/// Complete records in `bytes`, which start at `base` in the input, and how
/// many bytes they cover. Blank lines are skipped: they hold no record.
pub fn lines(bytes: &[u8], base: u64) -> (Vec<Record<'_>>, usize) {
    let mut records = Vec::new();
    let mut start = 0;
    while let Some(len) = bytes[start..].iter().position(|&b| b == b'\n') {
        let end = start + len;
        let mut line = &bytes[start..end];
        if let [rest @ .., b'\r'] = line {
            line = rest;
        }
        if !line.iter().all(u8::is_ascii_whitespace) {
            records.push(Record {
                offset: base + start as u64,
                bytes: line,
            });
        }
        start = end + 1;
    }
    (records, start)
}

#[cfg(test)]
mod tests;
