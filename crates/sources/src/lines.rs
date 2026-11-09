//! Splitting a line input into records. A record is one complete line. A trailing line without
//! its newline is not a record yet and is left for the next append, unless the input is
//! finished (an import, an export member): then the end of the bytes ends the last record.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

/// One line record: its bytes without the line ending, and where they
/// start in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}

/// Complete records in `bytes`, which start at `base` in the input, and how
/// many bytes they cover. Blank lines are skipped: they hold no record.
/// `finished` says the bytes reach the end of an input that will not grow,
/// so a trailing line without its newline is a record too.
pub fn lines<'a>(bytes: &'a [u8], base: u64, finished: bool) -> (Vec<Record<'a>>, usize) {
    let mut records = Vec::new();
    let mut push = |start: usize, mut line: &'a [u8]| {
        if let [rest @ .., b'\r'] = line {
            line = rest;
        }
        if !line.iter().all(u8::is_ascii_whitespace) {
            records.push(Record {
                offset: base + start as u64,
                bytes: line,
            });
        }
    };
    let mut start = 0;
    while let Some(len) = bytes[start..].iter().position(|&b| b == b'\n') {
        let end = start + len;
        push(start, &bytes[start..end]);
        start = end + 1;
    }
    if finished && start < bytes.len() {
        push(start, &bytes[start..]);
        start = bytes.len();
    }
    (records, start)
}

#[cfg(test)]
mod tests;
