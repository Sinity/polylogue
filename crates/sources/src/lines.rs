//! Splitting a line input into records. A record is one complete line. A trailing line without
//! its newline is not a record yet and is left for the next append, unless the input is
//! finished (an import, an export member): then the end of the bytes ends the last record.

/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use crate::document::after_byte_order_mark;

/// One line record: its bytes without the line ending, and where they
/// start in the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Record<'a> {
    pub offset: u64,
    pub bytes: &'a [u8],
}

/// Complete records in `bytes`, which start at `input_offset` in the input, and how
/// many bytes they cover. Blank lines are skipped: they hold no record.
/// `finished` says the bytes reach the end of an input that will not grow,
/// so a trailing line without its newline is a record too. A byte order mark
/// at the start of the input is not part of the first record; its offset
/// still counts the mark.
pub fn lines<'a>(bytes: &'a [u8], input_offset: u64, finished: bool) -> (Vec<Record<'a>>, usize) {
    let mut records = Vec::new();
    let mut push = |line_start: usize, mut line: &'a [u8]| {
        let mut offset = input_offset + line_start as u64;
        if input_offset == 0 && line_start == 0 {
            let body = after_byte_order_mark(line);
            offset += (line.len() - body.len()) as u64;
            line = body;
        }
        if let [rest @ .., b'\r'] = line {
            line = rest;
        }
        if !line.iter().all(u8::is_ascii_whitespace) {
            records.push(Record {
                offset,
                bytes: line,
            });
        }
    };
    let mut consumed = 0;
    while let Some(len) = bytes[consumed..].iter().position(|&b| b == b'\n') {
        let end = consumed + len;
        push(consumed, &bytes[consumed..end]);
        consumed = end + 1;
    }
    if finished && consumed < bytes.len() {
        push(consumed, &bytes[consumed..]);
        consumed = bytes.len();
    }
    (records, consumed)
}

#[cfg(test)]
mod tests;
