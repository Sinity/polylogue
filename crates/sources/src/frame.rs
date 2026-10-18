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
mod tests {
    use super::*;

    #[test]
    fn leaves_a_partial_last_line() {
        let (records, consumed) = lines(b"{\"a\":1}\n{\"b\":", 100);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].offset, 100);
        assert_eq!(records[0].bytes, b"{\"a\":1}");
        assert_eq!(consumed, 8);
    }

    #[test]
    fn strips_crlf_and_skips_blank_lines() {
        let (records, consumed) = lines(b"a\r\n\n  \nb\n", 0);
        let got: Vec<_> = records.iter().map(|r| (r.offset, r.bytes)).collect();
        assert_eq!(got, [(0, &b"a"[..]), (7, &b"b"[..])]);
        assert_eq!(consumed, 9);
    }

    #[test]
    fn splitting_anywhere_frames_the_same_records() {
        let input = b"{\"x\":1}\n{\"y\":22}\n\n{\"z\":333}\n";
        let (whole, _) = lines(input, 0);
        for cut in 0..=input.len() {
            let (first, used) = lines(&input[..cut], 0);
            let (second, _) = lines(&input[used..], used as u64);
            let joined: Vec<_> = first.into_iter().chain(second).collect();
            assert_eq!(joined, whole, "cut at {cut}");
        }
    }
}
