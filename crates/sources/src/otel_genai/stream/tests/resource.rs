use super::*;
use std::io::{BufReader, Seek, SeekFrom};

struct Spans {
    part: Cursor<Vec<u8>>,
    next: usize,
}
impl Read for Spans {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            let count = self.part.read(out)?;
            if count > 0 {
                return Ok(count);
            }
            let bytes = match self.next {
                0 => br#"{"resourceSpans":[{"scopeSpans":[{"spans":["#.to_vec(),
                1..=12_000 => format!(
                    "{}{{\"spanId\":\"{}\",\"name\":\"{}\"}}",
                    if self.next == 1 { "" } else { "," },
                    self.next,
                    "x".repeat(2048)
                )
                .into_bytes(),
                12_001 => b"]}]}]}".to_vec(),
                _ => return Ok(0),
            };
            self.next += 1;
            self.part = Cursor::new(bytes);
        }
    }
}
impl Seek for Spans {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if from != SeekFrom::Start(0) {
            return Err(io::ErrorKind::Unsupported.into());
        }
        self.part = Cursor::new(vec![]);
        self.next = 0;
        Ok(0)
    }
}
#[cfg(target_os = "linux")]
#[test]
fn many_spans_scan_with_one_record_buffer() {
    fn peak() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .map(|s| s.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let before = peak();
    let mut input = BufReader::new(Spans {
        part: Cursor::new(vec![]),
        next: 0,
    });
    let mut sink = Counter::default();
    let (stats, refused) = scan(&mut input, &mut sink).unwrap();
    assert_eq!(refused, 0);
    assert_eq!(sink.accepted, 12_000);
    assert_eq!(stats.elements, 12_000);
    assert!(stats.peak_element_bytes < 4500, "{stats:?}");
    let growth = peak().saturating_sub(before);
    eprintln!(
        "12,000 spans: scan high-water growth {growth}KiB, record buffer {}bytes",
        stats.peak_element_bytes
    );
    assert!(growth < 8 * 1024, "span scan grew by {growth}KiB");
}
