use super::*;

struct Activities {
    part: Cursor<Vec<u8>>,
    next: usize,
}
impl Read for Activities {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            let count = self.part.read(out)?;
            if count > 0 {
                return Ok(count);
            }
            let bytes = match self.next {
                0 => b"<html>".to_vec(),
                1..=2048 => vec![b'h'; 4096],
                2049..=14048 => {
                    format!("{CELL} mdl-cell\">{}</div>", "x".repeat(2048)).into_bytes()
                }
                14049 => b"</html>".to_vec(),
                _ => return Ok(0),
            };
            self.next += 1;
            self.part = Cursor::new(bytes);
        }
    }
}
impl Seek for Activities {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        if from != SeekFrom::Start(0) {
            return Err(io::ErrorKind::Unsupported.into());
        }
        self.next = 0;
        self.part = Cursor::new(vec![]);
        Ok(0)
    }
}
#[cfg(target_os = "linux")]
#[test]
fn a_large_header_and_many_cells_keep_one_cell_buffer() {
    fn peak() -> u64 {
        std::fs::read_to_string("/proc/self/status")
            .unwrap()
            .lines()
            .find_map(|s| {
                s.strip_prefix("VmHWM:")
                    .map(|s| s.split_whitespace().next().unwrap().parse().unwrap())
            })
            .unwrap()
    }
    let before = peak();
    let mut count = 0;
    let stats = scan(
        &mut BufReader::new(Activities {
            part: Cursor::new(vec![]),
            next: 0,
        }),
        |_, raw| {
            assert!(raw.starts_with(CELL.as_bytes()));
            assert!(raw.len() < 2200);
            count += 1;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(count, 12_000);
    assert_eq!(stats.elements, 12_000);
    assert!(stats.peak_element_bytes < 2200, "{stats:?}");
    let growth = peak().saturating_sub(before);
    eprintln!(
        "8MiB header +12,000 cells: scan high-water growth {growth}KiB, cell buffer {}bytes",
        stats.peak_element_bytes
    );
    assert!(growth < 8 * 1024, "cell scan grew by {growth}KiB");
}
