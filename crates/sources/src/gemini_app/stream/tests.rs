use super::*;
use std::io::{self, BufReader, Cursor, Read, Seek, SeekFrom};

#[test]
fn all_small_chunk_sizes_keep_original_cells_and_ignore_header_bytes() {
    let cells = [
        format!("{CELL} mdl-cell\">é🦆 &amp; first</div>"),
        format!("{CELL} mdl-cell\">second</div>"),
        format!("{CELL} mdl-cell\">last</div></html>"),
    ];
    let header = format!("<html>{}x{}", &CELL[..CELL.len() - 1], "h".repeat(20_000));
    let bytes = format!("{header}{}", cells.concat()).into_bytes();
    for capacity in 1..=65 {
        let mut input = BufReader::with_capacity(capacity, Cursor::new(&bytes));
        let mut at = header.len();
        let mut count = 0;
        let stats = scan(&mut input, |range, raw| {
            assert_eq!(raw, cells[count].as_bytes());
            assert_eq!(range, at as u64..(at + raw.len()) as u64);
            assert_eq!(raw, &bytes[at..at + raw.len()]);
            at += raw.len();
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 3, "capacity {capacity}");
        assert_eq!(stats.elements, 3);
        assert_eq!(at, bytes.len());
    }
    assert_eq!(
        scan(
            &mut Cursor::new(b"<html>no activities</html>"),
            |_, _| panic!("no cell")
        )
        .unwrap()
        .elements,
        0
    );
}

struct Failing {
    bytes: Cursor<Vec<u8>>,
    limit: u64,
}
impl Read for Failing {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.bytes.position() >= self.limit {
            return Err(io::Error::other("synthetic input failure"));
        }
        let n = out.len().min((self.limit - self.bytes.position()) as usize);
        self.bytes.read(&mut out[..n])
    }
}
impl Seek for Failing {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.bytes.seek(from)
    }
}
#[test]
fn source_and_callback_failures_stop_the_scan() {
    let first = format!("{CELL}\">first</div>");
    let bytes = format!("{first}{CELL}\">second</div>{CELL}\">third</div>").into_bytes();
    let mut calls = 0;
    let error = scan(
        &mut BufReader::with_capacity(
            5,
            Failing {
                bytes: Cursor::new(bytes.clone()),
                limit: (first.len() + CELL.len() + 3) as u64,
            },
        ),
        |_, _| {
            calls += 1;
            Ok(())
        },
    )
    .unwrap_err();
    assert!(matches!(error, ScanError::Input(_)), "{error}");
    assert_eq!(calls, 1);
    calls = 0;
    let error = scan(&mut Cursor::new(bytes), |_, _| {
        calls += 1;
        Err("synthetic callback failure".into())
    })
    .unwrap_err();
    assert!(matches!(error, ScanError::Callback(_)), "{error}");
    assert_eq!(calls, 1);
}

mod resource;
