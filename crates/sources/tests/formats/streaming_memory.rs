//! Streaming scans hold one record at a time: the peak heap a scan keeps
//! live on its thread stays near one record, however many records the
//! input has.

use polylogue_sources::export::ExportParser;
use polylogue_sources::openrouter::OpenRouter;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::{Read, Seek, SeekFrom};

/// Counts the bytes each thread holds live and the most it has held since
/// the last reset, so one test's figure is not mixed with other threads'.
struct ThreadCounting;

thread_local! {
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn add(bytes: isize) {
    let _ = LIVE.try_with(|live| {
        let now = live.get() + bytes;
        live.set(now);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
}

// SAFETY: every call is forwarded to `System`; the counters only observe.
#[allow(unsafe_code)]
unsafe impl GlobalAlloc for ThreadCounting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            add(layout.size() as isize);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        add(-(layout.size() as isize));
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            add(new_size as isize - layout.size() as isize);
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: ThreadCounting = ThreadCounting;

/// Peak live bytes on this thread while `work` runs, above what was live
/// when it started.
fn peak_growth(work: impl FnOnce()) -> isize {
    let start = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(start));
    work();
    PEAK.with(Cell::get) - start
}

/// An OpenRouter room export generated as it is read, so the input itself
/// is never held in memory: 12,000 messages of about 2 KiB each.
struct Room {
    part: std::io::Cursor<Vec<u8>>,
    next: usize,
}

const MESSAGES: usize = 12_000;

impl Read for Room {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        loop {
            let count = self.part.read(out)?;
            if count > 0 {
                return Ok(count);
            }
            let bytes = match self.next {
                0 => br#"{"version":"example","messages":{"#.to_vec(),
                n if n <= MESSAGES => format!(
                    "{}\"{n}\":{{\"id\":\"{n}\",\"type\":\"user\",\"content\":\"{}\"}}",
                    if n == 1 { "" } else { "," },
                    "x".repeat(2048)
                )
                .into_bytes(),
                n if n == MESSAGES + 1 => b"}}".to_vec(),
                _ => return Ok(0),
            };
            self.next += 1;
            self.part = std::io::Cursor::new(bytes);
        }
    }
}

impl Seek for Room {
    fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
        if from != SeekFrom::Start(0) {
            return Err(std::io::ErrorKind::Unsupported.into());
        }
        self.part = std::io::Cursor::new(vec![]);
        self.next = 0;
        Ok(0)
    }
}

#[test]
fn an_openrouter_room_scan_holds_one_message_at_a_time() {
    let mut count = 0;
    let mut stats = None;
    let growth = peak_growth(|| {
        let mut input = std::io::BufReader::new(Room {
            part: std::io::Cursor::new(vec![]),
            next: 0,
        });
        stats = Some(
            OpenRouter::default()
                .scan(
                    "synthetic-input",
                    &mut input,
                    &mut |_: std::ops::Range<u64>, _: Vec<u8>, _: Option<&str>, _: &[u8]| {
                        count += 1;
                        Ok(())
                    },
                )
                .unwrap(),
        );
    });
    let stats = stats.unwrap();
    assert_eq!(count, MESSAGES);
    assert_eq!(stats.elements, MESSAGES as u64);
    // The record buffer holds one message (2,048 content bytes and its
    // fields) and the scanner's copy of it.
    assert!(stats.peak_element_bytes < 4_500, "{stats:?}");
    // The whole input is about 24 MiB. Holding it, or any per-message
    // state, would exceed this by orders of magnitude; one message (about
    // 2 KiB, seen through the reader's 8 KiB buffer, the generator's
    // current part and the scanner's copy) fits many times over.
    assert!(growth < 64 * 1024, "scan held {growth} bytes at its peak");
}
