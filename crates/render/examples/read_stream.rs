//! Synthetic comparison of incremental delivery and retaining the same layout.
//! Run each mode in its own process so Linux peak RSS remains comparable.
use polylogue_model::read::*;
use polylogue_render::{EncodeOptions, Encoder, Format};
use std::{
    io::{self, Write},
    time::Instant,
};
struct Sink {
    bytes: u64,
    largest: usize,
}
impl Write for Sink {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes += bytes.len() as u64;
        self.largest = self.largest.max(bytes.len());
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn item(n: usize) -> ReadItem {
    ReadItem {
        source: Source::record(format!("block:test:session:message{n}#0")),
        kind: ItemKind::Text {
            text: format!("record {n}: {} λ", "synthetic archived text ".repeat(40)),
        },
    }
}
fn main() -> io::Result<()> {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "stream".into());
    if !matches!(mode.as_str(), "stream" | "collect") {
        return Err(io::Error::other("use stream or collect"));
    }
    let header = ReadHeader {
        target: "session:test:session".into(),
        view: View::Transcript,
        title: ValueState::known("Synthetic stream".into()),
        kind: HeaderKind::Record {
            record_type: "benchmark".into(),
        },
        figures: vec![],
    };
    let mut sink = Sink {
        bytes: 0,
        largest: 0,
    };
    let start = Instant::now();
    let mut encoder = Encoder::new(
        &header,
        Format::Markdown,
        EncodeOptions::default(),
        &mut sink,
    )?;
    let mut first_item_ms = 0.0;
    if mode == "collect" {
        let items: Vec<_> = (0..100_000).map(item).collect();
        for (n, item) in items.iter().enumerate() {
            encoder.push(item)?;
            if n == 0 {
                first_item_ms = start.elapsed().as_secs_f64() * 1000.;
            }
        }
    } else {
        for n in 0..100_000 {
            encoder.push(&item(n))?;
            if n == 0 {
                first_item_ms = start.elapsed().as_secs_f64() * 1000.;
            }
        }
    }
    encoder.finish(&ReadFooter::default())?;
    let elapsed = start.elapsed().as_secs_f64() * 1000.;
    let peak = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines()
                .find(|line| line.starts_with("VmHWM:"))
                .map(str::to_owned)
        });
    println!(
        "mode={mode} records=100000 bytes={} largest_write={} elapsed_ms={elapsed:.3} first_item_ms={first_item_ms:.3} {}",
        sink.bytes,
        sink.largest,
        peak.as_deref().unwrap_or("VmHWM unavailable")
    );
    Ok(())
}
