//! Parses files with an origin's parser and prints counts only: records by
//! disposition, unread wire fields, and throughput. For checking a parser
//! against real data without storing or printing any content.
//!
//! `cargo run --release -p polylogue-sources --example scan -- claude-code FILE...`

use polylogue_sources::Accounting;
use polylogue_sources::origin::by_name;
use std::path::Path;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let origin = args.next().expect("origin name");
    let spec = by_name(&origin).expect("known origin");
    let mut total = Accounting::default();
    let (mut bytes, mut files, mut messages, mut failures) = (0u64, 0u64, 0u64, 0u64);
    let started = Instant::now();
    for path in args {
        let path = Path::new(&path);
        let Ok(input) = std::fs::read(path) else {
            failures += 1;
            continue;
        };
        let Some(parser) = (spec.open)(path) else {
            continue;
        };
        match parser.step(None, &input, 0) {
            Ok((step, _)) => {
                total.merge(&step.accounting);
                messages += step.output.messages.len() as u64;
                bytes += input.len() as u64;
                files += 1;
            }
            Err(e) => {
                eprintln!("{}: {e}", path.display());
                failures += 1;
            }
        }
    }
    let secs = started.elapsed().as_secs_f64();
    println!(
        "files {files} failures {failures} bytes {bytes} records {} message rows {messages} \
         seconds {secs:.2} MB/s {:.1}",
        total.framed,
        bytes as f64 / 1e6 / secs
    );
    let top = |name: &str, map: &std::collections::BTreeMap<String, u64>| {
        let mut v: Vec<_> = map.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        println!("{name}:");
        for (k, n) in v.into_iter().take(40) {
            println!("  {n:>9}  {k}");
        }
    };
    top("materialized", &total.materialized);
    top("unknown", &total.unknown);
    top("refused", &total.refused);
    top("unread fields", &total.unread_fields);
}
