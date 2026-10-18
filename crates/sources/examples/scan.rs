//! Parses files with an origin's parser and prints counts only: records by
//! disposition, unread wire fields, rows produced and throughput. For
//! checking a parser against real data without storing or printing any
//! content.
//!
//! ```text
//! cargo run --release -p polylogue-sources --example scan -- <origin> [--threads N] FILE... | @LIST
//! ```
//!
//! `@LIST` reads one path per line (a leading size column, as `find
//! -printf '%s %p\n'` writes it, is skipped). Each file is read in 64 MiB
//! steps with the carry passed between them, the way the engine does.

use polylogue_sources::Accounting;
use polylogue_sources::origin::by_name;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const STEP: usize = 64 << 20;

#[derive(Default)]
struct Totals {
    files: u64,
    bytes: u64,
    accounting: Accounting,
    messages: u64,
    events: u64,
    links: u64,
    background: u64,
    sessions: u64,
    failed: Vec<String>,
}

fn scan_file(origin: &str, path: &Path, t: &Mutex<Totals>) {
    let spec = by_name(origin).expect("known origin");
    let Some(parser) = (spec.open)(path) else {
        t.lock()
            .unwrap()
            .failed
            .push(format!("{}: not opened", path.display()));
        return;
    };
    let Ok(mut f) = File::open(path) else {
        t.lock()
            .unwrap()
            .failed
            .push(format!("{}: unreadable", path.display()));
        return;
    };
    let mut carry = None;
    let mut buf: Vec<u8> = Vec::new();
    let mut base = 0u64;
    let mut acc = Accounting::default();
    let (mut messages, mut events, mut links, mut background) = (0, 0, 0, 0);
    let mut sessions = 0;
    let mut read = 0u64;
    loop {
        let mut chunk = vec![0u8; STEP];
        let n = f.read(&mut chunk).unwrap_or(0);
        chunk.truncate(n);
        read += n as u64;
        buf.extend_from_slice(&chunk);
        match parser.step(carry.as_ref(), &buf, base) {
            Ok((step, next)) => {
                acc.merge(&step.accounting);
                messages += step.output.messages.len() as u64;
                events += step.output.events.len() as u64;
                links += step.output.links.len() as u64;
                background += step.output.background.len() as u64;
                sessions = sessions.max(step.output.sessions.len() as u64);
                carry = Some(next);
                let used = step.consumed as usize;
                buf.drain(..used);
                base += used as u64;
            }
            Err(e) => {
                t.lock()
                    .unwrap()
                    .failed
                    .push(format!("{}: {e}", path.display()));
                return;
            }
        }
        if n == 0 {
            break;
        }
    }
    let mut t = t.lock().unwrap();
    t.files += 1;
    t.bytes += read;
    t.accounting.merge(&acc);
    t.messages += messages;
    t.events += events;
    t.links += links;
    t.background += background;
    t.sessions += sessions;
    if !buf.is_empty() {
        // A trailing line without its newline: not a record yet.
        t.accounting
            .unknown
            .entry("<partial last line>".into())
            .and_modify(|n| *n += 1)
            .or_insert(1);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let origin = args.next().expect("origin name");
    let mut threads = 1;
    let mut files: Vec<PathBuf> = Vec::new();
    while let Some(a) = args.next() {
        if a == "--threads" {
            threads = args
                .next()
                .and_then(|n| n.parse().ok())
                .expect("--threads N");
        } else if let Some(list) = a.strip_prefix('@') {
            let list = std::fs::read_to_string(list).expect("path list");
            files.extend(list.lines().filter(|l| !l.trim().is_empty()).map(|l| {
                match l.split_once(' ') {
                    Some((size, path)) if size.chars().all(|c| c.is_ascii_digit()) => {
                        PathBuf::from(path)
                    }
                    _ => PathBuf::from(l),
                }
            }));
        } else {
            files.push(PathBuf::from(a));
        }
    }
    let totals = Mutex::new(Totals::default());
    let next = AtomicUsize::new(0);
    let started = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = files.get(i) else { break };
                    scan_file(&origin, path, &totals);
                }
            });
        }
    });
    let secs = started.elapsed().as_secs_f64();
    let t = totals.into_inner().unwrap();
    let top = |m: &BTreeMap<String, u64>| {
        let mut v: Vec<_> = m.iter().collect();
        v.sort_by(|a, b| b.1.cmp(a.1));
        v.into_iter()
            .map(|(k, n)| (k.clone(), *n))
            .collect::<Vec<_>>()
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "origin": origin,
            "files": t.files,
            "mb": t.bytes as f64 / 1e6,
            "seconds": secs,
            "mb_per_s": t.bytes as f64 / 1e6 / secs,
            "threads": threads,
            "records": t.accounting.framed,
            "materialized": t.accounting.materialized.values().sum::<u64>(),
            "unknown": t.accounting.unknown.values().sum::<u64>(),
            "refused": t.accounting.refused.values().sum::<u64>(),
            "messages": t.messages,
            "events": t.events,
            "links": t.links,
            "background_updates": t.background,
            "sessions": t.sessions,
            "failed_files": t.failed,
            "unknown_by_reason": top(&t.accounting.unknown),
            "refused_by_reason": top(&t.accounting.refused),
            "unread_fields": top(&t.accounting.unread_fields),
            "materialized_by_kind": top(&t.accounting.materialized),
        }))
        .unwrap()
    );
}
