//! Counts-only parser scan; source text is never printed or stored.
//!
//! `cargo run -p polylogue-sources --example scan -- ORIGIN [--threads N] FILE... @LIST`
//!
//! Lists contain one path per line, optionally prefixed by a decimal size and
//! one space. Selection must be nonempty. Every selected input, including
//! duplicates, has one ordered outcome. Failed/incomplete inputs exit nonzero;
//! accounted unknown/refused records remain separate parser dispositions.
//!
//! Lines and capture logs use 64 MiB read steps and ordinary parser carry.
//! Documents use the parser's whole-input contract. Exports require
//! their item reader and are refused here. A trailing unconsumed suffix is
//! incomplete input, not an invented unknown record. Counts include rows
//! produced by completed steps, including upserts, not unique stored rows.

#[path = "scan/input.rs"]
mod input;

use polylogue_sources::{Accounting, Parse, ParseError};
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const STEP: usize = 64 << 20;

#[derive(Default)]
struct Counts {
    bytes: u64,
    accounting: Accounting,
    messages: u64,
    documents: u64,
    events: u64,
    links: u64,
    background: u64,
    sessions: u64,
}

#[derive(Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
enum Outcome {
    Complete,
    Failed { stage: &'static str, error: Failure },
    Incomplete { reason: &'static str },
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Failure {
    Io {
        kind: String,
        os_code: Option<i32>,
        message: String,
    },
    Parse {
        kind: &'static str,
    },
    UnsupportedPath,
}

impl Failure {
    fn io(error: std::io::Error) -> Self {
        Self::Io {
            kind: format!("{:?}", error.kind()),
            os_code: error.raw_os_error(),
            message: error.to_string(),
        }
    }
    fn parse(error: ParseError) -> Self {
        // Capture-log and carry diagnostics may contain source values. Keep the
        // error category, not the diagnostic text, in this counts-only tool.
        Self::Parse {
            kind: match error {
                ParseError::NotWhole(_) => "not_whole",
                ParseError::Export => "export_reader_required",
                ParseError::CaptureLog(_) => "capture_log",
                ParseError::Counts(_) => "counts",
                ParseError::Carry(_) => "carry",
            },
        }
    }
}

struct Scanned {
    outcome: Outcome,
    counts: Counts,
    remaining: usize,
}

fn scan_file(origin: &str, path: &Path) -> Scanned {
    let parser = (polylogue_sources::origin::by_name(origin)
        .expect("validated origin")
        .open)(path);
    let Some(parser) = parser else {
        return Scanned {
            outcome: Outcome::Failed {
                stage: "open_parser",
                error: Failure::UnsupportedPath,
            },
            counts: Counts::default(),
            remaining: 0,
        };
    };
    match File::open(path) {
        Ok(file) => scan_reader(&parser, file),
        Err(error) => Scanned {
            outcome: Outcome::Failed {
                stage: "open",
                error: Failure::io(error),
            },
            counts: Counts::default(),
            remaining: 0,
        },
    }
}

fn scan_reader(parser: &Parse, mut reader: impl Read) -> Scanned {
    let mut counts = Counts::default();
    let mut carry = None;
    let mut pending = Vec::new();
    let mut base = 0;
    let mut chunk = vec![0; STEP];
    loop {
        let n = match reader.read(&mut chunk) {
            Ok(n) => n,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => {
                return Scanned {
                    outcome: Outcome::Failed {
                        stage: "read",
                        error: Failure::io(error),
                    },
                    counts,
                    remaining: pending.len(),
                };
            }
        };
        counts.bytes += n as u64;
        pending.extend_from_slice(&chunk[..n]);
        if parser.is_document() && n != 0 {
            continue;
        }
        let (steps, next) = match parser.steps(carry.as_ref(), &pending, base) {
            Ok(result) => result,
            Err(error) => {
                return Scanned {
                    outcome: Outcome::Failed {
                        stage: "parse",
                        error: Failure::parse(error),
                    },
                    counts,
                    remaining: pending.len(),
                };
            }
        };
        let mut consumed = 0;
        for step in steps {
            consumed = consumed.max(step.consumed as usize);
            counts.accounting.merge(&step.accounting);
            counts.messages += step.output.messages.len() as u64;
            counts.documents += step.output.documents.len() as u64;
            counts.events += step.output.events.len() as u64;
            counts.links += step.output.links.len() as u64;
            counts.background += step.output.background.len() as u64;
            counts.sessions = counts.sessions.max(step.output.sessions.len() as u64);
        }
        carry = Some(next);
        pending.drain(..consumed);
        base += consumed as u64;
        if n == 0 {
            return Scanned {
                outcome: if pending.is_empty() {
                    Outcome::Complete
                } else {
                    Outcome::Incomplete {
                        reason: "unconsumed_tail",
                    }
                },
                counts,
                remaining: pending.len(),
            };
        }
    }
}

fn top(map: &BTreeMap<String, u64>) -> Vec<(&str, u64)> {
    let mut values: Vec<_> = map
        .iter()
        .map(|(key, count)| (key.as_str(), *count))
        .collect();
    values.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    values
}

fn run(selection: input::Selection) -> (serde_json::Value, bool) {
    let results: Mutex<Vec<Option<Scanned>>> =
        Mutex::new((0..selection.files.len()).map(|_| None).collect());
    let next = AtomicUsize::new(0);
    let started = Instant::now();
    let worker_error = std::thread::scope(|scope| {
        let mut workers = Vec::new();
        let mut error = None;
        for _ in 0..selection.threads.min(selection.files.len()) {
            match std::thread::Builder::new().spawn_scoped(scope, || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(path) = selection.files.get(index) else {
                        break;
                    };
                    let result = scan_file(&selection.origin, path);
                    results.lock().unwrap()[index] = Some(result);
                }
            }) {
                Ok(worker) => workers.push(worker),
                Err(failure) => {
                    error = Some(Failure::io(failure));
                    break;
                }
            }
        }
        for worker in workers {
            let _ = worker.join(); // Missing outcomes below make the run incomplete.
        }
        error
    });
    let mut total = Counts::default();
    let (mut complete, mut failed, mut incomplete) = (0, 0, 0);
    let mut outcomes = Vec::new();
    let mut failed_inputs = Vec::new();
    let mut unknown_bytes = false;
    for (index, result) in results.into_inner().unwrap().into_iter().enumerate() {
        let Some(result) = result else {
            incomplete += 1;
            unknown_bytes = true;
            outcomes.push(json!({"input": index, "outcome": {"status": "incomplete",
                "reason": "worker_did_not_finish"}, "bytes_read": null,
                "unconsumed_bytes": null}));
            continue;
        };
        match result.outcome {
            Outcome::Complete => complete += 1,
            Outcome::Failed { .. } => {
                failed += 1;
                failed_inputs.push(index);
            }
            Outcome::Incomplete { .. } => incomplete += 1,
        }
        outcomes.push(json!({"input": index, "outcome": result.outcome,
            "bytes_read": result.counts.bytes, "unconsumed_bytes": result.remaining,
            "records": result.counts.accounting.framed}));
        total.bytes += result.counts.bytes;
        total.accounting.merge(&result.counts.accounting);
        total.messages += result.counts.messages;
        total.documents += result.counts.documents;
        total.events += result.counts.events;
        total.links += result.counts.links;
        total.background += result.counts.background;
        total.sessions += result.counts.sessions;
    }
    let secs = started.elapsed().as_secs_f64();
    (
        json!({
            "origin": selection.origin, "selected_inputs": selection.files.len(),
            "worker_error": worker_error,
            "files": complete, "failed_inputs": failed, "incomplete_inputs": incomplete,
            "inputs": outcomes, "failed_files": failed_inputs,
            "bytes_read": (!unknown_bytes).then_some(total.bytes), "known_bytes_read": total.bytes,
            "mb": (!unknown_bytes).then_some(total.bytes as f64 / 1e6), "seconds": secs,
            "mb_per_s": (!unknown_bytes).then_some(total.bytes as f64 / 1e6 / secs), "threads": selection.threads,
            "records": total.accounting.framed,
            "materialized": total.accounting.materialized.values().sum::<u64>(),
            "unknown": total.accounting.unknown.values().sum::<u64>(),
            "refused": total.accounting.refused.values().sum::<u64>(),
            "messages": total.messages, "documents": total.documents, "events": total.events, "links": total.links,
            "background_updates": total.background, "sessions": total.sessions,
            "unknown_by_reason": top(&total.accounting.unknown),
            "refused_by_reason": top(&total.accounting.refused),
            "unread_fields": top(&total.accounting.unread_fields),
            "materialized_by_kind": top(&total.accounting.materialized),
        }),
        failed == 0 && incomplete == 0 && worker_error.is_none(),
    )
}

fn main() -> ExitCode {
    let selection = match input::selection(std::env::args().skip(1)) {
        Ok(selection) => selection,
        Err(error) => {
            eprintln!("scan: {error}");
            return ExitCode::from(2);
        }
    };
    // A parser panic must not print captured content from its panic payload.
    std::panic::set_hook(Box::new(|_| eprintln!("scan: worker failed")));
    let (report, complete) = run(selection);
    let written = writeln!(
        std::io::stdout().lock(),
        "{}",
        serde_json::to_string_pretty(&report).expect("counts serialize")
    )
    .is_ok();
    if complete && written {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(test)]
#[path = "scan/tests.rs"]
mod tests;
