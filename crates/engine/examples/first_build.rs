//! The first-build performance test (architecture 2, 9).
//!
//! ```text
//! first_build --generate-mb 2000 --sizes <stat.tsv> --dir <scratch>
//! first_build --home <dir with .claude/projects> --dir <scratch>
//! ```
//!
//! `--generate-mb` writes a corpus shaped like the operator's: file sizes
//! drawn from the measured size list (`stat.tsv` from the corpus
//! measurement: kind, bytes, ...; Claude Code session and subagent rows),
//! records in Claude Code's layout (one record per content block, tool
//! calls with results of varied size, progress and system records), and
//! prose from a vocabulary large enough for the search index to be real.
//! Then one pass of the loop builds an archive from nothing, and the run
//! prints input MB/s and the index size against the input.

use polylogue_engine::{Config, Engine};
use polylogue_store::Archive;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    /// A length with a long tail: mostly short, sometimes very long.
    fn heavy(&mut self, typical: u64, max: u64) -> u64 {
        let r = (self.next() % 10_000) as f64 / 10_000.0;
        ((typical as f64) * (-(1.0 - r).ln()).powf(2.0)).min(max as f64) as u64 + 1
    }
}

const WORDS: &[&str] = &[
    "parser",
    "archive",
    "session",
    "index",
    "message",
    "daemon",
    "writer",
    "query",
    "token",
    "commit",
    "branch",
    "review",
    "failure",
    "fixture",
    "golden",
    "record",
    "offset",
    "chain",
    "revision",
    "extent",
    "blob",
    "search",
    "render",
    "budget",
    "context",
    "brief",
    "lineage",
    "subagent",
    "compaction",
    "the",
    "a",
    "to",
    "of",
    "and",
    "in",
    "is",
    "it",
    "that",
    "for",
    "with",
    "this",
    "we",
    "should",
    "check",
    "test",
    "build",
    "error",
    "warning",
    "cargo",
    "nix",
    "rust",
    "python",
    "sqlite",
    "fts",
    "trigram",
    "operator",
    "measure",
    "throughput",
    "latency",
    "memory",
    "cache",
    "thread",
    "pool",
    "channel",
    "batch",
    "transaction",
    "schema",
    "migration",
];

fn prose(rng: &mut Rng, len: u64) -> String {
    let mut s = String::with_capacity(len as usize + 16);
    while (s.len() as u64) < len {
        s.push_str(WORDS[rng.below(WORDS.len() as u64) as usize]);
        if rng.below(12) == 0 {
            s.push_str(&rng.below(100_000).to_string());
        }
        s.push(if rng.below(15) == 0 { '\n' } else { ' ' });
    }
    s
}

fn prose_h(rng: &mut Rng, typical: u64, max: u64) -> String {
    let len = rng.heavy(typical, max);
    prose(rng, len)
}

/// Writes one Claude Code session file of about `target` bytes.
fn session_file(path: &Path, session: &str, target: u64, rng: &mut Rng) -> std::io::Result<u64> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    let mut written = 0u64;
    let mut n = 0u64;
    let mut parent = serde_json::Value::Null;
    let emit = |w: &mut BufWriter<fs::File>, v: serde_json::Value| -> std::io::Result<u64> {
        let line = v.to_string();
        w.write_all(line.as_bytes())?;
        w.write_all(b"\n")?;
        Ok(line.len() as u64 + 1)
    };
    let base = |kind: &str, uuid: &str, parent: &serde_json::Value| {
        serde_json::json!({
            "type": kind, "uuid": uuid, "parentUuid": parent, "sessionId": session,
            "timestamp": "2026-09-01T10:00:00.000Z", "cwd": "/realm/project/polylogue/repo",
            "gitBranch": "master", "version": "2.1.200", "isSidechain": false,
            "userType": "external",
        })
    };
    while written < target {
        n += 1;
        let u = format!("{session}-u{n}");
        let mut rec = base("user", &u, &parent);
        rec["message"] = serde_json::json!({"role": "user", "content": prose_h(rng, 200, 20_000)});
        written += emit(&mut w, rec)?;
        parent = u.clone().into();
        // A response: thinking, text and one to four tool calls, one
        // record per block, then a tool result record per call.
        let msg_id = format!("msg_{session}_{n}");
        let calls = 1 + rng.below(4);
        let mut blocks = vec![
            serde_json::json!({"type": "thinking", "thinking": prose_h(rng, 300, 8_000), "signature": "sig"}),
            serde_json::json!({"type": "text", "text": prose_h(rng, 300, 10_000)}),
        ];
        for c in 0..calls {
            blocks.push(serde_json::json!({
                "type": "tool_use", "id": format!("toolu_{session}_{n}_{c}"), "name": "Bash",
                "input": {"command": format!("cargo test -p crate{}", rng.below(20)), "description": "run tests"}
            }));
        }
        for (i, block) in blocks.iter().enumerate() {
            let a = format!("{session}-a{n}-{i}");
            let mut rec = base("assistant", &a, &parent);
            rec["requestId"] = format!("req_{n}").into();
            rec["message"] = serde_json::json!({
                "id": msg_id, "role": "assistant", "model": "claude-opus-5-5", "type": "message",
                "content": [block],
                "stop_reason": if i + 1 == blocks.len() { serde_json::json!("tool_use") } else { serde_json::Value::Null },
                "usage": {"input_tokens": 10, "output_tokens": 300, "cache_read_input_tokens": 40_000,
                          "cache_creation_input_tokens": 2_000, "service_tier": "standard"}
            });
            written += emit(&mut w, rec)?;
            parent = a.into();
        }
        for c in 0..calls {
            let r = format!("{session}-r{n}-{c}");
            let mut rec = base("user", &r, &parent);
            rec["message"] = serde_json::json!({"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": format!("toolu_{session}_{n}_{c}"),
                "content": prose_h(rng, 1_500, 2_000_000), "is_error": rng.below(8) == 0
            }]});
            written += emit(&mut w, rec)?;
            parent = r.into();
        }
        if rng.below(3) == 0 {
            let mut rec = base("progress", &format!("{session}-p{n}"), &parent);
            rec["data"] = serde_json::json!({"type": "bash_progress", "output": prose(rng, 200)});
            written += emit(&mut w, rec)?;
        }
    }
    w.flush()?;
    Ok(written)
}

fn generate(home: &Path, sizes_path: &Path, total_mb: u64) -> std::io::Result<u64> {
    let mut sizes: Vec<u64> = fs::read_to_string(sizes_path)?
        .lines()
        .filter_map(|l| {
            let mut f = l.split('\t');
            let kind = f.next()?;
            let size: u64 = f.next()?.parse().ok()?;
            matches!(kind, "claude-jsonl" | "claude-subagent").then_some(size)
        })
        .collect();
    sizes.sort_unstable();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let project = home.join(".claude/projects/-generated");
    fs::create_dir_all(&project)?;
    let target = total_mb << 20;
    let mut total = 0;
    let mut i = 0;
    while total < target {
        let size = sizes[rng.below(sizes.len() as u64) as usize].max(500);
        let session = format!("{:08x}-gen{i}", rng.next() as u32);
        total += session_file(
            &project.join(format!("{session}.jsonl")),
            &session,
            size,
            &mut rng,
        )?;
        i += 1;
    }
    eprintln!("generated {i} files, {:.1} MB", total as f64 / 1e6);
    Ok(total)
}

fn dir_size(p: &Path) -> u64 {
    let Ok(rd) = fs::read_dir(p) else {
        return fs::metadata(p).map_or(0, |m| m.len());
    };
    rd.filter_map(Result::ok)
        .map(|e| {
            let p = e.path();
            if p.is_dir() {
                dir_size(&p)
            } else {
                e.metadata().map_or(0, |m| m.len())
            }
        })
        .sum()
}

fn input_size(config: &Config) -> u64 {
    polylogue_engine::collect::scan(config)
        .0
        .iter()
        .filter_map(|f| fs::metadata(&f.path).ok())
        .map(|m| m.len())
        .sum()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let dir = PathBuf::from(arg("--dir").expect("--dir <scratch directory>"));
    let home = match arg("--home") {
        Some(h) => PathBuf::from(h),
        None => {
            let home = dir.join("home");
            let mb: u64 = arg("--generate-mb")
                .expect("--home or --generate-mb")
                .parse()
                .unwrap();
            let sizes = PathBuf::from(arg("--sizes").expect("--sizes <stat.tsv>"));
            generate(&home, &sizes, mb).unwrap();
            home
        }
    };
    let root = dir.join("archive");
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    let mut config = Config::from_env();
    config.home = home;
    // The measured tree is exactly `--home`; the tools' own directory
    // variables point at the real one.
    config.tool_dirs.clear();
    let input = input_size(&config);
    if let Some(w) = arg("--workers") {
        config.workers = w.parse().unwrap();
    }
    let workers = config.workers;
    let mut engine = Engine::new(config, archive.clone()).unwrap();
    let started = Instant::now();
    let pass = engine.converge().unwrap();
    let secs = started.elapsed().as_secs_f64();
    drop(engine);
    // Bytes this process made the kernel read and write, for write
    // amplification (bytes written against bytes kept).
    let io = fs::read_to_string("/proc/self/io").unwrap_or_default();
    let io_field = |name: &str| {
        io.lines()
            .find_map(|l| l.strip_prefix(name))
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    };
    let (read_bytes, write_bytes) = (io_field("read_bytes:"), io_field("write_bytes:"));
    let index = dir_size(&archive.index_db()) + dir_size(&root.join("index.db-wal"));
    let stored = dir_size(&archive.archive_db()) + dir_size(&archive.blobs_dir());
    println!(
        "{}",
        serde_json::json!({
            "workers": workers,
            "inputs": pass.inputs_changed,
            "input_mb": input as f64 / 1e6,
            "seconds": secs,
            "mb_per_s": input as f64 / 1e6 / secs,
            "messages_written": pass.messages_written,
            "index_mb": index as f64 / 1e6,
            "index_over_input": index as f64 / input.max(1) as f64,
            "stored_mb": stored as f64 / 1e6,
            "disk_read_mb": read_bytes as f64 / 1e6,
            "disk_written_mb": write_bytes as f64 / 1e6,
            "written_over_kept": write_bytes as f64 / (index + stored).max(1) as f64,
            "errors": pass.errors.len(),
            "first_errors": pass.errors.iter().take(5).collect::<Vec<_>>(),
        })
    );
}
