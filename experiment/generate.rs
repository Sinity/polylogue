// The existing deterministic first-build generator, isolated from the engine.
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;

const GENERATOR_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

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
            "timestamp": "2026-09-01T10:00:00.000Z", "cwd": "/workspace/project",
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

fn generate_from_lengths(home: &Path, lengths_path: &Path) -> std::io::Result<u64> {
    let lengths = fs::read_to_string(lengths_path)?;
    let mut rng = Rng(GENERATOR_SEED);
    let project = home.join(".claude/projects/-generated");
    fs::create_dir_all(&project)?;
    let mut total = 0_u64;
    for (i, line) in lengths.split_ascii_whitespace().enumerate() {
        let length: u64 = line
            .parse()
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        let _size_draw = rng.next();
        let session = format!("{:08x}-gen{i}", rng.next() as u32);
        let written = session_file(
            &project.join(format!("{session}.jsonl")),
            &session,
            length,
            &mut rng,
        )?;
        if written != length {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("synthetic file {i} has {written} bytes, expected {length}"),
            ));
        }
        total += written;
    }
    Ok(total)
}

fn main() -> std::io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "generate-synthetic HOME LENGTHS");
    println!(
        "{}",
        generate_from_lengths(Path::new(&args[1]), Path::new(&args[2]))?
    );
    Ok(())
}
