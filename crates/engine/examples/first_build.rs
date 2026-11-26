//! The first-build performance test.
//!
//! ```text
//! first_build --generate-mb 2000 --sizes <stat.tsv> --dir <scratch>
//! first_build --home <dir with .claude/projects> --dir <scratch>
//! ```
//!
//! `--max-rss-mb N` fails the run when its peak resident memory exceeds N.
//!
//! `--from-record` stores the corpus first, unmeasured, and removes the
//! index; the measured pass then builds the index from the stored record.
//!
//! `--generate-mb` writes a corpus shaped like the operator's, with prose
//! from a vocabulary large enough for the search index to be real. Then one
//! pass of the loop builds an archive from nothing, and the run prints input
//! MB/s and the index size against the input.
//!
//! `--shape claude-code` (the default) writes Claude Code sessions: file
//! sizes drawn from the measured size list (`--sizes stat.tsv` from the
//! corpus measurement: kind, bytes, ...; its `claude-jsonl` and
//! `claude-subagent` rows), records in Claude Code's layout (one record per
//! content block, tool calls with results of varied size, progress records).
//!
//! `--shape codex` writes Codex rollouts under `.codex/sessions/`, with
//! bytes shared out as in real rollouts: about 38% `item_completed` events,
//! mostly command executions that carry their output three times; 25% call
//! outputs carrying it once more; 11% reasoning items, mostly an opaque
//! encrypted string; 10% compactions restating the session's earlier
//! messages; 6% token counts, which are most event records by count; and
//! the rest calls, messages, their mirroring events and turn contexts. File
//! sizes come from the `codex-rollout` rows of `--sizes` when given, else
//! from a log-normal (median 1.5 MB, sigma 1.7) under which the 1% largest
//! files hold about a quarter of the bytes. `--largest-mb N` adds one
//! rollout of N MB, counted in the total, for one very long chain.
//!
//! ```text
//! first_build --shape codex --generate-mb 4000 --largest-mb 600 --from-record --dir <scratch>
//! ```

use polylogue_engine::{Config, Engine};
use polylogue_store::Archive;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// The daemon's allocator, so the measured memory is the daemon's.
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

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
    /// Uniform in (0, 1].
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
    /// Log-normal with the given median and sigma (Box-Muller).
    fn log_normal(&mut self, median: f64, sigma: f64) -> u64 {
        let z = (-2.0 * self.unit().ln()).sqrt() * (std::f64::consts::TAU * self.unit()).cos();
        (median * (sigma * z).exp()) as u64
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

/// Writes one JSON record as a line; returns its length.
fn emit(w: &mut BufWriter<fs::File>, v: serde_json::Value) -> std::io::Result<u64> {
    let line = v.to_string();
    w.write_all(line.as_bytes())?;
    w.write_all(b"\n")?;
    Ok(line.len() as u64 + 1)
}

/// Writes one Claude Code session file of about `target` bytes.
fn session_file(path: &Path, session: &str, target: u64, rng: &mut Rng) -> std::io::Result<u64> {
    let mut w = BufWriter::new(fs::File::create(path)?);
    let mut written = 0u64;
    let mut n = 0u64;
    let mut parent = serde_json::Value::Null;
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

/// Random base64, as incompressible as a real encrypted reasoning string.
fn opaque(rng: &mut Rng, len: u64) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(len as usize + 10);
    while (s.len() as u64) < len {
        let mut bits = rng.next();
        for _ in 0..10 {
            s.push(ALPHABET[(bits & 63) as usize] as char);
            bits >>= 6;
        }
    }
    s
}

/// One Codex rollout being written: outer records `{timestamp, type,
/// payload}`, and what its compactions restate.
struct RolloutWriter<'a> {
    w: BufWriter<fs::File>,
    rng: &'a mut Rng,
    thread: String,
    start: jiff::Timestamp,
    seconds: i64,
    written: u64,
    compacted: u64,
    tokens: u64,
    /// The latest user and assistant message items, for compactions.
    history: std::collections::VecDeque<serde_json::Value>,
}

impl RolloutWriter<'_> {
    fn record(&mut self, kind: &str, payload: serde_json::Value) -> std::io::Result<u64> {
        self.seconds += 1 + self.rng.below(20) as i64;
        let at = jiff::Timestamp::from_second(self.start.as_second() + self.seconds).unwrap();
        let n = emit(
            &mut self.w,
            serde_json::json!({"timestamp": at.to_string(), "type": kind, "payload": payload}),
        )?;
        self.written += n;
        Ok(n)
    }

    fn event(&mut self, payload: serde_json::Value) -> std::io::Result<u64> {
        self.record("event_msg", payload)
    }

    fn item(&mut self, payload: serde_json::Value) -> std::io::Result<u64> {
        self.record("response_item", payload)
    }

    /// A token count; the rate limits ride on one count per turn.
    fn token_count(&mut self, limits: bool) -> std::io::Result<u64> {
        let last = 20_000 + self.rng.below(100_000);
        self.tokens += last;
        let usage = |input: u64| {
            serde_json::json!({"input_tokens": input, "cached_input_tokens": input / 10 * 9,
            "output_tokens": input / 50, "reasoning_output_tokens": input / 100, "total_tokens": input + input / 50})
        };
        let mut payload = serde_json::json!({"type": "token_count",
            "info": {"total_token_usage": usage(self.tokens), "last_token_usage": usage(last), "model_context_window": 258_400}});
        if limits {
            let used = self.rng.below(100);
            payload["rate_limits"] = serde_json::json!({"limit_id": "codex", "plan_type": "pro",
                "primary": {"used_percent": used, "window_minutes": 300, "resets_at": 1_788_300_000},
                "secondary": {"used_percent": used / 4, "window_minutes": 10_080, "resets_at": 1_788_800_000}});
        }
        self.event(payload)
    }

    /// A user or assistant message item, remembered for compactions.
    fn message(&mut self, role: &str, id: String, text: String) -> std::io::Result<u64> {
        let part = if role == "user" {
            "input_text"
        } else {
            "output_text"
        };
        let item = serde_json::json!({"type": "message", "id": id, "role": role,
            "content": [{"type": part, "text": text}]});
        self.history.push_back(item.clone());
        if self.history.len() > 64 {
            self.history.pop_front();
        }
        self.item(item)
    }

    /// One tool call, its completion record and its output: a code-mode
    /// script whose command the harness records with its output three
    /// times, a direct command whose completion carries only its status, or
    /// a patch.
    fn call(&mut self, turn: u64, index: u64) -> std::io::Result<()> {
        let id = &format!("call_{}_{turn}_{index}", self.thread);
        let command = format!("cargo test -p crate{}", self.rng.below(20));
        let output = prose_h(self.rng, 2_800, 4_000_000);
        let exit_code = if self.rng.below(8) == 0 { 101 } else { 0 };
        let duration =
            serde_json::json!({"secs": self.rng.below(90), "nanos": self.rng.below(1_000_000_000)});
        let (call, completed, result) = match self.rng.below(16) {
            0..7 => (
                serde_json::json!({"type": "custom_tool_call", "id": format!("ctc_{id}"),
                    "call_id": id, "name": "exec", "status": "completed",
                    "input": format!("text(await tools.exec_command({{cmd:\"{command}\"}}));\n")}),
                serde_json::json!({"type": "CommandExecution", "id": format!("{id}_cmd"),
                    "command": command, "cwd": "/workspace/project",
                    "status": "completed", "exit_code": exit_code, "duration": duration,
                    "aggregated_output": output, "stdout": output, "formatted_output": output}),
                serde_json::json!({"type": "custom_tool_call_output", "id": format!("ctco_{id}"),
                    "call_id": id, "output": [{"type": "input_text", "text": "Script completed\nOutput:\n"},
                                              {"type": "input_text", "text": output}]}),
            ),
            7..14 => (
                serde_json::json!({"type": "function_call", "id": format!("fc_{id}"),
                    "call_id": id, "name": "exec_command",
                    "arguments": serde_json::json!({"cmd": command, "workdir": "/workspace/project"}).to_string()}),
                serde_json::json!({"type": "CommandExecution", "id": id, "command": command,
                    "status": "completed", "exit_code": exit_code, "duration": duration}),
                serde_json::json!({"type": "function_call_output", "call_id": id,
                    "output": format!("Chunk ID: {id}\nWall time: 1.5 seconds\nProcess exited with code {exit_code}\nOutput:\n{output}")}),
            ),
            _ => {
                let file = format!("src/module{}.rs", self.rng.below(50));
                let patch = format!(
                    "*** Begin Patch\n*** Update File: {file}\n@@\n-{}\n+{}\n*** End Patch\n",
                    prose_h(self.rng, 200, 20_000),
                    prose_h(self.rng, 200, 20_000)
                );
                (
                    serde_json::json!({"type": "custom_tool_call", "id": format!("ctc_{id}"),
                        "call_id": id, "name": "apply_patch", "status": "completed", "input": patch}),
                    serde_json::json!({"type": "FileChange", "id": id, "status": "completed",
                        "changes": {file.clone(): {"type": "update"}}}),
                    serde_json::json!({"type": "custom_tool_call_output", "id": format!("ctco_{id}"),
                        "call_id": id, "output": format!("Success. Updated the following files:\nM {file}")}),
                )
            }
        };
        self.item(call)?;
        self.token_count(false)?;
        self.event(
            serde_json::json!({"type": "item_completed", "thread_id": self.thread,
            "turn_id": format!("turn-{turn}"), "item": completed}),
        )?;
        self.item(result)?;
        self.token_count(false)?;
        Ok(())
    }

    /// One turn: context, prompt, reasoning and calls, the answer, and a
    /// compaction when compactions are behind their share of the bytes.
    fn turn(&mut self, n: u64) -> std::io::Result<()> {
        self.record(
            "turn_context",
            serde_json::json!({"cwd": "/workspace/project", "model": "gpt-6.1-sol",
            "effort": "high", "summary": "auto", "approval_policy": "never",
            "sandbox_policy": {"type": "danger-full-access"}, "timezone": "UTC"}),
        )?;
        if n == 0 {
            let text = format!(
                "# AGENTS.md instructions for /workspace/project\n\n{}",
                prose(self.rng, 6_000)
            );
            self.message("user", format!("msg_{}_agents", self.thread), text)?;
        }
        let prompt = prose_h(self.rng, 300, 30_000);
        self.message("user", format!("msg_{}_{n}_u", self.thread), prompt.clone())?;
        self.event(serde_json::json!({"type": "user_message", "message": prompt}))?;
        self.token_count(false)?;
        for c in 0..1 + self.rng.below(6) {
            let summary: Vec<_> = (0..self.rng.below(3))
                .map(|_| serde_json::json!({"type": "summary_text", "text": prose_h(self.rng, 150, 2_000)}))
                .collect();
            let length = self.rng.heavy(1_000, 100_000);
            let encrypted = opaque(self.rng, length);
            self.item(
                serde_json::json!({"type": "reasoning", "id": format!("rs_{}_{n}_{c}", self.thread),
                "summary": summary, "encrypted_content": encrypted}),
            )?;
            self.token_count(false)?;
            self.call(n, c)?;
        }
        let answer = prose_h(self.rng, 400, 20_000);
        self.message(
            "assistant",
            format!("msg_{}_{n}_a", self.thread),
            answer.clone(),
        )?;
        self.event(serde_json::json!({"type": "agent_message", "message": answer}))?;
        self.token_count(true)?;
        self.event(serde_json::json!({"type": "task_complete", "turn_id": format!("turn-{n}")}))?;
        if self.compacted * 10 < self.written {
            let summary = prose_h(self.rng, 1_500, 20_000);
            let history: Vec<_> = self.history.iter().cloned().collect();
            let n = self.record(
                "compacted",
                serde_json::json!({"message": summary, "replacement_history": history}),
            )?;
            self.compacted += n;
        }
        Ok(())
    }
}

/// Writes one Codex rollout of about `target` bytes under `sessions`.
fn rollout_file(sessions: &Path, index: u64, target: u64, rng: &mut Rng) -> std::io::Result<u64> {
    let r = rng.next();
    let thread = format!(
        "{:08x}-{:04x}-7{:03x}-8{:03x}-{:012x}",
        r >> 32,
        r & 0xffff,
        index & 0xfff,
        (index >> 12) & 0xfff,
        rng.next() & 0xffff_ffff_ffff
    );
    let base: jiff::Timestamp = "2026-09-01T00:00:00Z".parse().unwrap();
    let start = jiff::Timestamp::from_second(base.as_second() + index as i64 * 5_000).unwrap();
    let dir = sessions.join(start.strftime("%Y/%m/%d").to_string());
    fs::create_dir_all(&dir)?;
    let name = format!(
        "rollout-{}-{thread}.jsonl",
        start.strftime("%Y-%m-%dT%H-%M-%S")
    );
    let mut rollout = RolloutWriter {
        w: BufWriter::new(fs::File::create(dir.join(name))?),
        rng,
        thread,
        start,
        seconds: 0,
        written: 0,
        compacted: 0,
        tokens: 0,
        history: Default::default(),
    };
    let meta = serde_json::json!({"id": rollout.thread, "timestamp": start.to_string(),
        "cwd": "/workspace/project", "originator": "codex_cli_rs", "cli_version": "0.170.0", "source": "cli",
        "thread_source": "user", "model_provider": "openai",
        "base_instructions": {"text": prose(rollout.rng, 2_000)},
        "git": {"commit_hash": format!("{:016x}", rollout.rng.next()), "branch": "master",
                "repository_url": "https://example.com/project.git"}});
    rollout.record("session_meta", meta)?;
    let mut n = 0;
    while rollout.written < target {
        rollout.turn(n)?;
        n += 1;
    }
    rollout.w.flush()?;
    Ok(rollout.written)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    ClaudeCode,
    Codex,
}

fn generate(
    home: &Path,
    shape: Shape,
    sizes_path: Option<&Path>,
    total_mb: u64,
    largest_mb: Option<u64>,
) -> std::io::Result<u64> {
    let kinds: &[&str] = match shape {
        Shape::ClaudeCode => &["claude-jsonl", "claude-subagent"],
        Shape::Codex => &["codex-rollout"],
    };
    let mut sizes: Vec<u64> = match sizes_path {
        Some(path) => fs::read_to_string(path)?
            .lines()
            .filter_map(|l| {
                let mut f = l.split('\t');
                let kind = f.next()?;
                let size: u64 = f.next()?.parse().ok()?;
                kinds.contains(&kind).then_some(size)
            })
            .collect(),
        None if shape == Shape::Codex => Vec::new(),
        None => panic!("--sizes <stat.tsv> for the Claude Code shape"),
    };
    sizes.sort_unstable();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let project = home.join(".claude/projects/-generated");
    let sessions = home.join(".codex/sessions");
    let mut largest = largest_mb.map(|mb| mb << 20);
    let target = (total_mb << 20).max(largest.unwrap_or(0));
    let mut total = 0;
    let mut i = 0;
    while total < target {
        let drawn = largest.take().unwrap_or_else(|| {
            if sizes.is_empty() {
                rng.log_normal(1.5e6, 1.7)
            } else {
                sizes[rng.below(sizes.len() as u64) as usize]
            }
        });
        let size = drawn.min(target - total).max(500);
        total += match shape {
            Shape::ClaudeCode => {
                fs::create_dir_all(&project)?;
                let session = format!("{:08x}-gen{i}", rng.next() as u32);
                session_file(
                    &project.join(format!("{session}.jsonl")),
                    &session,
                    size,
                    &mut rng,
                )?
            }
            Shape::Codex => rollout_file(&sessions, i, size, &mut rng)?,
        };
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
    polylogue_store::timing::set_workers_enabled(
        std::env::var_os("POLYLOGUE_TIMING_WORKERS").is_some(),
    );
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    if let Some(index) = arg("--compression-sizes") {
        println!("{}", compression_sizes(Path::new(&index)).unwrap());
        return;
    }
    let dir = PathBuf::from(arg("--dir").expect("--dir <scratch directory>"));
    let home = match arg("--home") {
        Some(h) => PathBuf::from(h),
        None => {
            let home = dir.join("home");
            let mb: u64 = arg("--generate-mb")
                .expect("--home or --generate-mb")
                .parse()
                .unwrap();
            let shape = match arg("--shape").as_deref() {
                None | Some("claude-code") => Shape::ClaudeCode,
                Some("codex") => Shape::Codex,
                Some(other) => panic!("--shape claude-code|codex, not {other}"),
            };
            let sizes = arg("--sizes").map(PathBuf::from);
            let largest = arg("--largest-mb").map(|mb| mb.parse().expect("--largest-mb MB"));
            generate(&home, shape, sizes.as_deref(), mb, largest).unwrap();
            home
        }
    };
    if args.iter().any(|arg| arg == "--generate-only") {
        return;
    }
    if args.iter().any(|arg| arg == "--capture-only") {
        capture_only(&home, &dir.join("archive")).unwrap();
        return;
    }
    let from_record = args.iter().any(|arg| arg == "--from-record");
    if from_record {
        capture_only(&home, &dir.join("archive")).unwrap();
        for suffix in ["", "-wal", "-shm"] {
            let _ = fs::remove_file(dir.join(format!("archive/index.db{suffix}")));
        }
    }
    let io_before = fs::read_to_string("/proc/self/io").ok();
    let cpu_before = fs::read_to_string("/proc/self/stat").ok();
    let started = Instant::now();
    let root = dir.join("archive");
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    let mut config = measurement_config(home);
    // The measured tree is exactly `--home`; the tools' own directory
    // variables point at the real one.
    config.tool_dirs.clear();
    // The stored corpus is the record; its files are not discovered again.
    config.reads_sources = !from_record;
    let input = input_size(&config);
    if let Some(w) = arg("--workers") {
        config.workers = w.parse().unwrap();
    }
    let workers = config.workers;
    let mut engine = Engine::new(config, archive.clone()).unwrap();
    let setup_seconds = started.elapsed().as_secs_f64();
    let ingest_started = Instant::now();
    // Seconds per stage so far, on stderr each minute: a run stopped before
    // it ends still shows where its time went.
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(60));
            let stages: serde_json::Map<_, _> = polylogue_store::timing::stage_seconds()
                .into_iter()
                .map(|(s, secs)| (s.name().to_owned(), serde_json::json!(secs.round())))
                .collect();
            eprintln!(
                "{}",
                serde_json::json!({"seconds": ingest_started.elapsed().as_secs(), "stages": stages})
            );
        }
    });
    let pass = engine.run_pass().unwrap();
    drop(engine);
    let ingest_seconds = ingest_started.elapsed().as_secs_f64();
    let engine_seconds = started.elapsed().as_secs_f64();
    let checkpoint_started = Instant::now();
    let databases = checkpoint_build(&archive).unwrap();
    let checkpoint_seconds = checkpoint_started.elapsed().as_secs_f64();
    let secs = started.elapsed().as_secs_f64();
    // Both arms include the same final WAL checkpoint into the database file. This explicit
    // benchmark step is separate from normal engine shutdown.

    let io = fs::read_to_string("/proc/self/io").ok();
    let read_bytes = counter_delta(
        io_count(io_before.as_deref(), "read_bytes:"),
        io_count(io.as_deref(), "read_bytes:"),
    );
    let write_bytes = counter_delta(
        io_count(io_before.as_deref(), "write_bytes:"),
        io_count(io.as_deref(), "write_bytes:"),
    );
    let cancelled_write_bytes = counter_delta(
        io_count(io_before.as_deref(), "cancelled_write_bytes:"),
        io_count(io.as_deref(), "cancelled_write_bytes:"),
    );
    let cpu_after = fs::read_to_string("/proc/self/stat").ok();
    let cpu_ticks = counter_delta(
        cpu_ticks(cpu_before.as_deref()),
        cpu_ticks(cpu_after.as_deref()),
    );
    // Write-call bytes include tmpfs work as well as physical disk writes.
    let wchar = counter_delta(
        io_count(io_before.as_deref(), "wchar:"),
        io_count(io.as_deref(), "wchar:"),
    );
    let status = fs::read_to_string("/proc/self/status").ok();
    // Peak memory covers the process, including setup and generation.
    let peak_rss_mb = peak_memory_mb(status.as_deref());
    // The index with its full-text files, which live in `search-*` beside it.
    let full_text: u64 = fs::read_dir(&root)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("search-"))
        .map(|entry| dir_size(&entry.path()))
        .sum();
    let index = dir_size(&archive.index_db()) + dir_size(&root.join("index.db-wal")) + full_text;
    let stored = dir_size(&archive.archive_db()) + dir_size(&archive.blobs_dir());
    println!(
        "{}",
        serde_json::json!({
            "workers": workers,
            "inputs": pass.inputs_changed,
            "input_mb": input as f64 / 1e6,
            "seconds": secs,
            "setup_seconds": setup_seconds,
            "ingest_seconds": ingest_seconds,
            "engine_seconds": engine_seconds,
            "checkpoint_seconds": checkpoint_seconds,
            "databases": databases,
            "cpu_ticks": cpu_ticks,
            "mb_per_s": input as f64 / 1e6 / secs,
            "messages_written": pass.messages_written,
            "bytes_read": pass.bytes_read,
            "bytes_stored": pass.bytes_stored,
            "unreadable_root_count": pass.unreadable_roots.len(),
            "index_mb": index as f64 / 1e6,
            "index_over_input": index as f64 / input.max(1) as f64,
            "stored_mb": stored as f64 / 1e6,
            "disk_read_mb": read_bytes.map(|n|n as f64 / 1e6),
            "disk_written_mb": write_bytes.map(|n|n as f64 / 1e6),
            "cancelled_write_mb": cancelled_write_bytes.map(|n|n as f64 / 1e6),
            "written_over_kept": write_bytes.map(|n|n as f64 / (index + stored).max(1) as f64),
            "written_mb": wchar.map(|n|n as f64 / 1e6),
            "written_over_input": wchar.map(|n|n as f64 / input.max(1) as f64),
            "peak_rss_mb": peak_rss_mb,
            "sqlite_memory_peak_bytes": polylogue_store::timing::sqlite_memory_peak(),
            // Seconds per stage; pool stages are thread-seconds.
            "stages": polylogue_store::timing::stage_seconds()
                .into_iter()
                .map(|(s, secs)| (s.name().to_owned(), serde_json::json!(secs)))
                .collect::<serde_json::Map<_, _>>(),
            "errors": pass.errors.len(),
            "first_errors": pass.errors.iter().take(5).collect::<Vec<_>>(),
        })
    );
    assert!(
        pass.errors.is_empty(),
        "ingestion errors: {:?}",
        pass.errors.iter().take(5).collect::<Vec<_>>()
    );
    assert!(
        pass.unreadable_roots.is_empty(),
        "source roots are unreadable: {:?}",
        pass.unreadable_roots.iter().take(5).collect::<Vec<_>>()
    );
    if let Some(max) = arg("--max-rss-mb").map(|m| m.parse::<u64>().expect("--max-rss-mb MB")) {
        let peak_rss_mb = peak_rss_mb.expect("peak memory measurement is unavailable");
        assert!(
            peak_rss_mb <= max,
            "peak resident memory {peak_rss_mb} MB exceeds {max} MB"
        );
    }
}

/// Prepare a stored corpus without paying for an initial parse before measuring a rebuild.
fn capture_only(home: &Path, root: &Path) -> polylogue_store::Result<()> {
    use polylogue_store::{Batch, Change, InputKey, Observation, Writer};
    use std::os::unix::ffi::OsStrExt as _;
    let (archive, _lock) = Archive::open_or_create(root)?;
    let mut writer = Writer::start(&archive)?;
    let (mut inputs, mut bytes) = (0_u64, 0_u64);
    let mut files = Vec::new();
    jsonl_files(&home.join(".claude/projects/-generated"), &mut files)?;
    jsonl_files(&home.join(".codex/sessions"), &mut files)?;
    for path in files {
        let prepared = archive.prepare_bytes(fs::File::open(&path)?)?;
        let length = prepared.len();
        let name = path.file_name().unwrap().to_string_lossy();
        // Inputs named as capture names them: a rollout by its thread.
        let (id, format) = match polylogue_sources::codex::File::from_path(&path) {
            Some(polylogue_sources::codex::File::Rollout {
                thread: Some(thread),
            }) => (format!("codex:rollout:{thread}"), "codex"),
            _ => (format!("generated:{name}"), "claude-code"),
        };
        let batch = Batch {
            input: InputKey {
                id,
                format: format.into(),
                route: polylogue_model::Route::Live,
            },
            change: Change::First,
            observation: Some(Observation {
                machine: "measurement-machine".into(),
                host: "measurement-host".into(),
                path: path.to_string_lossy().into_owned(),
                path_bytes: path.as_os_str().as_bytes().to_vec(),
                facts: Some(polylogue_store::FileFacts::of(&fs::metadata(&path)?)),
                seen_at_ms: 1_788_256_800_000,
                import: None,
                container: None,
                database: None,
            }),
            continues: false,
            new_blob: None,
            blobs: Vec::new(),
            length,
            parse: Vec::new(),
            release: Default::default(),
            capture_read: None,
        };
        writer.store_bytes(prepared, batch)?;
        inputs += 1;
        bytes += length;
    }
    writer.stop();
    println!(
        "{}",
        serde_json::json!({"stored_inputs":inputs, "stored_bytes":bytes})
    );
    Ok(())
}

/// The `.jsonl` files under `dir`, recursively, in name order; none when
/// `dir` is missing.
fn jsonl_files(dir: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Ok(());
    };
    let mut paths: Vec<PathBuf> = entries
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    paths.sort();
    for path in paths {
        if path.is_dir() {
            jsonl_files(&path, files)?;
        } else if path.extension().is_some_and(|e| e == "jsonl") {
            files.push(path);
        }
    }
    Ok(())
}

/// Compare levels on exactly the same groups, independently of parsing/chunk sizes.
fn compression_sizes(index: &Path) -> polylogue_store::Result<serde_json::Value> {
    let conn = polylogue_store::archive::open_read(index)?;
    let mut statement = conn.prepare("SELECT body FROM content_group ORDER BY id")?;
    let mut rows = statement.query([])?;
    let (mut groups, mut stored, mut raw, mut three, mut nine) =
        (0_u64, 0_u64, 0_u64, 0_u64, 0_u64);
    while let Some(row) = rows.next()? {
        let body: Vec<u8> = row.get(0)?;
        let decoded = zstd::stream::decode_all(body.as_slice())?;
        stored += body.len() as u64;
        raw += decoded.len() as u64;
        three += zstd::bulk::compress(&decoded, 3)?.len() as u64;
        nine += zstd::bulk::compress(&decoded, 9)?.len() as u64;
        groups += 1;
    }
    Ok(
        serde_json::json!({"groups":groups,"stored_bytes":stored,"raw_bytes":raw,"level3_bytes":three,"level9_bytes":nine,"level3_over9":three as f64/nine.max(1) as f64}),
    )
}

fn io_count(contents: Option<&str>, field: &str) -> Option<u64> {
    contents?
        .lines()
        .find_map(|line| line.strip_prefix(field))?
        .trim()
        .parse()
        .ok()
}

fn cpu_ticks(contents: Option<&str>) -> Option<u64> {
    // A comm name can itself contain spaces and parentheses.
    let (_, fields) = contents?.rsplit_once(')')?;
    let mut fields = fields.split_whitespace().skip(11);
    let user: u64 = fields.next()?.parse().ok()?;
    let system: u64 = fields.next()?.parse().ok()?;
    user.checked_add(system)
}

fn peak_memory_mb(contents: Option<&str>) -> Option<u64> {
    let value = contents?
        .lines()
        .find_map(|line| line.strip_prefix("VmHWM:"))?
        .trim()
        .strip_suffix("kB")?
        .trim();
    value.parse::<u64>().ok().map(|kb| kb / 1024)
}

fn counter_delta(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    after?.checked_sub(before?)
}

#[cfg(test)]
#[path = "first_build/tests.rs"]
mod tests;

fn measurement_config(home: std::path::PathBuf) -> Config {
    Config::new(
        home,
        Default::default(),
        "measurement-machine".into(),
        "measurement-host".into(),
        std::thread::available_parallelism()
            .map(usize::from)
            .expect("CPU allowance"),
        polylogue_model::WallClock::new(jiff::Timestamp::now),
    )
    .expect("measurement configuration")
}

/// One completed build's database after its committed WAL was copied home.
#[derive(Debug, serde::Serialize)]
struct BuildDatabase {
    name: &'static str,
    wal_before: u64,
    wal_after: u64,
    bytes: u64,
    page_size: u64,
    pages: u64,
}

/// Count a benchmark's pending checkpoint work after its engine has stopped.
/// A live reader is a failed measurement, rather than unpaid work omitted
/// from one arm. This does not change the daemon's shutdown policy.
/// The build's private file is installed as `index.db`, in WAL, when the
/// pass ends its first build; a pass that left it unfinished is no
/// measurement of a first build.
fn checkpoint_build(archive: &Archive) -> polylogue_store::Result<Vec<BuildDatabase>> {
    if let Some(build) = archive.index_build() {
        return Err(polylogue_store::StoreError::internal(format!(
            "the pass ended before installing the built index: {build}"
        )));
    }
    let mut databases = Vec::new();
    for (name, path) in [
        ("archive", archive.archive_db()),
        ("index", archive.index_db()),
    ] {
        let wal = path.with_extension("db-wal");
        let wal_bytes = || match std::fs::metadata(&wal) {
            Ok(metadata) => Ok(metadata.len()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(0),
            Err(error) => Err(error),
        };
        let before = wal_bytes()?;
        let conn =
            polylogue_store::archive::open_write(&path, polylogue_store::archive::cache::SETUP)?;
        conn.set_db_config(
            rusqlite::config::DbConfig::SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE,
            true,
        )?;
        conn.busy_timeout(std::time::Duration::ZERO)?;
        let busy: i64 = conn.query_row("PRAGMA main.wal_checkpoint(TRUNCATE)", [], |r| r.get(0))?;
        if busy != 0 {
            return Err(polylogue_store::StoreError::internal(format!(
                "completed build still has a reader of {name}"
            )));
        }
        let page_size = conn.query_row("PRAGMA main.page_size", [], |r| r.get(0))?;
        let pages = conn.query_row("PRAGMA main.page_count", [], |r| r.get(0))?;
        drop(conn);
        let after = wal_bytes()?;
        if after != 0 {
            return Err(polylogue_store::StoreError::internal(format!(
                "completed build left {after} WAL bytes in {name}"
            )));
        }
        databases.push(BuildDatabase {
            name,
            wal_before: before,
            wal_after: after,
            bytes: std::fs::metadata(path)?.len(),
            page_size,
            pages,
        });
    }
    Ok(databases)
}
