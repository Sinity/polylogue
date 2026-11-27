//! The last pages of real sessions: a sample run's archive served by the
//! daemon, read through the CLI in pages. Each session is walked
//! to its end in large pages to find the cursors of its last two small
//! pages; the daemon restarts, then those pages are timed on the first
//! request (cold) and three times more (warm).

use super::*;
use std::io::{BufRead, BufReader};
use std::process::{Child, Stdio};
use std::time::Instant;

/// Tokens in a timed page: eight times the 4,000-token page a model reads.
const PAGE: &str = "32768";
/// Tokens per page while walking to the end.
const WALK: &str = "4000000";

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// The sample's archive served in a sandbox: the daemon reads no files
/// outside the sample.
struct Reader {
    binary: PathBuf,
    archive: PathBuf,
    sandbox: polylogue_testkit::Sandbox,
}

impl Reader {
    fn command(&self) -> Command {
        let mut command = Command::new(&self.binary);
        self.sandbox
            .apply(&mut command)
            .arg("--archive")
            .arg(&self.archive);
        command
    }

    fn start(&self) -> Result<Daemon, String> {
        let mut daemon = Daemon(
            polylogue_testkit::spawn_daemon(self.command().arg("daemon").stderr(Stdio::piped()))
                .map_err(|e| e.to_string())?,
        );
        let mut line = String::new();
        let mut stderr = BufReader::new(daemon.0.stderr.take().ok_or("no daemon stderr")?);
        stderr.read_line(&mut line).map_err(|e| e.to_string())?;
        if !line.contains("serving") {
            return Err(format!("the daemon did not start: {line}"));
        }
        // Keep draining the pipe: a daemon that logs into a full pipe would
        // block and stall the timed requests.
        std::thread::spawn(move || {
            let _ = std::io::copy(&mut stderr, &mut std::io::sink());
        });
        Ok(daemon)
    }

    fn stop(&self, mut daemon: Daemon) -> Result<(), String> {
        let stopped = Command::new("kill")
            .arg("-TERM")
            .arg(daemon.0.id().to_string())
            .status()
            .map_err(|e| e.to_string())?;
        if !stopped.success() {
            return Err(format!("stopping the benchmark daemon failed: {stopped}"));
        }
        let status = daemon.0.wait().map_err(|e| e.to_string())?;
        if !status.success() {
            return Err(format!("the benchmark daemon exited with {status}"));
        }
        Ok(())
    }

    /// One CLI request's JSON answer and how long it took.
    fn request(&self, args: &[&str]) -> Result<(Value, f64), String> {
        let began = Instant::now();
        let out = self
            .command()
            .arg("--json")
            .args(args)
            .output()
            .map_err(|e| e.to_string())?;
        let took = began.elapsed().as_secs_f64();
        if !out.status.success() {
            return Err(format!(
                "{args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            ));
        }
        let envelope: Value = serde_json::from_slice(&out.stdout).map_err(|e| e.to_string())?;
        let answer = envelope
            .get("value")
            .cloned()
            .ok_or_else(|| format!("CLI returned no value: {envelope}"))?;
        Ok((answer, took))
    }

    fn page(
        &self,
        session: &str,
        budget: &str,
        cursor: Option<&str>,
    ) -> Result<(Value, f64), String> {
        let mut args = vec![session, "read", "--budget", budget];
        if let Some(cursor) = cursor {
            args.extend(["--cursor", cursor]);
        }
        let (answer, took) = self.request(&args)?;
        let page = answer
            .pointer("/reply/steps/0/outcome/reply/reply")
            .cloned()
            .ok_or_else(|| format!("read returned no page: {answer}"))?;
        Ok((page, took))
    }

    fn idle(&self) -> Result<f64, String> {
        loop {
            let (status, took) = self.request(&["status"])?;
            if status["busy"] != true {
                return Ok(took);
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }
}

/// The rows of a query listing.
fn rows(answer: &Value) -> Result<&[Value], String> {
    answer
        .get("rows")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("query returned no rows: {answer}"))
}

fn next_cursor(page: &Value) -> Result<Option<String>, String> {
    let footer = page
        .pointer("/document/footer")
        .ok_or("read page has no footer")?;
    if footer["next"].is_null() {
        return Ok(None);
    }
    footer
        .pointer("/next/cursor")
        .and_then(Value::as_str)
        .map(|cursor| Some(cursor.to_owned()))
        .ok_or_else(|| format!("read continuation has no cursor: {footer}"))
}

/// The cursors of a session's last two small pages: from the start of its
/// last large page, read on in small pages to the end.
fn tail_cursors(reader: &Reader, session: &str) -> Result<(Vec<Option<String>>, usize), String> {
    let mut cursor: Option<String> = None;
    let mut last_large;
    let mut pages = 0;
    loop {
        let (page, _) = reader.page(session, WALK, cursor.as_deref())?;
        pages += 1;
        last_large = cursor.clone();
        cursor = next_cursor(&page)?;
        if cursor.is_none() {
            break;
        }
    }
    let mut cursor = last_large;
    let mut small = Vec::new();
    loop {
        let (page, _) = reader.page(session, PAGE, cursor.as_deref())?;
        small.push(cursor.clone());
        cursor = next_cursor(&page)?;
        if cursor.is_none() {
            break;
        }
    }
    let tail = small.split_off(small.len().saturating_sub(2));
    Ok((tail, pages))
}

pub(super) fn run(
    drawn: &Drawn,
    largest: usize,
    named: &[String],
    binary: Option<PathBuf>,
) -> Result<(), String> {
    let archive = drawn.dir().join("out").join("archive");
    if !archive.join("index.db").exists() {
        return Err(format!(
            "{} holds no archive; run `xtask bench run --name {}` first",
            archive.display(),
            drawn.name
        ));
    }
    let binary = match binary {
        Some(binary) => binary,
        None => polylogue_binary()?,
    };
    let scratch = drawn.dir().join("tails");
    if scratch.exists() {
        fs::remove_dir_all(&scratch).map_err(|e| e.to_string())?;
    }
    let sandbox = polylogue_testkit::Sandbox::new(&scratch).map_err(|e| e.to_string())?;
    sandbox.create_directories().map_err(|e| e.to_string())?;
    let reader = Reader {
        binary,
        archive,
        sandbox,
    };
    let load = bench::LoadMonitor::start()?;
    let daemon = reader.start()?;
    let status_seconds = reader.idle()?;
    let (answer, _) = reader.request(&[&format!(
        "sessions | sort by messages desc | limit {largest}"
    )])?;
    let mut sessions: Vec<String> = rows(&answer["reply"])?
        .iter()
        .filter_map(|row| row["id"].as_str().map(|id| format!("session:{id}")))
        .collect();
    sessions.extend(named.iter().cloned());
    let mut tails = Vec::new();
    for session in &sessions {
        let began = Instant::now();
        let (tail, pages) = tail_cursors(&reader, session)?;
        tails.push((session.clone(), tail, pages, began.elapsed().as_secs_f64()));
    }
    reader.stop(daemon)?;
    let mut measured = Vec::new();
    let mut slowest: f64 = 0.0;
    for (session, tail, pages, walk_seconds) in tails {
        let daemon = reader.start()?;
        let timed = || -> Result<Vec<f64>, String> {
            tail.iter()
                .map(|cursor| Ok(reader.page(&session, PAGE, cursor.as_deref())?.1))
                .collect()
        };
        let cold = timed()?;
        let mut warm = Vec::new();
        for _ in 0..3 {
            warm.extend(timed()?);
        }
        reader.stop(daemon)?;
        slowest = cold.iter().chain(&warm).fold(slowest, |a, b| a.max(*b));
        measured.push(json!({
            "session": session, "walk_pages": pages, "walk_seconds": walk_seconds,
            "cold_tail_page_seconds": cold, "warm_tail_page_seconds": warm,
        }));
    }
    let load = load.finish()?;
    let metrics = json!({
        "status_seconds": status_seconds,
        "slowest_tail_page_seconds": slowest,
        "sessions": measured,
    });
    let store = Store::open().map_err(|e| e.to_string())?;
    let checkout = checkout()?;
    let id = bench::record(
        &store,
        &checkout,
        "polylogue",
        "tails",
        &drawn.name,
        &reader.archive,
        &json!({"result": metrics, "load": load.to_json()}),
    )
    .map_err(|e| e.to_string())?;
    let _ = fs::remove_dir_all(&scratch);
    println!(
        "{}",
        serde_json::to_string_pretty(
            &json!({"run": id, "result": metrics, "load": load.to_json()})
        )
        .unwrap_or_default()
    );
    Ok(())
}

fn polylogue_binary() -> Result<PathBuf, String> {
    let ok = devkit::cargo_command()
        .args(["build", "--locked", "--release", "--bin", "polylogue"])
        .status()
        .map_err(|e| e.to_string())?
        .success();
    if !ok {
        return Err("building polylogue failed".into());
    }
    let target = env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target"));
    Ok(target.join("release/polylogue"))
}
