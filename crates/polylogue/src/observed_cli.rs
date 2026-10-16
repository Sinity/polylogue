//! Explicit finite startup acquisition. No watcher is registered.
use crate::{error, import_cli, io_error};
use polylogue_model::types::{ErrorKind, Failure};
use serde::Deserialize;
use serde_json::json;
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    claude_main: Option<Source>,
    codex_home: Option<polylogue_store::configured_codex::Config>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    path: PathBuf,
    machine: String,
    host: String,
}
pub(crate) fn run(args: &[String], archive: &Path, out: &mut impl Write) -> Result<i32, Failure> {
    let mut config = None;
    let mut until_idle = false;
    let mut json_output = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| error(ErrorKind::InvalidQuery, "--config requires a file"))?;
                if config.replace(value).is_some() {
                    return Err(error(ErrorKind::InvalidQuery, "duplicate --config"));
                }
            }
            "--until-idle" if !until_idle => until_idle = true,
            "--json" if !json_output => json_output = true,
            _ => {
                return Err(error(
                    ErrorKind::InvalidQuery,
                    "supported daemon options: --config FILE [--until-idle] [--json]",
                ));
            }
        }
        i += 1;
    }
    let config=config.ok_or_else(||error(ErrorKind::NotYetAvailable,"--until-idle requires one explicit --config source; general source collection is unavailable"))?;
    let mut bytes = Vec::new();
    fs::File::open(config)
        .map_err(io_error)?
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() > 65536 {
        return Err(error(
            ErrorKind::InvalidQuery,
            "source configuration exceeds 65536 bytes",
        ));
    }
    let config: Config = serde_json::from_slice(&bytes).map_err(|e| {
        error(
            ErrorKind::InvalidQuery,
            format!("source configuration: {e}"),
        )
    })?;
    if config.claude_main.is_some() == config.codex_home.is_some() {
        return Err(error(
            ErrorKind::InvalidQuery,
            "configure exactly one of claude_main or codex_home",
        ));
    }
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| error(ErrorKind::InvalidQuery, "absolute XDG_RUNTIME_DIR required"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io_error)?;
    let mut random = [0; 16];
    fs::File::open("/dev/urandom")
        .map_err(io_error)?
        .read_exact(&mut random)
        .map_err(io_error)?;
    let id = import_cli::uuid_v7(u64::try_from(now.as_millis()).map_err(io_error)?, random)?;
    let stop = Arc::new(AtomicBool::new(false));
    let signalled = Arc::clone(&stop);
    ctrlc::set_handler(move || signalled.store(true, Ordering::Release)).map_err(io_error)?;
    if let Some(codex) = config.codex_home {
        return run_codex(
            archive,
            &runtime,
            &codex,
            &id,
            i64::try_from(now.as_micros()).map_err(io_error)?,
            &stop,
            until_idle,
            out,
        );
    }
    let claude = config.claude_main.expect("validated source selection");
    let (mut running, source) = polylogue::Running::start_observed(
        archive,
        &runtime,
        &claude.path,
        &claude.machine,
        &claude.host,
        &id,
        i64::try_from(now.as_micros()).map_err(io_error)?,
        &stop,
    )
    .map_err(|e| match e {
        polylogue_store::Error::ObservedSourceRefused(reason) => error(
            ErrorKind::Refused {
                reason: reason.clone(),
            },
            reason,
        ),
        other => io_error(other),
    })?;
    eprintln!(
        "listening: {} (one observed Claude location; no watcher)",
        running.socket().display()
    );
    if json_output && !until_idle {
        serde_json::to_writer(&mut *out,&json!({"outcome":"ok","acquisition":"observed-claude-location-v1","scope":"retention_only","retained":{"id":source.input.id,"chain":source.input.chain,"revision":source.revision,"length":source.input.length}})).map_err(io_error)?;
        writeln!(out).map_err(io_error)?;
        out.flush().map_err(io_error)?;
    }
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut final_progress = None;
    while !stop.load(Ordering::Acquire) {
        if until_idle {
            let progress = polylogue::request_value_until(
                running.socket(),
                &json!({"command":"progress","input":source.input.id}),
                deadline,
            )
            .map_err(io_error)?;
            if progress["outcome"] == "error"
                || !progress["failure"].is_null()
                || !progress["recovery_error"].is_null()
                || matches!(
                    progress["source_rows"]["state"].as_str(),
                    Some("settled" | "unavailable")
                )
                || (progress["parser"]["status"] == "waiting_tail" && progress["active"] == false)
            {
                final_progress = Some(progress);
                break;
            }
            if Instant::now() >= deadline {
                running.stop();
                return Err(error(
                    ErrorKind::Timeout,
                    "observed source did not reach idle within 120 seconds",
                ));
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    running.stop();
    if let Some(progress) = final_progress {
        let ok = progress["source_rows"]["state"] == "settled";
        let result = json!({"outcome":if ok{"ok"}else{"degraded"},"acquisition":"observed-claude-location-v1","input":source.input.id,"chain":source.input.chain,"revision":source.revision,"length":source.input.length,"progress":progress});
        if json_output {
            serde_json::to_writer(&mut *out, &result).map_err(io_error)?;
            writeln!(out).map_err(io_error)?;
        } else {
            writeln!(out, "{}", result).map_err(io_error)?;
        }
        return Ok(if ok { 0 } else { 5 });
    }
    if until_idle {
        return Err(error(
            ErrorKind::Refused {
                reason: "interrupted before target settlement".into(),
            },
            "observed source interrupted before target settlement",
        ));
    }
    Ok(0)
}

fn run_codex(
    archive: &Path,
    runtime: &Path,
    config: &polylogue_store::configured_codex::Config,
    id: &str,
    seen_at: i64,
    stop: &AtomicBool,
    until_idle: bool,
    out: &mut impl Write,
) -> Result<i32, Failure> {
    let (mut running, acquisition) =
        polylogue::Running::start_codex(archive, runtime, config, id, seen_at, stop)
            .map_err(io_error)?;
    eprintln!(
        "listening: {} (finite configured Codex snapshots; no watcher)",
        running.socket().display()
    );
    if !until_idle {
        while !stop.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(10));
        }
        running.stop();
        return Ok(0);
    }
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut progress = Vec::new();
    for input in &acquisition.retained {
        if acquisition.prompt_input.as_ref() == Some(input) {
            continue;
        }
        loop {
            if stop.load(Ordering::Acquire) {
                running.stop();
                return Err(error(
                    ErrorKind::Refused {
                        reason: "startup interrupted".into(),
                    },
                    "startup interrupted before settlement",
                ));
            }
            let value = polylogue::request_value_until(
                running.socket(),
                &json!({"command":"progress","input":input}),
                deadline,
            )
            .map_err(io_error)?;
            // Prompt snapshots are retained pending references, not parser jobs.
            if value["source_rows"]["state"] == "settled"
                || value["source_rows"]["state"] == "unavailable"
                || !value["failure"].is_null()
                || !value["recovery_error"].is_null()
                || value["outcome"] == "error"
            {
                progress.push(value);
                break;
            }
            if Instant::now() >= deadline {
                running.stop();
                return Err(error(
                    ErrorKind::Timeout,
                    "configured Codex inputs did not reach idle within 120 seconds",
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let status =
        polylogue::request_value_until(running.socket(), &json!({"command":"status"}), deadline)
            .map_err(io_error)?;
    running.stop();
    serde_json::to_writer(&mut *out,&json!({"outcome":"degraded","acquisition":acquisition,"progress":progress,"status":status,"missing":["general_session_coverage"]})).map_err(io_error)?;
    writeln!(out).map_err(io_error)?;
    Ok(5)
}
