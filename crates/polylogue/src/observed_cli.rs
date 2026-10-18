//! `daemon --config`: run the daemon and collect from one configured source.
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
    claude_projects: Option<polylogue_store::configured_claude::Config>,
    codex_home: Option<polylogue_store::configured_codex::Config>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Source {
    path: PathBuf,
    machine: String,
    host: String,
    #[serde(default = "default_quantum")]
    acquisition_quantum_bytes: u64,
}
fn default_quantum() -> u64 {
    8 * 1024 * 1024
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
                    "usage: daemon --config FILE [--until-idle] [--json]",
                ));
            }
        }
        i += 1;
    }
    let config = config.ok_or_else(|| {
        error(
            ErrorKind::NotYetAvailable,
            "usage: daemon --config FILE [--until-idle] [--json]",
        )
    })?;
    let mut bytes = Vec::new();
    fs::File::open(config)
        .map_err(io_error)?
        .take(65537)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() > 65536 {
        return Err(error(
            ErrorKind::InvalidQuery,
            "configuration file is larger than 65536 bytes",
        ));
    }
    let config: Config = serde_json::from_slice(&bytes)
        .map_err(|e| error(ErrorKind::InvalidQuery, format!("configuration file: {e}")))?;
    if usize::from(config.claude_main.is_some())
        + usize::from(config.codex_home.is_some())
        + usize::from(config.claude_projects.is_some())
        != 1
    {
        return Err(error(
            ErrorKind::InvalidQuery,
            "configure exactly one of claude_main, claude_projects or codex_home",
        ));
    }
    if let Some(projects) = &config.claude_projects {
        projects.validate().map_err(io_error)?;
        if until_idle {
            return Err(error(
                ErrorKind::NotYetAvailable,
                "--until-idle is not supported for claude_projects; run the daemon without it",
            ));
        }
    }
    let runtime = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .ok_or_else(|| {
            error(
                ErrorKind::InvalidQuery,
                "XDG_RUNTIME_DIR must be set to an absolute path",
            )
        })?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(io_error)?;
    let mut random = [0; 16];
    fs::File::open("/dev/urandom")
        .map_err(io_error)?
        .read_exact(&mut random)
        .map_err(io_error)?;
    let id = import_cli::uuid_v7(u64::try_from(now.as_millis()).map_err(io_error)?, random)?;
    let stop = crate::stop_on_ctrl_c()?;
    if let Some(projects) = config.claude_projects {
        let mut running =
            polylogue::Running::start_claude_projects(archive, &runtime, &projects, &id, &stop)
                .map_err(io_error)?;
        eprintln!(
            "listening: {} (Claude projects directory)",
            running.socket().display()
        );
        if json_output {
            serde_json::to_writer(&mut *out,&json!({"outcome":"degraded","acquisition":"configured-claude-projects-v1","scope":"discovery_pending"})).map_err(io_error)?;
            writeln!(out).map_err(io_error)?;
            out.flush().map_err(io_error)?;
        }
        while !stop.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(10));
        }
        running.stop();
        return Ok(0);
    }
    if let Some(codex) = config.codex_home {
        return run_codex(
            archive,
            &runtime,
            &codex,
            &id,
            i64::try_from(now.as_micros()).map_err(io_error)?,
            &stop,
            until_idle,
            json_output,
            out,
        );
    }
    let claude = config.claude_main.expect("validated source selection");
    if !(65536..=8 * 1024 * 1024).contains(&claude.acquisition_quantum_bytes) {
        return Err(error(
            ErrorKind::InvalidQuery,
            "acquisition_quantum_bytes must be between 65536 and 8388608",
        ));
    }
    let start = |archive: &Path,
                 runtime: &Path,
                 source: &Path,
                 machine: &str,
                 host: &str,
                 id: &str,
                 seen_at: i64,
                 stop: &AtomicBool| {
        if until_idle {
            polylogue::Running::start_observed(
                archive, runtime, source, machine, host, id, seen_at, stop,
            )
        } else {
            polylogue::Running::start_observed_continuing(
                archive,
                runtime,
                source,
                machine,
                host,
                id,
                seen_at,
                stop,
                claude.acquisition_quantum_bytes,
            )
        }
    };
    let (mut running, source) = start(
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
        "listening: {} (Claude session file)",
        running.socket().display()
    );
    if json_output && !until_idle {
        serde_json::to_writer(&mut *out,&json!({"outcome":"ok","acquisition":"observed-claude-location-v1","scope":"retention_only","input":{"id":source.input.id,"chain":source.input.chain,"revision":source.revision,"length":source.input.length}})).map_err(io_error)?;
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
                    Some("complete" | "unavailable")
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
                    "source was not parsed within 120 seconds",
                ));
            }
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    running.stop();
    if let Some(progress) = final_progress {
        let ok = progress["source_rows"]["state"] == "complete";
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
                reason: "interrupted".into(),
            },
            "interrupted before the source was fully parsed",
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
    json_output: bool,
    out: &mut impl Write,
) -> Result<i32, Failure> {
    let (mut running, acquisition) =
        polylogue::Running::start_codex(archive, runtime, config, id, seen_at, stop, !until_idle)
            .map_err(io_error)?;
    eprintln!("listening: {} (Codex home)", running.socket().display());
    if !until_idle {
        if json_output {
            serde_json::to_writer(&mut *out,&json!({"outcome":"degraded","acquisition":acquisition,"scope":"retention_only","missing":["general_session_coverage"]})).map_err(io_error)?;
            writeln!(out).map_err(io_error)?;
            out.flush().map_err(io_error)?;
        }
        while !stop.load(Ordering::Acquire) {
            std::thread::sleep(Duration::from_millis(10));
        }
        running.stop();
        return Ok(0);
    }
    let deadline = Instant::now() + Duration::from_secs(120);
    let mut admitted = acquisition.admitted.clone();
    if config.paginated_history.is_some() {
        loop {
            if stop.load(Ordering::Acquire) {
                running.stop();
                return Err(error(
                    ErrorKind::Refused {
                        reason: "startup interrupted".into(),
                    },
                    "interrupted before the paginated history was stored",
                ));
            }
            let status = polylogue::request_value_until(
                running.socket(),
                &json!({"command":"status"}),
                deadline,
            )
            .map_err(io_error)?;
            let physical = &status["continuation"];
            if physical["paginated_pending"] == false && physical["paginated_state"] != "disabled" {
                if let Some(input) = physical["paginated_input"].as_str() {
                    if !admitted.iter().any(|old| old == input) {
                        admitted.push(input.to_owned());
                    }
                }
                break;
            }
            if Instant::now() >= deadline {
                running.stop();
                return Err(error(
                    ErrorKind::Timeout,
                    "paginated history was not stored within 120 seconds",
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let mut progress = Vec::new();
    for input in &admitted {
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
                    "interrupted before the startup snapshot was parsed",
                ));
            }
            let value = polylogue::request_value_until(
                running.socket(),
                &json!({"command":"progress","input":input}),
                deadline,
            )
            .map_err(io_error)?;
            // Prompt-history snapshots have no parser job to wait for.
            if value["source_rows"]["state"] == "complete"
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
                    "Codex files were not parsed within 120 seconds",
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
