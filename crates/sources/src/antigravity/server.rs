//! Imports the running vendor server's Markdown without decoding trajectories.
use polylogue_model::{BlobHash, Json};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

const SERVICE: &str = "exa.language_server_pb.LanguageServerService";

struct Server {
    port: u16,
    token: String,
    agent: ureq::Agent,
}

impl Server {
    fn new(port: u16, token: String) -> Self {
        Self {
            port,
            token,
            agent: ureq::Agent::config_builder()
                .proxy(None)
                .max_redirects(0)
                .timeout_global(Some(Duration::from_secs(5)))
                .build()
                .into(),
        }
    }

    fn post(&self, method: &str, value: Json) -> Result<Json, String> {
        let body = serde_json::to_vec(&value).map_err(|e| e.to_string())?;
        let mut response = self
            .agent
            .post(format!("http://127.0.0.1:{}/{SERVICE}/{method}", self.port))
            .header("content-type", "application/json")
            .header("x-codeium-csrf-token", &self.token)
            .send(body.as_slice())
            .map_err(|e| format!("{method}: {e}"))?;
        let bytes = response
            .body_mut()
            .with_config()
            .limit(u64::MAX)
            .read_to_vec()
            .map_err(|e| format!("{method}: {e}"))?;
        let value: Json = serde_json::from_slice(&bytes).map_err(|e| format!("{method}: {e}"))?;
        if !value.is_object() {
            return Err(format!("{method}: expected an object"));
        }
        Ok(value)
    }
}

// Discovery identifies the running process; its command line supplies the
// CSRF token. Credentials never enter the spool or an error message.
fn discover(root: &Path, processes: &Path) -> Vec<Server> {
    let Ok(entries) = fs::read_dir(root.join("daemon")) else {
        return Vec::new();
    };
    let mut paths: Vec<_> = entries.filter_map(Result::ok).map(|e| e.path()).collect();
    paths.sort();
    paths
        .into_iter()
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            if !name.starts_with("ls_") || !name.ends_with(".json") {
                return None;
            }
            let value: Json = serde_json::from_slice(&fs::read(path).ok()?).ok()?;
            let port = u16::try_from(value["httpPort"].as_u64()?)
                .ok()
                .filter(|p| *p != 0)?;
            let pid = value["pid"].as_u64()?;
            let bytes = fs::read(processes.join(pid.to_string()).join("cmdline")).ok()?;
            let args: Vec<_> = bytes
                .split(|b| *b == 0)
                .filter_map(|b| std::str::from_utf8(b).ok())
                .collect();
            if !Path::new(args.first()?)
                .file_name()?
                .to_str()?
                .starts_with("language_server")
            {
                return None;
            }
            let token = args
                .iter()
                .enumerate()
                .find_map(|(i, arg)| {
                    arg.strip_prefix("-csrf_token=")
                        .or_else(|| arg.strip_prefix("--csrf_token="))
                        .or_else(|| {
                            matches!(*arg, "-csrf_token" | "--csrf_token")
                                .then(|| args.get(i + 1).copied())
                                .flatten()
                        })
                })?
                .to_owned();
            if token.is_empty() {
                return None;
            }
            Some(Server::new(port, token))
        })
        .collect()
}

pub const COLLECTOR: crate::origin::Collector = crate::origin::Collector {
    setting: None,
    background: true,
    sync: |paths| sync(paths.home, paths.spool),
};

fn sync(home: &Path, spool: &Path) -> Vec<String> {
    let root = home.join(".gemini/antigravity");
    let mut paths = Vec::new();
    if let Err(e) = trajectories(&root.join("conversations"), &mut paths) {
        return vec![format!("Antigravity trajectories not converted: {e}")];
    }
    if paths.is_empty() {
        return Vec::new();
    }
    let servers = discover(&root, Path::new("/proc"));
    sync_paths(&paths, spool, &servers)
}

fn trajectories(dir: &Path, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            trajectories(&entry.path(), paths)?;
        } else if entry.path().extension().is_some_and(|e| e == "pb") {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct FileState {
    size: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
}

impl FileState {
    fn read(path: &Path) -> Result<Self, String> {
        let metadata = fs::metadata(path).map_err(|e| e.to_string())?;
        Ok(Self {
            size: metadata.len(),
            modified_seconds: metadata.mtime(),
            modified_nanoseconds: metadata.mtime_nsec(),
        })
    }
}

fn sync_paths(paths: &[PathBuf], spool: &Path, servers: &[Server]) -> Vec<String> {
    let mut errors = Vec::new();
    let mut pending = Vec::new();
    let mut counts = std::collections::HashMap::new();
    for path in paths {
        *counts.entry(path.file_stem()).or_insert(0) += 1;
    }
    for path in paths {
        let state = match FileState::read(path) {
            Ok(state) => state,
            Err(e) => {
                errors.push(format!(
                    "Antigravity trajectory {} not converted: {e}",
                    path.display()
                ));
                continue;
            }
        };
        if counts[&path.file_stem()] > 1 {
            errors.push(format!(
                "Antigravity trajectory {} not converted: duplicate conversation identity",
                path.display()
            ));
            continue;
        }
        if converted(path, spool, state) {
            if let Err(e) = conversion_status(path, spool, state, "converted", None) {
                errors.push(format!(
                    "Antigravity conversion status {}: {e}",
                    path.display()
                ));
            }
        } else {
            pending.push((path, state));
        }
    }
    if pending.is_empty() {
        return errors;
    }
    let mut selected = None;
    for server in servers {
        match server.post(
            "SearchConversations",
            serde_json::json!({"query":"","limit":paths.len()}),
        ) {
            Ok(value) if value["results"].is_array() => {
                selected = Some((server, value));
                break;
            }
            _ => continue,
        }
    }
    for (path, state) in pending {
        let result = match &selected {
            Some((server, listing)) => convert(server, listing, path, spool, state),
            None => Ok(()),
        };
        let status = match (&selected, &result) {
            (None, _) => "server_unavailable",
            (Some(_), Ok(())) => "converted",
            (Some(_), Err(_)) => "not_converted",
        };
        if let Err(e) = conversion_status(path, spool, state, status, result.as_ref().err()) {
            errors.push(format!(
                "Antigravity conversion status {}: {e}",
                path.display()
            ));
        }
        if let Err(e) = result {
            errors.push(format!(
                "Antigravity trajectory {} not converted: {e}",
                path.display()
            ));
        }
    }
    errors
}

fn destination(path: &Path, spool: &Path) -> PathBuf {
    spool.join(format!(
        "{}.json",
        BlobHash::of(path.as_os_str().as_encoded_bytes())
    ))
}

fn converted(path: &Path, spool: &Path, state: FileState) -> bool {
    // Checkpoints stay small even when the Markdown export is large.
    destination(path, spool).exists()
        && fs::read(destination(path, spool).with_extension("checkpoint"))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<FileState>(&bytes).ok())
            == Some(state)
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if fs::read(path).ok().as_deref() == Some(bytes.as_slice()) {
        return Ok(());
    }
    fs::create_dir_all(path.parent().ok_or("missing spool directory")?)
        .map_err(|e| e.to_string())?;
    atomicwrites::AtomicFile::new(path, atomicwrites::AllowOverwrite)
        .write(|f| f.write_all(&bytes))
        .map_err(|e| e.to_string())
}

fn conversion_status(
    path: &Path,
    spool: &Path,
    state: FileState,
    status: &str,
    error: Option<&String>,
) -> Result<(), String> {
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid trajectory name")?;
    let value = serde_json::json!({
        "source":"antigravity_conversion", "cascadeId":id,
        "trajectoryFile":state, "status":status, "error":error,
    });
    write_json(
        &destination(path, spool).with_extension("status.json"),
        &value,
    )
}

fn convert(
    server: &Server,
    listing: &Json,
    path: &Path,
    spool: &Path,
    state: FileState,
) -> Result<(), String> {
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid trajectory name")?;
    let value = server.post(
        "ConvertTrajectoryToMarkdown",
        serde_json::json!({"conversationId":id}),
    )?;
    let markdown = value["markdown"].as_str().ok_or("missing Markdown")?;
    let mut in_section = false;
    let mut content = false;
    for line in markdown.lines() {
        if matches!(line.trim_end(), "### User Input" | "### Planner Response") {
            in_section = true;
        } else if in_section && !line.trim().is_empty() {
            content = true;
        }
    }
    if !content {
        return Err("empty or partial conversation export".into());
    }
    if state != FileState::read(path)? {
        return Err("trajectory changed during conversion".into());
    }
    let mut export = serde_json::json!({"source":"antigravity_language_server","cascadeId":id,"markdown":markdown,"trajectoryFile":state});
    if let Some(summary) = listing["results"]
        .as_array()
        .and_then(|rows| rows.iter().find(|v| v["cascadeId"] == id))
    {
        for key in ["title", "workspaceName", "snippet", "lastModifiedTime"] {
            if let Some(value) = summary.get(key) {
                export[key] = value.clone();
            }
        }
    }
    // Write the export before its checkpoint, so an interrupted write retries.
    let destination = destination(path, spool);
    write_json(&destination, &export)?;
    write_json(&destination.with_extension("checkpoint"), &state)
}

#[cfg(test)]
mod tests;
