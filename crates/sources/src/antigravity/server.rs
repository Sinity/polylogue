//! Imports the running vendor server's Markdown without decoding conversation files.
use polylogue_model::{BlobHash, Json};
use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const SERVICE: &str = "exa.language_server_pb.LanguageServerService";

struct Server {
    pid: u64,
    port: u16,
    token: String,
    agent: ureq::Agent,
}

impl Server {
    fn new(pid: u64, port: u16, token: String) -> Self {
        Self {
            pid,
            port,
            token,
            agent: ureq::Agent::config_builder()
                .proxy(None)
                .max_redirects(0)
                .timeout_connect(Some(Duration::from_secs(5)))
                .timeout_global(Some(Duration::from_secs(60)))
                .build()
                .into(),
        }
    }

    fn post(&self, method: &str, value: Json) -> Result<Json, ConversionError> {
        let body =
            serde_json::to_vec(&value).map_err(|e| ConversionError::NotConverted(e.to_string()))?;
        let mut response = self
            .agent
            .post(format!("http://127.0.0.1:{}/{SERVICE}/{method}", self.port))
            .header("content-type", "application/json")
            .header("x-codeium-csrf-token", &self.token)
            .send(body.as_slice())
            .map_err(|e| ConversionError::request(method, e))?;
        let bytes = response
            .body_mut()
            .with_config()
            .limit(u64::MAX)
            .read_to_vec()
            .map_err(|e| ConversionError::request(method, e))?;
        let value: Json = serde_json::from_slice(&bytes).map_err(|e| format!("{method}: {e}"))?;
        if !value.is_object() {
            return Err(format!("{method}: expected an object").into());
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
            Some(Server::new(pid, port, token))
        })
        .collect()
}

pub const COLLECTOR: crate::origin::Collector = crate::origin::Collector {
    setting: None,
    background: true,
    sync: |paths, stop| sync(paths.home, paths.spool, stop),
};

fn sync(home: &Path, spool: &Path, stop: &AtomicBool) -> Vec<String> {
    let root = home.join(".gemini/antigravity");
    let mut paths = Vec::new();
    if let Err(e) = conversation_files(&root.join("conversations"), &mut paths) {
        return vec![format!("Antigravity conversation files not converted: {e}")];
    }
    if paths.is_empty() {
        return Vec::new();
    }
    let servers = discover(&root, Path::new("/proc"));
    sync_paths(&paths, spool, &servers, stop)
}

fn conversation_files(dir: &Path, paths: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            conversation_files(&entry.path(), paths)?;
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

#[derive(Debug)]
enum ConversionError {
    ServerUnavailable(String),
    NotConverted(String),
}

impl ConversionError {
    fn request(method: &str, error: ureq::Error) -> Self {
        let text = format!("{method}: {error}");
        match error {
            ureq::Error::Io(_) | ureq::Error::Timeout(_) | ureq::Error::ConnectionFailed => {
                Self::ServerUnavailable(text)
            }
            _ => Self::NotConverted(text),
        }
    }

    fn text(&self) -> &str {
        match self {
            Self::ServerUnavailable(text) | Self::NotConverted(text) => text,
        }
    }
}

impl From<String> for ConversionError {
    fn from(text: String) -> Self {
        Self::NotConverted(text)
    }
}

impl From<&str> for ConversionError {
    fn from(text: &str) -> Self {
        Self::NotConverted(text.into())
    }
}

#[derive(Deserialize)]
struct ConversionState {
    #[serde(rename = "conversationFile")]
    file: FileState,
    status: String,
    server_pid: Option<u64>,
}

fn conversion_state(path: &Path, spool: &Path) -> Option<ConversionState> {
    fs::read(destination(path, spool).with_extension("conversion-state.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

fn failed_for(path: &Path, spool: &Path, state: FileState, pid: u64) -> bool {
    conversion_state(path, spool).is_some_and(|previous| {
        previous.file == state
            && previous.status == "not_converted"
            && previous.server_pid == Some(pid)
    })
}

fn sync_paths(
    paths: &[PathBuf],
    spool: &Path,
    servers: &[Server],
    stop: &AtomicBool,
) -> Vec<String> {
    let mut errors = Vec::new();
    let mut pending = Vec::new();
    let mut counts = std::collections::HashMap::new();
    for path in paths {
        *counts.entry(path.file_stem()).or_insert(0) += 1;
    }
    for path in paths {
        if stop.load(Ordering::Relaxed) {
            return errors;
        }
        let state = match FileState::read(path) {
            Ok(state) => state,
            Err(e) => {
                errors.push(format!(
                    "Antigravity conversation file {} not converted: {e}",
                    path.display()
                ));
                continue;
            }
        };
        if counts[&path.file_stem()] > 1 {
            errors.push(format!(
                "Antigravity conversation file {} not converted: duplicate conversation identity",
                path.display()
            ));
            continue;
        }
        if !converted(path, spool, state) {
            pending.push((path, state));
        }
    }
    if pending.is_empty() {
        return errors;
    }
    let mut selected = None;
    for server in servers {
        if stop.load(Ordering::Relaxed) {
            return errors;
        }
        if pending
            .iter()
            .all(|(path, state)| failed_for(path, spool, *state, server.pid))
        {
            continue;
        }
        match server.post(
            "SearchConversations",
            serde_json::json!({"query":"","limit":paths.len()}),
        ) {
            Ok(value) if value["results"].is_array() => {
                selected = Some((server, value));
                break;
            }
            Err(ConversionError::ServerUnavailable(error)) => {
                for (path, state) in pending {
                    record_status(
                        path,
                        spool,
                        state,
                        Some(server.pid),
                        "server_unavailable",
                        Some(&error),
                        &mut errors,
                    );
                }
                return errors;
            }
            _ => continue,
        }
    }
    let Some((server, listing)) = selected else {
        // A refusal remains useful while the same server process is present.
        for (path, state) in pending {
            if !servers
                .iter()
                .any(|server| failed_for(path, spool, state, server.pid))
            {
                record_status(
                    path,
                    spool,
                    state,
                    None,
                    "server_unavailable",
                    None,
                    &mut errors,
                );
            }
        }
        return errors;
    };
    let mut unavailable: Option<String> = None;
    for (path, state) in pending {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if let Some(error) = &unavailable {
            record_status(
                path,
                spool,
                state,
                Some(server.pid),
                "server_unavailable",
                Some(error),
                &mut errors,
            );
            continue;
        }
        if failed_for(path, spool, state, server.pid) {
            continue;
        }
        let result = convert(server, &listing, path, spool, state);
        let status = match &result {
            Ok(()) => "converted",
            Err(ConversionError::ServerUnavailable(_)) => "server_unavailable",
            Err(ConversionError::NotConverted(_)) => "not_converted",
        };
        record_status(
            path,
            spool,
            state,
            Some(server.pid),
            status,
            result.as_ref().err().map(|e| e.text()),
            &mut errors,
        );
        if let Err(error) = result {
            errors.push(format!(
                "Antigravity conversation file {} not converted: {}",
                path.display(),
                error.text()
            ));
            if let ConversionError::ServerUnavailable(error) = error {
                unavailable = Some(error);
            }
        }
    }
    errors
}

fn record_status(
    path: &Path,
    spool: &Path,
    state: FileState,
    pid: Option<u64>,
    status: &str,
    error: Option<&str>,
    errors: &mut Vec<String>,
) {
    if let Err(e) = write_conversion_state(path, spool, state, pid, status, error) {
        errors.push(format!(
            "Antigravity conversion state {}: {e}",
            path.display()
        ));
    }
}

fn destination(path: &Path, spool: &Path) -> PathBuf {
    spool.join(format!(
        "{}.json",
        BlobHash::of(path.as_os_str().as_encoded_bytes())
    ))
}

fn converted(path: &Path, spool: &Path, state: FileState) -> bool {
    // Conversion state stays small even when the Markdown export is large.
    destination(path, spool).exists()
        && conversion_state(path, spool)
            .is_some_and(|previous| previous.file == state && previous.status == "converted")
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if fs::read(path).ok().as_deref() == Some(bytes.as_slice()) {
        return Ok(());
    }
    fs::create_dir_all(path.parent().ok_or("missing spool directory")?)
        .map_err(|e| e.to_string())?;
    polylogue_model::atomic_write::write_atomic(path, &bytes, None).map_err(|e| e.to_string())
}

fn write_conversion_state(
    path: &Path,
    spool: &Path,
    state: FileState,
    pid: Option<u64>,
    status: &str,
    error: Option<&str>,
) -> Result<(), String> {
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid conversation file name")?;
    let value = serde_json::json!({
        "source":"antigravity_conversion", "cascadeId":id,
        "conversationFile":state, "server_pid":pid, "status":status, "error":error,
    });
    write_json(
        &destination(path, spool).with_extension("conversion-state.json"),
        &value,
    )
}

fn convert(
    server: &Server,
    listing: &Json,
    path: &Path,
    spool: &Path,
    state: FileState,
) -> Result<(), ConversionError> {
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid conversation file name")?;
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
        return Err("conversation file changed during conversion".into());
    }
    let mut export = serde_json::json!({"source":"antigravity_language_server","cascadeId":id,"markdown":markdown,"conversationFile":state});
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
    // The conversion state is written only after the export, so an interrupted write retries.
    write_json(&destination(path, spool), &export)?;
    Ok(())
}

#[cfg(test)]
mod tests;
