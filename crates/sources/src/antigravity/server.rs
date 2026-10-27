//! Imports the running vendor server's Markdown without decoding trajectories.
use polylogue_model::{BlobHash, Json};
use std::fs;
use std::io::Write;
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
                .timeout_global(Some(Duration::from_secs(120)))
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

pub fn sync(home: &Path, spool: &Path) -> Vec<String> {
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

fn sync_paths(paths: &[PathBuf], spool: &Path, servers: &[Server]) -> Vec<String> {
    let mut errors = Vec::new();
    let pending: Vec<_> = paths
        .iter()
        .filter(|path| !converted(path, spool))
        .cloned()
        .collect();
    if pending.is_empty() {
        return errors;
    }
    let mut counts = std::collections::HashMap::new();
    for path in paths {
        *counts.entry(path.file_stem()).or_insert(0) += 1;
    }
    let pending: Vec<_> = pending
        .into_iter()
        .filter(|path| {
            if counts[&path.file_stem()] > 1 {
                errors.push(format!(
                    "Antigravity trajectory {} not converted: duplicate conversation identity",
                    path.display()
                ));
                false
            } else {
                true
            }
        })
        .collect();
    if pending.is_empty() {
        return errors;
    }
    let paths = &pending;
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
    let Some((server, listing)) = selected else {
        errors.extend(paths.iter().map(|p| {
            format!(
                "Antigravity trajectory {} not converted: no reachable loopback language server",
                p.display()
            )
        }));
        return errors;
    };
    for path in paths {
        if let Err(e) = convert(server, &listing, path, spool) {
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

fn converted(path: &Path, spool: &Path) -> bool {
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    fs::read(destination(path, spool))
        .ok()
        .and_then(|b| serde_json::from_slice::<Json>(&b).ok())
        .is_some_and(|v| v["trajectoryHash"] == BlobHash::of(&bytes).to_string())
}

fn convert(server: &Server, listing: &Json, path: &Path, spool: &Path) -> Result<(), String> {
    let id = path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid trajectory name")?;
    let read = || fs::read(path).map_err(|e| e.to_string());
    let hash = BlobHash::of(&read()?).to_string();
    let destination = destination(path, spool);
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
    if hash != BlobHash::of(&read()?).to_string() {
        return Err("trajectory changed during conversion".into());
    }
    let mut export = serde_json::json!({"source":"antigravity_language_server","cascadeId":id,"markdown":markdown,"trajectoryHash":hash});
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
    fs::create_dir_all(spool).map_err(|e| e.to_string())?;
    let bytes = serde_json::to_vec(&export).map_err(|e| e.to_string())?;
    atomicwrites::AtomicFile::new(destination, atomicwrites::AllowOverwrite)
        .write(|f| f.write_all(&bytes))
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
