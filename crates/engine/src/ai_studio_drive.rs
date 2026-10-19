//! The AI Studio collector (architecture 5.1): reads the prompts AI Studio
//! saves in the operator's Drive and writes them into the spool the
//! `ai-studio` origin watches. It runs only when the setting
//! `drive.ai-studio` is `on`, with the OAuth token the operator authorized
//! (`$XDG_STATE_HOME/polylogue/token.json`, Google's authorized-user format).
//!
//! Every Drive revision of a prompt is kept: each sync writes at most one
//! revision per prompt, the oldest not yet written, to `<fileId>.json`, so
//! the loop stores it as a revision of that one input before the next is
//! written over it. `files.json` holds the folder listing, which names the
//! prompts.

use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const SETTING: &str = "drive.ai-studio";
const FOLDER: &str = "Google AI Studio";
const FOLDER_MIME: &str = "application/vnd.google-apps.folder";
const STATE: &str = ".revisions.json";

#[derive(Debug, thiserror::Error)]
pub enum DriveError {
    #[error("Drive token {0}: {1}")]
    Token(PathBuf, String),
    #[error("Drive request {0}: {1}")]
    Http(String, String),
    #[error("no Drive folder named {FOLDER:?}")]
    NoFolder,
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Where the collector talks to. The defaults are Google's; tests point
/// both at a local server.
#[derive(Debug, Clone)]
pub struct Endpoints {
    pub api: String,
    pub token: String,
}

impl Default for Endpoints {
    fn default() -> Self {
        Self {
            api: "https://www.googleapis.com/drive/v3".into(),
            token: "https://oauth2.googleapis.com/token".into(),
        }
    }
}

pub struct Collector {
    pub endpoints: Endpoints,
    pub token_path: PathBuf,
    pub spool: PathBuf,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SyncReport {
    pub prompts: usize,
    pub revisions_written: usize,
    /// Revisions still to write: another pass should follow.
    pub pending: usize,
}

#[derive(Deserialize)]
struct AuthorizedUser {
    client_id: String,
    client_secret: String,
    refresh_token: String,
}

#[derive(Debug, Clone, Deserialize)]
struct Revision {
    id: String,
    #[serde(rename = "modifiedTime")]
    modified: String,
}

impl Collector {
    pub fn sync(&self) -> Result<SyncReport, DriveError> {
        let token = self.access_token()?;
        let agent = ureq::Agent::new_with_defaults();
        let get = |url: &str, query: &[(&str, &str)]| -> Result<Vec<u8>, DriveError> {
            let mut req = agent
                .get(url)
                .header("Authorization", &format!("Bearer {token}"));
            for (k, v) in query {
                req = req.query(*k, *v);
            }
            let mut resp = req
                .call()
                .map_err(|e| DriveError::Http(url.to_owned(), e.to_string()))?;
            resp.body_mut()
                .with_config()
                .limit(u64::MAX)
                .read_to_vec()
                .map_err(|e| DriveError::Http(url.to_owned(), e.to_string()))
        };
        let json = |bytes: Vec<u8>, url: &str| -> Result<serde_json::Value, DriveError> {
            serde_json::from_slice(&bytes)
                .map_err(|e| DriveError::Http(url.to_owned(), e.to_string()))
        };

        let files_url = format!("{}/files", self.endpoints.api);
        let folder_q =
            format!("name = '{FOLDER}' and mimeType = '{FOLDER_MIME}' and trashed = false");
        let found = json(
            get(&files_url, &[("q", &folder_q), ("fields", "files(id)")])?,
            &files_url,
        )?;
        let folder = found
            .pointer("/files/0/id")
            .and_then(serde_json::Value::as_str)
            .ok_or(DriveError::NoFolder)?
            .to_owned();

        let mut files = Vec::new();
        let mut page: Option<String> = None;
        let list_q = format!("'{folder}' in parents and trashed = false");
        loop {
            let mut query = vec![
                ("q", list_q.as_str()),
                (
                    "fields",
                    "nextPageToken,files(id,name,mimeType,createdTime,modifiedTime,headRevisionId)",
                ),
                ("pageSize", "1000"),
            ];
            if let Some(p) = &page {
                query.push(("pageToken", p.as_str()));
            }
            let v = json(get(&files_url, &query)?, &files_url)?;
            files.extend(
                v.get("files")
                    .and_then(serde_json::Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
            );
            match v.get("nextPageToken").and_then(serde_json::Value::as_str) {
                Some(p) => page = Some(p.to_owned()),
                None => break,
            }
        }

        std::fs::create_dir_all(&self.spool)?;
        let listing = serde_json::to_vec_pretty(&serde_json::json!({ "files": files }))
            .expect("JSON value serializes");
        write_if_changed(
            &self.spool.join(polylogue_sources::ai_studio::LISTING),
            &listing,
        )?;

        let mut state = self.load_state();
        let mut report = SyncReport {
            prompts: files.len(),
            ..SyncReport::default()
        };
        for f in &files {
            let Some(id) = f.get("id").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let head = f.get("headRevisionId").and_then(serde_json::Value::as_str);
            if head.is_some() && state.get(id).map(String::as_str) == head {
                continue;
            }
            let rev_url = format!("{files_url}/{id}/revisions");
            let v = json(
                get(&rev_url, &[("fields", "revisions(id,modifiedTime)")])?,
                &rev_url,
            )?;
            let mut revisions: Vec<Revision> = v
                .get("revisions")
                .cloned()
                .map(serde_json::from_value)
                .transpose()
                .map_err(|e| DriveError::Http(rev_url.clone(), e.to_string()))?
                .unwrap_or_default();
            revisions.sort_by(|a, b| a.modified.cmp(&b.modified));
            let done = state.get(id);
            let start = done
                .and_then(|d| revisions.iter().position(|r| &r.id == d))
                .map_or(0, |i| i + 1);
            let Some(next) = revisions.get(start) else {
                continue;
            };
            let url = format!("{rev_url}/{}", next.id);
            let bytes = get(&url, &[("alt", "media")])?;
            write_atomic(&self.spool.join(format!("{id}.json")), &bytes)?;
            state.insert(id.to_owned(), next.id.clone());
            report.revisions_written += 1;
            report.pending += revisions.len() - start - 1;
        }
        self.save_state(&state)?;
        Ok(report)
    }

    fn access_token(&self) -> Result<String, DriveError> {
        let err = |m: String| DriveError::Token(self.token_path.clone(), m);
        let raw = std::fs::read(&self.token_path).map_err(|e| err(e.to_string()))?;
        let user: AuthorizedUser = serde_json::from_slice(&raw).map_err(|e| err(e.to_string()))?;
        let mut resp = ureq::post(&self.endpoints.token)
            .send_form([
                ("client_id", user.client_id.as_str()),
                ("client_secret", user.client_secret.as_str()),
                ("refresh_token", user.refresh_token.as_str()),
                ("grant_type", "refresh_token"),
            ])
            .map_err(|e| err(format!("refresh failed: {e}")))?;
        let v: serde_json::Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| err(format!("refresh reply: {e}")))?;
        v.get("access_token")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| err("refresh reply has no access_token".into()))
    }

    fn load_state(&self) -> BTreeMap<String, String> {
        std::fs::read(self.spool.join(STATE))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default()
    }

    fn save_state(&self, state: &BTreeMap<String, String>) -> Result<(), DriveError> {
        let bytes = serde_json::to_vec_pretty(state).expect("map serializes");
        write_if_changed(&self.spool.join(STATE), &bytes)
    }
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<(), DriveError> {
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    write_atomic(path, bytes)
}

/// Written whole or not at all, so the loop never stores half a revision.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), DriveError> {
    let tmp = path.with_extension("json.part");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests;
