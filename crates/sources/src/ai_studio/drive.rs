//! The AI Studio collector: reads the prompts AI Studio
//! saves in the operator's Drive and writes them into the spool the
//! `ai-studio` origin watches. It runs only with the archive's explicitly
//! authorized `drive-token.json`, in Google's authorized-user format.
//!
//! Each prompt has one immutable pending revision. Its bytes are published at
//! `<fileId>.json` until the engine acknowledges that the archive stored them. Only
//! then does `.revisions.json` advance and another revision replace that path.
//! Hidden pending bodies survive restart; their cleanup leaves the watched
//! prompt in place. `files.json` holds the folder listing.

use polylogue_model::JsonExt;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// One credential location shared by authorization and collection.
pub fn token_path(archive: &Path) -> PathBuf {
    archive.join("drive-token.json")
}
/// No archive credential has been published. Present but unreadable or invalid
/// credentials still reach the collector and report their actual error.
pub fn needs_authorization(archive: &Path) -> bool {
    std::fs::symlink_metadata(token_path(archive))
        .is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound)
}
pub const SCOPE: &str = "https://www.googleapis.com/auth/drive.readonly";
const FOLDER: &str = "Google AI Studio";
const FOLDER_MIME: &str = "application/vnd.google-apps.folder";
const STATE: &str = ".revisions.json";
const PENDING: &str = ".pending";

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

pub struct Drive {
    pub endpoints: Endpoints,
    pub token_path: PathBuf,
    pub spool: PathBuf,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SyncReport {
    pub prompts: usize,
    pub revisions_written: usize,
    /// Pending revisions, including further revisions observed in this sync.
    pub pending: usize,
}

#[derive(Serialize, Deserialize)]
pub struct AuthorizedUser {
    #[serde(rename = "type", default = "authorized_user")]
    kind: String,
    pub client_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    pub refresh_token: String,
}
fn authorized_user() -> String {
    "authorized_user".into()
}
impl AuthorizedUser {
    pub fn new(client_id: String, refresh_token: String) -> Self {
        Self {
            kind: authorized_user(),
            client_id,
            client_secret: None,
            refresh_token,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Revision {
    id: String,
    #[serde(rename = "modifiedTime")]
    modified: String,
}

impl Drive {
    pub fn sync(&self) -> Result<SyncReport, DriveError> {
        let token = self.access_token()?;
        let state = self.load_state()?;
        let pending = self.pending_revisions()?;
        for revision in pending.values() {
            self.publish(revision)?;
        }
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
            files.extend(v.array("files").cloned().unwrap_or_default());
            match v.str("nextPageToken") {
                Some(p) => page = Some(p.to_owned()),
                None => break,
            }
        }

        std::fs::create_dir_all(&self.spool)?;
        let listing = serde_json::to_vec_pretty(&serde_json::json!({ "files": files }))
            .expect("JSON value serializes");
        write_if_changed(&self.spool.join(super::LISTING), &listing)?;

        let mut report = SyncReport {
            prompts: files.len(),
            pending: pending.len(),
            ..SyncReport::default()
        };
        for f in &files {
            let Some(id) = f.str("id") else {
                continue;
            };
            if pending.contains_key(id) {
                continue;
            }
            let head = f.str("headRevisionId");
            if head.is_some() && state.revision(id) == head {
                continue;
            }
            let rev_url = format!("{files_url}/{id}/revisions");
            let mut revisions: Vec<Revision> = Vec::new();
            let mut page: Option<String> = None;
            loop {
                let mut query = vec![("fields", "nextPageToken,revisions(id,modifiedTime)")];
                if let Some(token) = &page {
                    query.push(("pageToken", token.as_str()));
                }
                let v = json(get(&rev_url, &query)?, &rev_url)?;
                let batch: Vec<Revision> = v
                    .get("revisions")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|e| DriveError::Http(rev_url.clone(), e.to_string()))?
                    .unwrap_or_default();
                revisions.extend(batch);
                match v.str("nextPageToken") {
                    Some(token) => page = Some(token.to_owned()),
                    None => break,
                }
            }
            revisions.sort_by(|a, b| a.modified.cmp(&b.modified));
            let start = state.next_revision(id, &revisions);
            let Some(next) = revisions.get(start) else {
                continue;
            };
            let url = format!("{rev_url}/{}", next.id);
            let bytes = get(&url, &[("alt", "media")])?;
            let revision = self.keep_revision(
                id,
                Some(Accepted {
                    revision: next.id.clone(),
                    modified: Some(next.modified.clone()),
                }),
                &bytes,
            )?;
            self.publish(&revision)?;
            report.revisions_written += 1;
            report.pending += revisions.len() - start;
        }
        Ok(report)
    }

    fn access_token(&self) -> Result<String, DriveError> {
        let err = |m: String| DriveError::Token(self.token_path.clone(), m);
        let raw = std::fs::read(&self.token_path).map_err(|error| {
            err(if error.kind() == std::io::ErrorKind::NotFound {
                "not authorized; run `polylogue install drive`".into()
            } else {
                error.to_string()
            })
        })?;
        let user: AuthorizedUser = serde_json::from_slice(&raw).map_err(|e| err(e.to_string()))?;
        let mut form = vec![
            ("client_id", user.client_id.as_str()),
            ("refresh_token", user.refresh_token.as_str()),
            ("grant_type", "refresh_token"),
        ];
        if let Some(secret) = user.client_secret.as_deref() {
            form.push(("client_secret", secret));
        }
        let mut resp = ureq::post(&self.endpoints.token)
            .send_form(form)
            .map_err(|e| err(format!("refresh failed: {e}")))?;
        let v: serde_json::Value = resp
            .body_mut()
            .read_json()
            .map_err(|e| err(format!("refresh reply: {e}")))?;
        v.str("access_token")
            .map(str::to_owned)
            .ok_or_else(|| err("refresh reply has no access_token".into()))
    }

    fn load_state(&self) -> Result<Cursor, DriveError> {
        let bytes = match std::fs::read(self.spool.join(STATE)) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Cursor::default());
            }
            Err(error) => return Err(error.into()),
        };
        let invalid = |error: &str| std::io::Error::new(std::io::ErrorKind::InvalidData, error);
        match serde_json::from_slice::<RevisionState>(&bytes)
            .map_err(|e| invalid(&e.to_string()))?
        {
            RevisionState::Accepted {
                version: 1,
                accepted,
                modified,
            } => Ok(Cursor { accepted, modified }),
            RevisionState::Accepted { .. } => {
                Err(invalid("unsupported Drive revision state version").into())
            }
            RevisionState::Downloaded(downloaded) => {
                // The Python Polylogue cursor acknowledged downloads, not storage.
                // Keep its last body before assigning the new meaning.
                for (id, _) in downloaded {
                    match std::fs::read(self.prompt_path(&id)?) {
                        Ok(bytes) => {
                            // Its cursor and body were separate writes, so their
                            // revision pairing cannot be assumed after a crash.
                            self.keep_revision(&id, None, &bytes)?;
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                        Err(error) => return Err(error.into()),
                    }
                }
                let cursor = Cursor::default();
                self.save_state(&cursor)?;
                Ok(cursor)
            }
        }
    }

    fn keep_revision(
        &self,
        id: &str,
        revision: Option<Accepted>,
        bytes: &[u8],
    ) -> Result<PendingRevision, DriveError> {
        self.prompt_path(id)?;
        let directory = self.spool.join(PENDING);
        std::fs::create_dir_all(&directory)?;
        std::fs::File::open(&self.spool)?.sync_all()?;
        let name = match &revision {
            Some(Accepted {
                revision,
                modified: Some(modified),
            }) => format!(
                "{}-{}-{}.json",
                polylogue_model::hex(id.as_bytes()),
                polylogue_model::hex(revision.as_bytes()),
                polylogue_model::hex(modified.as_bytes())
            ),
            Some(Accepted {
                revision,
                modified: None,
            }) => format!(
                "{}-{}.json",
                polylogue_model::hex(id.as_bytes()),
                polylogue_model::hex(revision.as_bytes())
            ),
            None => format!("{}.json", polylogue_model::hex(id.as_bytes())),
        };
        let pending = PendingRevision {
            file: id.to_owned(),
            revision,
            body: directory.join(name),
        };
        match std::fs::read(&pending.body) {
            Ok(original) if original == bytes => return Ok(pending),
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "pending Drive revision bytes changed",
                )
                .into());
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        let mut file = tempfile::NamedTempFile::new_in(&directory)?;
        file.write_all(bytes)?;
        file.as_file().sync_all()?;
        file.persist_noclobber(&pending.body).map_err(|e| e.error)?;
        std::fs::File::open(directory)?.sync_all()?;
        Ok(pending)
    }

    fn pending_revisions(&self) -> Result<BTreeMap<String, PendingRevision>, DriveError> {
        let directory = self.spool.join(PENDING);
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(BTreeMap::new());
            }
            Err(error) => return Err(error.into()),
        };
        let mut pending = BTreeMap::new();
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            let Some(stem) = name.to_str().and_then(|name| name.strip_suffix(".json")) else {
                continue; // An unpublished atomic-write temporary file.
            };
            let revision = (|| {
                let mut parts = stem.split('-');
                let file = decode(parts.next()?)?;
                let revision = match (parts.next(), parts.next()) {
                    (Some(revision), modified) => Some(Accepted {
                        revision: decode(revision)?,
                        modified: match modified {
                            Some(modified) => Some(decode(modified)?),
                            None => None,
                        },
                    }),
                    (None, _) => None,
                };
                if parts.next().is_some() {
                    return None;
                }
                Some(PendingRevision {
                    file,
                    revision,
                    body: entry.path(),
                })
            })()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "invalid pending Drive revision",
                )
            })?;
            if pending.insert(revision.file.clone(), revision).is_some() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "several pending Drive revisions for one prompt",
                )
                .into());
            }
        }
        Ok(pending)
    }

    fn publish(&self, revision: &PendingRevision) -> Result<(), DriveError> {
        let path = self.prompt_path(&revision.file)?;
        write_if_changed(&path, &std::fs::read(&revision.body)?)
    }

    fn prompt_path(&self, id: &str) -> Result<PathBuf, DriveError> {
        // Keep the provider ID as the parser's stable session identity.
        if id.is_empty() || id.contains(['/', '\\']) || id == "." || id == ".." || id == "files" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "invalid Drive prompt ID",
            )
            .into());
        }
        Ok(self.spool.join(format!("{id}.json")))
    }

    /// Advances only after the archive accepts the pending bytes. The
    /// watched path remains present when private pending resources are removed.
    pub fn advance(
        &self,
        accepted: &crate::origin::CollectorAcceptance<'_>,
    ) -> Result<(), DriveError> {
        let mut state = self.load_state()?;
        let mut changed = false;
        let mut retired = Vec::new();
        for revision in self.pending_revisions()?.into_values() {
            if revision.revision.as_ref().is_none_or(|accepted| {
                state.revision(&revision.file) != Some(accepted.revision.as_str())
            }) {
                let mut body = std::fs::File::open(&revision.body)?;
                let mut hash = blake3::Hasher::new();
                let length = std::io::copy(&mut body, &mut hash)?;
                if !accepted(
                    &self.prompt_path(&revision.file)?,
                    polylogue_model::BlobHash(*hash.finalize().as_bytes()),
                    length,
                )
                .map_err(std::io::Error::other)?
                {
                    continue;
                }
                if let Some(accepted) = revision.revision {
                    state.accept(revision.file, accepted);
                    changed = true;
                }
            }
            retired.push(revision.body);
        }
        if changed {
            self.save_state(&state)?;
        }
        if !retired.is_empty() {
            for body in retired {
                std::fs::remove_file(body)?;
            }
            std::fs::File::open(self.spool.join(PENDING))?.sync_all()?;
        }
        Ok(())
    }

    fn save_state(&self, state: &Cursor) -> Result<(), DriveError> {
        let bytes = serde_json::to_vec_pretty(&RevisionState::Accepted {
            version: 1,
            accepted: state.accepted.clone(),
            modified: state.modified.clone(),
        })
        .expect("map serializes");
        write_if_changed(&self.spool.join(STATE), &bytes)
    }
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum RevisionState {
    Accepted {
        version: u32,
        accepted: BTreeMap<String, String>,
        /// The accepted revision's Drive `modifiedTime`, per prompt, when
        /// known: it places the cursor once Drive prunes that revision.
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        modified: BTreeMap<String, String>,
    },
    Downloaded(BTreeMap<String, String>),
}

/// The last revision the archive accepted for each prompt.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Cursor {
    accepted: BTreeMap<String, String>,
    modified: BTreeMap<String, String>,
}

impl Cursor {
    fn revision(&self, file: &str) -> Option<&str> {
        self.accepted.get(file).map(String::as_str)
    }

    fn accept(&mut self, file: String, accepted: Accepted) {
        match accepted.modified {
            Some(modified) => self.modified.insert(file.clone(), modified),
            None => self.modified.remove(&file),
        };
        self.accepted.insert(file, accepted.revision);
    }

    /// Where the walk over `revisions` (oldest first) continues for `file`:
    /// after the accepted revision, or, when Drive has pruned it, at the
    /// first revision newer than it. Without a known time for a pruned
    /// revision the walk starts again from the oldest; the archive stores
    /// repeated bytes once.
    fn next_revision(&self, file: &str, revisions: &[Revision]) -> usize {
        let Some(accepted) = self.revision(file) else {
            return 0;
        };
        if let Some(at) = revisions.iter().position(|r| r.id == accepted) {
            return at + 1;
        }
        self.modified.get(file).map_or(0, |modified| {
            revisions
                .iter()
                .position(|r| r.modified > *modified)
                .unwrap_or(revisions.len())
        })
    }
}

/// A Drive revision of one prompt and its `modifiedTime`, when known.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Accepted {
    revision: String,
    modified: Option<String>,
}

struct PendingRevision {
    file: String,
    revision: Option<Accepted>,
    body: PathBuf,
}

fn decode(value: &str) -> Option<String> {
    if !value.len().is_multiple_of(2) {
        return None;
    }
    let bytes = value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect::<Option<Vec<_>>>()?;
    String::from_utf8(bytes).ok()
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<(), DriveError> {
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(());
    }
    Ok(polylogue_model::atomic_write::write_atomic(
        path, bytes, None,
    )?)
}

/// The origin's collector: Google's Drive, the token the operator
/// explicitly authorized in its archive, and the origin's spool.
pub const COLLECTOR: crate::origin::Collector = crate::origin::Collector {
    background: false,
    advance: Some(|paths, accepted| {
        if !token_path(paths.archive).is_file() {
            return Ok(());
        }
        Drive {
            endpoints: Endpoints::default(),
            token_path: token_path(paths.archive),
            spool: paths.spool.to_path_buf(),
        }
        .advance(accepted)
        .map_err(|e| e.to_string())
    }),
    setting: None,
    sync: |paths, _stop| {
        if needs_authorization(paths.archive) {
            return Vec::new();
        }
        Drive {
            endpoints: Endpoints::default(),
            token_path: token_path(paths.archive),
            spool: paths.spool.to_path_buf(),
        }
        .sync()
        .err()
        .map(|e| e.to_string())
        .into_iter()
        .collect()
    },
};

#[cfg(test)]
mod tests;
