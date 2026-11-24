//! The AI Studio collector: reads the prompts AI Studio
//! saves in the operator's Drive and writes them into the spool the
//! `ai-studio` origin watches. It runs only with the archive's explicitly
//! authorized `drive-token.json`, in Google's authorized-user format.
//!
//! The first sync takes a start page token from Drive's changes feed and
//! lists the prompts folder once; every later sync reads the changes since
//! the saved token and applies them to its listing (`files.json`). A prompt
//! deleted, trashed or moved out of the folder keeps its entry, marked
//! `removed`.
//!
//! Each prompt has one immutable pending revision. Its bytes are published at
//! `<fileId>.json` until the engine acknowledges that the archive stored them. Only
//! then does `.revisions.json` advance and another revision replace that path.
//! Hidden pending bodies survive restart; their cleanup leaves the watched
//! prompt in place.

use polylogue_model::{JsonExt, Timestamp};
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
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
/// The file metadata the listing keeps.
const FILE_FIELDS: &str = "id,name,mimeType,createdTime,modifiedTime,headRevisionId";
/// Change fields: the listing's file metadata, and where the file is.
const CHANGE_FIELDS: &str = "nextPageToken,newStartPageToken,changes(changeType,fileId,removed,file(id,name,mimeType,createdTime,modifiedTime,headRevisionId,trashed,parents))";

#[derive(Debug, thiserror::Error)]
pub enum DriveError {
    #[error("Drive token {0}: {1}")]
    Token(PathBuf, String),
    #[error("Drive request {0}: {1}")]
    Http(String, String),
    /// A reply that arrived but does not say what the request asks.
    #[error("Drive reply to {0} is unreadable: {1}")]
    Unreadable(String, String),
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
    /// Prompts in the folder.
    pub prompts: usize,
    /// Prompts the listing keeps marked removed.
    pub removed: usize,
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

/// A Drive revision of one prompt, with its `modifiedTime` as Drive wrote
/// it and as an instant.
#[derive(Debug, Clone)]
struct Revision {
    id: String,
    modified: String,
    at: Timestamp,
}

/// The prompts folder's files by ID, as the listing keeps them.
type Listing = BTreeMap<String, Json>;

/// Drive's API with one access token.
struct Api {
    agent: ureq::Agent,
    token: String,
    base: String,
}

impl Api {
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn bytes(&self, path: &str, query: &[(&str, &str)]) -> Result<Vec<u8>, DriveError> {
        let url = self.url(path);
        let mut request = self
            .agent
            .get(&url)
            .header("Authorization", &format!("Bearer {}", self.token));
        for (key, value) in query {
            request = request.query(*key, *value);
        }
        let mut reply = request
            .call()
            .map_err(|e| DriveError::Http(url.clone(), e.to_string()))?;
        reply
            .body_mut()
            .with_config()
            .limit(u64::MAX)
            .read_to_vec()
            .map_err(|e| DriveError::Http(url, e.to_string()))
    }

    fn json(&self, path: &str, query: &[(&str, &str)]) -> Result<Json, DriveError> {
        serde_json::from_slice(&self.bytes(path, query)?)
            .map_err(|e| DriveError::Unreadable(self.url(path), e.to_string()))
    }

    fn unreadable(&self, path: &str, reason: impl Into<String>) -> DriveError {
        DriveError::Unreadable(self.url(path), reason.into())
    }

    fn start_page_token(&self) -> Result<String, DriveError> {
        let path = "/changes/startPageToken";
        self.json(path, &[("fields", "startPageToken")])?
            .str("startPageToken")
            .map(str::to_owned)
            .ok_or_else(|| self.unreadable(path, "no `startPageToken`"))
    }

    fn folder(&self) -> Result<String, DriveError> {
        let query = format!("name = '{FOLDER}' and mimeType = '{FOLDER_MIME}' and trashed = false");
        let found = self.json("/files", &[("q", &query), ("fields", "files(id)")])?;
        found
            .pointer("/files/0/id")
            .and_then(Json::as_str)
            .map(str::to_owned)
            .ok_or(DriveError::NoFolder)
    }

    /// Every file in the folder, through every page.
    fn list(&self, folder: &str) -> Result<Vec<Json>, DriveError> {
        let mut files = Vec::new();
        let mut page: Option<String> = None;
        let query = format!("'{folder}' in parents and trashed = false");
        let fields = format!("nextPageToken,files({FILE_FIELDS})");
        loop {
            let mut params = vec![
                ("q", query.as_str()),
                ("fields", fields.as_str()),
                ("pageSize", "1000"),
            ];
            if let Some(p) = &page {
                params.push(("pageToken", p.as_str()));
            }
            let reply = self.json("/files", &params)?;
            // A reply without `files` is not an empty folder: reading it so
            // would mark every prompt removed.
            let batch = reply
                .array("files")
                .ok_or_else(|| self.unreadable("/files", "no `files` array"))?;
            files.extend(batch.iter().cloned());
            match reply.str("nextPageToken") {
                Some(p) => page = Some(p.to_owned()),
                None => return Ok(files),
            }
        }
    }

    /// Applies every change since `token` to the listing and returns the
    /// token to continue from, or `None` when the folder itself is gone.
    fn follow(
        &self,
        token: &str,
        folder: &str,
        listing: &mut Listing,
    ) -> Result<Option<String>, DriveError> {
        let path = "/changes";
        let mut page = token.to_owned();
        loop {
            let reply = self.json(
                path,
                &[
                    ("pageToken", page.as_str()),
                    ("fields", CHANGE_FIELDS),
                    ("pageSize", "1000"),
                    ("spaces", "drive"),
                    ("includeRemoved", "true"),
                ],
            )?;
            let changes = reply
                .array("changes")
                .ok_or_else(|| self.unreadable(path, "no `changes` array"))?;
            for change in changes {
                if !apply(change, folder, listing).map_err(|e| self.unreadable(path, e))? {
                    return Ok(None);
                }
            }
            if let Some(next) = reply.str("nextPageToken") {
                page = next.to_owned();
            } else if let Some(start) = reply.str("newStartPageToken") {
                return Ok(Some(start.to_owned()));
            } else {
                return Err(
                    self.unreadable(path, "neither `nextPageToken` nor `newStartPageToken`")
                );
            }
        }
    }

    /// Every revision of a prompt, through every page, oldest first.
    fn revisions(&self, id: &str) -> Result<Vec<Revision>, DriveError> {
        let path = format!("/files/{id}/revisions");
        let mut revisions = Vec::new();
        let mut page: Option<String> = None;
        loop {
            let mut params = vec![("fields", "nextPageToken,revisions(id,modifiedTime)")];
            if let Some(token) = &page {
                params.push(("pageToken", token.as_str()));
            }
            let reply = self.json(&path, &params)?;
            // A reply without `revisions` is not a prompt without revisions.
            let batch = reply
                .array("revisions")
                .ok_or_else(|| self.unreadable(&path, "no `revisions` array"))?;
            for revision in batch {
                let (Some(id), Some(modified)) = (revision.str("id"), revision.str("modifiedTime"))
                else {
                    return Err(self.unreadable(&path, "a revision without `id` or `modifiedTime`"));
                };
                let at = Timestamp::parse_rfc3339(modified).ok_or_else(|| {
                    self.unreadable(
                        &path,
                        format!("revision {id} has modifiedTime {modified:?}"),
                    )
                })?;
                revisions.push(Revision {
                    id: id.to_owned(),
                    modified: modified.to_owned(),
                    at,
                });
            }
            match reply.str("nextPageToken") {
                Some(token) => page = Some(token.to_owned()),
                None => break,
            }
        }
        // Instants, not texts: Drive writes fractions of varying width.
        revisions.sort_by_key(|r| r.at);
        Ok(revisions)
    }
}

/// Applies one change to the listing; false when the change removes the
/// folder itself.
fn apply(change: &Json, folder: &str, listing: &mut Listing) -> Result<bool, String> {
    if change.str("changeType").is_some_and(|kind| kind != "file") {
        return Ok(true);
    }
    let id = change.str("fileId").ok_or("a change without `fileId`")?;
    let file = change
        .get("file")
        .filter(|_| change.bool("removed") != Some(true))
        .filter(|file| file.bool("trashed") != Some(true));
    if id == folder {
        return Ok(file.is_some());
    }
    let in_folder = file.filter(|file| {
        file.array("parents")
            .is_some_and(|parents| parents.iter().any(|p| p.as_str() == Some(folder)))
    });
    match in_folder {
        Some(file) => {
            let mut entry = file.clone();
            if let Some(entry) = entry.as_object_mut() {
                entry.remove("trashed");
                entry.remove("parents");
                entry.insert("id".into(), id.into());
            }
            listing.insert(id.to_owned(), entry);
        }
        None => {
            if let Some(entry) = listing.get_mut(id) {
                mark_removed(entry);
            }
        }
    }
    Ok(true)
}

fn mark_removed(entry: &mut Json) {
    if let Some(entry) = entry.as_object_mut() {
        entry.insert("removed".into(), true.into());
    }
}

fn is_removed(entry: &Json) -> bool {
    entry.bool("removed") == Some(true)
}

impl Drive {
    pub fn sync(&self) -> Result<SyncReport, DriveError> {
        let api = Api {
            agent: ureq::Agent::new_with_defaults(),
            token: self.access_token()?,
            base: self.endpoints.api.clone(),
        };
        let mut state = self.load_state()?;
        let pending = self.pending_revisions()?;
        for revision in pending.values() {
            self.publish(revision)?;
        }
        let mut listing = self.load_listing()?;
        let followed = match (&state.change_token, &state.folder) {
            (Some(token), Some(folder)) => api
                .follow(token, folder, &mut listing)?
                .map(|next| (folder.clone(), next)),
            _ => None,
        };
        let (folder, next_token) = match followed {
            Some(followed) => followed,
            None => {
                // The token comes first, so a change made during the listing
                // is read again from the feed.
                let token = api.start_page_token()?;
                let folder = api.folder()?;
                let files = api.list(&folder)?;
                let present: std::collections::BTreeSet<&str> =
                    files.iter().filter_map(|f| f.str("id")).collect();
                for (id, entry) in &mut listing {
                    if !present.contains(id.as_str()) {
                        mark_removed(entry);
                    }
                }
                for file in &files {
                    if let Some(id) = file.str("id") {
                        listing.insert(id.to_owned(), file.clone());
                    }
                }
                (folder, token)
            }
        };

        std::fs::create_dir_all(&self.spool)?;
        let files: Vec<&Json> = listing.values().collect();
        let bytes = serde_json::to_vec_pretty(&serde_json::json!({ "files": files }))
            .expect("JSON value serializes");
        write_if_changed(&self.spool.join(super::LISTING), &bytes)?;

        let removed = listing.values().filter(|f| is_removed(f)).count();
        let mut report = SyncReport {
            prompts: listing.len() - removed,
            removed,
            pending: pending.len(),
            ..SyncReport::default()
        };
        for (id, file) in &listing {
            if is_removed(file) || pending.contains_key(id) {
                continue;
            }
            let head = file.str("headRevisionId");
            if head.is_some() && state.revision(id) == head {
                continue;
            }
            let revisions = api.revisions(id)?;
            let start = state.next_revision(id, &revisions);
            let Some(next) = revisions.get(start) else {
                continue;
            };
            let bytes = api.bytes(
                &format!("/files/{id}/revisions/{}", next.id),
                &[("alt", "media")],
            )?;
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
        // Saved last: a sync that fails reads the same changes again.
        state.folder = Some(folder);
        state.change_token = Some(next_token);
        self.save_state(&state)?;
        Ok(report)
    }

    /// The listing the last sync wrote; empty before the first.
    fn load_listing(&self) -> Result<Listing, DriveError> {
        let bytes = match std::fs::read(self.spool.join(super::LISTING)) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(Listing::new());
            }
            Err(error) => return Err(error.into()),
        };
        let invalid = |error: String| std::io::Error::new(std::io::ErrorKind::InvalidData, error);
        let saved: Json = serde_json::from_slice(&bytes).map_err(|e| invalid(e.to_string()))?;
        let files = saved
            .array("files")
            .ok_or_else(|| invalid("the saved Drive listing has no `files` array".into()))?;
        Ok(files
            .iter()
            .filter_map(|f| Some((f.str("id")?.to_owned(), f.clone())))
            .collect())
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
                change_token,
                folder,
            } => Ok(Cursor {
                accepted,
                modified,
                change_token,
                folder,
            }),
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
            change_token: state.change_token.clone(),
            folder: state.folder.clone(),
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
        /// The changes-feed page token the next sync reads from.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        change_token: Option<String>,
        /// The prompts folder's Drive ID.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        folder: Option<String>,
    },
    Downloaded(BTreeMap<String, String>),
}

/// The last revision the archive accepted for each prompt, and where the
/// changes feed continues.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct Cursor {
    accepted: BTreeMap<String, String>,
    modified: BTreeMap<String, String>,
    change_token: Option<String>,
    folder: Option<String>,
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
        self.modified
            .get(file)
            .and_then(|modified| Timestamp::parse_rfc3339(modified))
            .map_or(0, |modified| {
                revisions
                    .iter()
                    .position(|r| r.at > modified)
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
        .map(|pair| {
            // `from_str_radix` also accepts a leading `+`.
            if !pair.iter().all(u8::is_ascii_hexdigit) {
                return None;
            }
            u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()
        })
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

#[cfg(any(test, feature = "fixture"))]
pub mod fake;
#[cfg(test)]
mod tests;
