//! A fake Drive API on a local port, for tests of the collector and of what
//! the archive makes of its spool: token refresh, the changes feed, the
//! folder lookup and listing, and each prompt's revisions and their bytes.
//! The test changes it as AI Studio and the operator would (saving a prompt,
//! deleting it, Drive pruning a revision); every request path is recorded.

use super::Endpoints;
use polylogue_model::Timestamp;
use serde_json::{Value as Json, json};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex, PoisonError};

/// The Drive ID of the `Google AI Studio` folder.
pub const FOLDER_ID: &str = "folder1";
const PROMPT_MIME: &str = "application/vnd.google-makersuite.prompt";

struct File {
    name: String,
    created: String,
    /// In the prompts folder (not elsewhere in the Drive).
    in_folder: bool,
    deleted: bool,
    /// In Drive's list order.
    revisions: Vec<(String, String, Vec<u8>)>,
}

type Answer = (Box<dyn Fn(&str) -> bool + Send>, String);

struct State {
    folder: bool,
    files: BTreeMap<String, File>,
    /// The file each change names, in order; a page token is an index.
    changes: Vec<String>,
    page: usize,
    seen: Vec<String>,
    answers: Vec<Answer>,
}

pub struct FakeDrive {
    base: String,
    state: Arc<Mutex<State>>,
}

impl FakeDrive {
    /// A Drive with an empty prompts folder and pages of any length.
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind a local port");
        let base = format!("http://{}", listener.local_addr().expect("local address"));
        let state = Arc::new(Mutex::new(State {
            folder: true,
            files: BTreeMap::new(),
            changes: Vec::new(),
            page: usize::MAX,
            seen: Vec::new(),
            answers: Vec::new(),
        }));
        let shared = Arc::clone(&state);
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                serve(stream, &shared);
            }
        });
        Self { base, state }
    }

    pub fn endpoints(&self) -> Endpoints {
        Endpoints {
            api: format!("{}/drive", self.base),
            token: format!("{}/token", self.base),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// No folder named `Google AI Studio`.
    pub fn without_folder(&self) {
        self.state().folder = false;
    }

    /// Lists, revisions and changes come in pages of `page` items.
    pub fn page_size(&self, page: usize) {
        self.state().page = page;
    }

    /// Creates the prompt `id` in the folder, named `name`.
    pub fn create(&self, id: &str, name: &str, created: &str) {
        self.add(id, name, created, true);
    }

    /// Creates a file outside the prompts folder.
    pub fn create_elsewhere(&self, id: &str, name: &str, created: &str) {
        self.add(id, name, created, false);
    }

    fn add(&self, id: &str, name: &str, created: &str, in_folder: bool) {
        let mut state = self.state();
        state.files.insert(
            id.to_owned(),
            File {
                name: name.to_owned(),
                created: created.to_owned(),
                in_folder,
                deleted: false,
                revisions: Vec::new(),
            },
        );
        state.changes.push(id.to_owned());
    }

    /// Saves a revision of `id`, listed after its earlier revisions.
    pub fn save(&self, id: &str, revision: &str, modified: &str, body: &[u8]) {
        let mut state = self.state();
        state
            .files
            .get_mut(id)
            .expect("created file")
            .revisions
            .push((revision.to_owned(), modified.to_owned(), body.to_vec()));
        state.changes.push(id.to_owned());
    }

    /// Deletes `id`, as emptying it from the trash does.
    pub fn delete(&self, id: &str) {
        let mut state = self.state();
        state.files.get_mut(id).expect("created file").deleted = true;
        state.changes.push(id.to_owned());
    }

    /// Drops revisions of `id`, as Drive prunes old revisions: no change.
    pub fn prune(&self, id: &str, keep: impl Fn(&str) -> bool) {
        self.state()
            .files
            .get_mut(id)
            .expect("created file")
            .revisions
            .retain(|(revision, _, _)| keep(revision));
    }

    /// Answers requests whose path and query match with `body` instead.
    pub fn answer_when(&self, matches: impl Fn(&str) -> bool + Send + 'static, body: &str) {
        self.state()
            .answers
            .push((Box::new(matches), body.to_owned()));
    }

    /// Every request's path and query, in order.
    pub fn seen(&self) -> Vec<String> {
        self.state().seen.clone()
    }
}

fn serve(mut stream: std::net::TcpStream, state: &Mutex<State>) {
    let Ok(clone) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(clone);
    let mut line = String::new();
    if reader.read_line(&mut line).is_err() {
        return;
    }
    let target = line.split_whitespace().nth(1).unwrap_or("").to_owned();
    let mut length = 0usize;
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).is_err() || header == "\r\n" || header.is_empty() {
            break;
        }
        if let Some(value) = header.to_ascii_lowercase().strip_prefix("content-length:") {
            length = value.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0; length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    let (status, reply) = {
        let mut state = state.lock().unwrap_or_else(PoisonError::into_inner);
        state.seen.push(target.clone());
        match state.answers.iter().find(|(matches, _)| matches(&target)) {
            Some((_, body)) => ("200 OK", body.clone().into_bytes()),
            None => answer(&state, &target),
        }
    };
    let _ = write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        reply.len()
    );
    let _ = stream.write_all(&reply);
}

fn answer(state: &State, target: &str) -> (&'static str, Vec<u8>) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let param = |name: &str| {
        query
            .split('&')
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
    };
    let from: usize = param("pageToken").map_or(0, |token| token.parse().unwrap_or(usize::MAX));
    let ok = |value: Json| ("200 OK", value.to_string().into_bytes());
    let not_found = ("404 Not Found", br#"{"error":{"code":404}}"#.to_vec());
    // One page of `items` from `from`, and the token of the next one.
    let page = |count: usize| {
        let to = from.saturating_add(state.page).min(count);
        (from.min(to)..to, (to < count).then(|| to.to_string()))
    };
    match path {
        "/token" => ok(json!({"access_token": "t", "expires_in": 3600})),
        "/drive/changes/startPageToken" => {
            ok(json!({"startPageToken": state.changes.len().to_string()}))
        }
        "/drive/changes" => {
            let (range, next) = page(state.changes.len());
            let changes: Vec<Json> = state.changes[range]
                .iter()
                .map(|id| {
                    let file = &state.files[id];
                    if file.deleted {
                        json!({"changeType": "file", "fileId": id, "removed": true})
                    } else {
                        let mut entry = metadata(id, file);
                        let parent = if file.in_folder { FOLDER_ID } else { "root" };
                        entry["parents"] = json!([parent]);
                        entry["trashed"] = json!(false);
                        json!({"changeType": "file", "fileId": id, "removed": false, "file": entry})
                    }
                })
                .collect();
            ok(match next {
                Some(next) => json!({"changes": changes, "nextPageToken": next}),
                None => {
                    json!({"changes": changes, "newStartPageToken": state.changes.len().to_string()})
                }
            })
        }
        "/drive/files" if query.contains("apps.folder") => ok(if state.folder {
            json!({"files": [{"id": FOLDER_ID}]})
        } else {
            json!({"files": []})
        }),
        "/drive/files" => {
            let listed: Vec<Json> = state
                .files
                .iter()
                .filter(|(_, f)| f.in_folder && !f.deleted)
                .map(|(id, f)| metadata(id, f))
                .collect();
            let (range, next) = page(listed.len());
            let mut reply = json!({"files": listed[range].to_vec()});
            if let Some(next) = next {
                reply["nextPageToken"] = json!(next);
            }
            ok(reply)
        }
        _ => {
            let Some(rest) = path.strip_prefix("/drive/files/") else {
                return not_found;
            };
            let (id, rest) = rest.split_once('/').unwrap_or((rest, ""));
            let Some(file) = state.files.get(id).filter(|f| !f.deleted) else {
                return not_found;
            };
            match rest.strip_prefix("revisions") {
                Some("") => {
                    let (range, next) = page(file.revisions.len());
                    let revisions: Vec<Json> = file.revisions[range]
                        .iter()
                        .map(|(id, modified, _)| json!({"id": id, "modifiedTime": modified}))
                        .collect();
                    let mut reply = json!({"revisions": revisions});
                    if let Some(next) = next {
                        reply["nextPageToken"] = json!(next);
                    }
                    ok(reply)
                }
                Some(revision) => {
                    match file
                        .revisions
                        .iter()
                        .find(|(id, _, _)| Some(id.as_str()) == revision.strip_prefix('/'))
                    {
                        Some((_, _, body)) => ("200 OK", body.clone()),
                        None => not_found,
                    }
                }
                None => not_found,
            }
        }
    }
}

/// The file's metadata as `files.list` gives it: the head revision is the
/// newest one.
fn metadata(id: &str, file: &File) -> Json {
    let head = file
        .revisions
        .iter()
        .max_by_key(|(_, modified, _)| Timestamp::parse_rfc3339(modified));
    let mut entry = json!({
        "id": id,
        "name": file.name,
        "mimeType": PROMPT_MIME,
        "createdTime": file.created,
    });
    if let Some((revision, modified, _)) = head {
        entry["headRevisionId"] = json!(revision);
        entry["modifiedTime"] = json!(modified);
    }
    entry
}
