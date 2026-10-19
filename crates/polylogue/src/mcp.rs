//! MCP over stdio, as a proxy to the daemon: each tool sends the same
//! request the CLI sends and returns the daemon's JSON answer. Context the
//! tools hand to the model (briefs, takeovers, recalls) is recorded as a
//! delivery from `mcp`. Agent-written notes come back marked with their
//! author, as information.

use crate::briefing::{Ask, StartSource};
use crate::daemon::{Request, Response};
use polylogue_model::{Author, NewNote, NoteKind, SessionId, Target};
use polylogue_store::{Archive, NoteFilter};
use rmcp::handler::server::wrapper::Parameters;
use rmcp::{ServiceExt, tool, tool_handler, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Clone, Default)]
pub struct Polylogue {
    socket: PathBuf,
}

/// An error as tool text, with the corrected query when the daemon formed
/// one, so an agent can retry with it as given.
fn error_text(kind: &str, message: &str, detail: Option<&serde_json::Value>) -> String {
    match detail.and_then(|d| d["suggestion"].as_str()) {
        Some(s) => format!("error ({kind}): {message}\ntry: {s}"),
        None => format!("error ({kind}): {message}"),
    }
}

/// Turns a daemon answer into tool text: the data as JSON, or the error.
fn answer(request: &Request, socket: &std::path::Path) -> String {
    let archive = Archive::at(socket.parent().unwrap_or(socket));
    match crate::cli::send_when_ready(&archive, request) {
        Ok(Response::Ok { data }) => serde_json::to_string_pretty(&data).unwrap_or_default(),
        Ok(Response::Accepted { request_id }) => format!("accepted {request_id}"),
        Ok(Response::Error {
            kind,
            message,
            detail,
        }) => error_text(&kind, &message, detail.as_ref()),
        Err(e) => format!(
            "error (daemon_required): no Polylogue daemon answered ({e}); start it with `polylogue daemon`"
        ),
    }
}

/// The context text of a `context` answer, or the error.
fn context_text(request: &Request, socket: &std::path::Path) -> String {
    let archive = Archive::at(socket.parent().unwrap_or(socket));
    match crate::cli::send_when_ready(&archive, request) {
        Ok(Response::Ok { data }) => data["prepared"]["text"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        Ok(Response::Error {
            kind,
            message,
            detail,
        }) => error_text(&kind, &message, detail.as_ref()),
        Ok(Response::Accepted { request_id }) => format!("accepted {request_id}"),
        Err(e) => format!("error (daemon_required): {e}"),
    }
}

fn session(text: &str) -> Result<SessionId, String> {
    SessionId::parse(text).map_err(|e| format!("error (bad_request): {e}"))
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListArgs {
    /// How many sessions, newest first.
    pub limit: Option<u64>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReadArgs {
    /// `origin:native_id`, as `list` prints it.
    pub session: String,
    /// Include messages inherited from a parent session.
    #[serde(default)]
    pub logical: bool,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct BriefArgs {
    /// The working directory the brief is for.
    pub cwd: String,
    /// The session asking, when known (`claude-code:<id>`).
    pub session: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct SessionArg {
    /// `origin:native_id`.
    pub session: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallArgs {
    /// What to look for; every word must match.
    pub text: String,
    /// Prefer sessions from this directory.
    pub cwd: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct NotesArgs {
    /// References: `session:<id>`, `repo:<name>`, `cwd:<path>`, ...
    #[serde(default)]
    pub targets: Vec<String>,
    /// Note kinds: lesson, handoff, decision, ...
    #[serde(default)]
    pub kinds: Vec<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct AddNoteArgs {
    /// References the note is about: `session:<id>`, `repo:<name>`, ...
    pub targets: Vec<String>,
    /// note, lesson, handoff, decision, blocker, question, finding, ...
    pub kind: String,
    pub text: String,
    /// The session writing the note (`claude-code:<id>`), when known.
    pub author_session: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct CandidatesArgs {
    /// The directory to look in.
    pub cwd: String,
}

#[tool_router]
impl Polylogue {
    pub fn new() -> Self {
        Self {
            socket: Archive::at(&Archive::default_root()).socket(),
        }
    }

    #[tool(description = "What the archive holds and what the daemon is doing.")]
    async fn status(&self) -> String {
        answer(&Request::Status, &self.socket)
    }

    #[tool(description = "Sessions by latest activity, newest first.")]
    async fn list(&self, Parameters(a): Parameters<ListArgs>) -> String {
        answer(
            &Request::List {
                limit: a.limit.unwrap_or(20),
                before: None,
            },
            &self.socket,
        )
    }

    #[tool(description = "Read one session's messages.")]
    async fn read(&self, Parameters(a): Parameters<ReadArgs>) -> String {
        let session = match session(&a.session) {
            Ok(s) => s,
            Err(e) => return e,
        };
        answer(
            &Request::Read {
                session,
                view: "full".into(),
                format: "json".into(),
                logical: a.logical,
                budget: None,
                cursor: None,
            },
            &self.socket,
        )
    }

    #[tool(
        description = "The session-start brief for a directory: the operator's lessons, handoffs and decisions there, and recent sessions."
    )]
    async fn brief(&self, Parameters(a): Parameters<BriefArgs>) -> String {
        let session = match a.session.as_deref().map(session).transpose() {
            Ok(s) => s,
            Err(e) => return e,
        };
        context_text(
            &Request::Context {
                request_id: polylogue_store::user::new_request_id(),
                ask: Ask::SessionStart {
                    cwd: a.cwd.into(),
                    session,
                    source: StartSource::Startup,
                },
                sender: Some("mcp".into()),
            },
            &self.socket,
        )
    }

    #[tool(
        description = "Context for taking over another session: its prompts, the operator's notes on it, what it left open and its last answer."
    )]
    async fn takeover(&self, Parameters(a): Parameters<SessionArg>) -> String {
        let session = match session(&a.session) {
            Ok(s) => s,
            Err(e) => return e,
        };
        context_text(
            &Request::Context {
                request_id: polylogue_store::user::new_request_id(),
                ask: Ask::Takeover {
                    session,
                    for_session: None,
                },
                sender: Some("mcp".into()),
            },
            &self.socket,
        )
    }

    #[tool(
        description = "Search past sessions for something relevant now. Returns matching messages grouped by session, notes on them, and a receipt of how it searched."
    )]
    async fn recall(&self, Parameters(a): Parameters<RecallArgs>) -> String {
        context_text(
            &Request::Context {
                request_id: polylogue_store::user::new_request_id(),
                ask: Ask::Recall {
                    text: a.text,
                    cwd: a.cwd.map(PathBuf::from),
                    for_session: None,
                },
                sender: Some("mcp".into()),
            },
            &self.socket,
        )
    }

    #[tool(description = "Notes on targets; agent-written notes are marked by their author.")]
    async fn notes(&self, Parameters(a): Parameters<NotesArgs>) -> String {
        let targets: Result<Vec<Target>, _> = a.targets.iter().map(|t| Target::parse(t)).collect();
        let Ok(targets) = targets else {
            return "error (bad_request): a target does not parse".into();
        };
        let kinds: Option<Vec<NoteKind>> = a.kinds.iter().map(|k| NoteKind::parse(k)).collect();
        let Some(kinds) = kinds else {
            return "error (bad_request): unknown note kind".into();
        };
        answer(
            &Request::Notes {
                filter: NoteFilter {
                    targets,
                    kinds,
                    ..NoteFilter::default()
                },
            },
            &self.socket,
        )
    }

    #[tool(
        description = "Write a note as an agent. It is stored with you as its author and shown on request; it is never injected into other sessions automatically."
    )]
    async fn add_note(&self, Parameters(a): Parameters<AddNoteArgs>) -> String {
        let targets: Result<Vec<Target>, _> = a.targets.iter().map(|t| Target::parse(t)).collect();
        let Ok(targets) = targets else {
            return "error (bad_request): a target does not parse".into();
        };
        let Some(kind) = NoteKind::parse(&a.kind) else {
            return format!("error (bad_request): unknown note kind {:?}", a.kind);
        };
        let author = match a.author_session.as_deref().map(session).transpose() {
            Ok(s) => Author::Agent(s),
            Err(e) => return e,
        };
        answer(
            &Request::AddNote {
                request_id: polylogue_store::user::new_request_id(),
                note: NewNote {
                    kind,
                    key: None,
                    value: Some(a.text),
                    author,
                    targets,
                    relations: vec![],
                    revises: None,
                    valid_until: None,
                },
            },
            &self.socket,
        )
    }

    #[tool(
        description = "Sessions in a directory that left something open (an unanswered prompt, an unfinished tool call, a todo, a handoff), with their native resume commands."
    )]
    async fn resume_candidates(&self, Parameters(a): Parameters<CandidatesArgs>) -> String {
        answer(
            &Request::Candidates {
                cwd: a.cwd.into(),
                limit: None,
            },
            &self.socket,
        )
    }
}

#[tool_handler(
    name = "polylogue",
    instructions = "Polylogue: the archive of your past AI sessions. Use brief at the start of work, recall to find earlier work, takeover to continue another session."
)]
impl rmcp::ServerHandler for Polylogue {}

pub fn serve_stdio() -> std::io::Result<()> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let service = Polylogue::new()
            .serve(rmcp::transport::stdio())
            .await
            .map_err(std::io::Error::other)?;
        service.waiting().await.map_err(std::io::Error::other)?;
        Ok(())
    })
}
