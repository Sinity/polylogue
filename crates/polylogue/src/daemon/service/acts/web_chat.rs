//! Web chats as recipients of the acts: `send` into a web-chat session (or a
//! pasted conversation URL not yet archived), `start --in chatgpt` for a new
//! conversation, and the operator's `answer allow|deny` on a held turn,
//! which the popup's Allow and Deny also give. Each turn is a chat action
//! row that the browser extension claims and carries out
//! ([`polylogue_store::chat_actions`]); its reply comes back as a capture.
use super::super::browser as chat;
use super::{Failure, error};
use crate::browser::actions::{self, ChatFile, ChatRequest};
use crate::daemon::handler::Call;
use crate::protocol::acts::Start;
use crate::protocol::browser::ChatAnswer;
use command_surface::Answer;
use polylogue_model::SessionId;
use polylogue_model::authorship::Author;
use polylogue_model::user::{ChatAction, ChatStatus, UserWrite};
use std::path::{Path, PathBuf};

#[cfg(test)]
mod tests;

/// The web chat a session is a conversation of: the extension's name for
/// the provider and the provider's conversation ID.
pub(crate) fn conversation(session: &SessionId) -> Option<(&'static str, String)> {
    let web = polylogue_sources::origin::by_name(&session.origin().to_string())?
        .web
        .as_ref()?;
    Some((web.extension_names.first()?, session.native().to_string()))
}

/// `send TEXT` to a web-chat session: a turn into its conversation, always
/// held for the operator's answer, whoever sent it.
pub(crate) fn send(
    call: &Call<'_>,
    session: &SessionId,
    text: &str,
    attach: &[String],
) -> Result<Answer<ChatAnswer>, Failure> {
    let Some((provider, conversation)) = conversation(session) else {
        return Err(Failure::bad_request(format!(
            "{session} is not a web-chat conversation"
        )));
    };
    let attach: Vec<PathBuf> = attach.iter().map(PathBuf::from).collect();
    queue(
        call,
        ChatRequest {
            provider: provider.to_owned(),
            conversation: Some(conversation),
            project: None,
            text: text.to_owned(),
            model: None,
            effort: None,
            submit: true,
        },
        &attach,
        true,
    )
}

/// `start TEXT --in chatgpt`: a new conversation the extension drafts, or
/// with `--submit` sends. A submit by anyone but the operator is held for
/// the operator's answer.
pub(crate) fn start(
    call: &Call<'_>,
    provider: &str,
    a: Start,
) -> Result<Answer<ChatAnswer>, Failure> {
    if a.role.is_some() || a.cwd.is_some() {
        return Err(Failure::bad_request(
            "--role and --cwd start harness sessions; a web conversation takes --model, --effort, --project, --attach and --submit",
        ));
    }
    let held = a.submit && call.invocation.authorship.author != Author::Operator;
    queue(
        call,
        ChatRequest {
            provider: provider.to_owned(),
            conversation: None,
            project: a.project,
            text: a.text,
            model: a.model,
            effort: a.effort,
            submit: a.submit,
        },
        &a.attach,
        held,
    )
}

/// Stores the attached files and queues the chat action; the request ID is
/// its idempotency key. The answer is the action as the extension sees it.
fn queue(
    call: &Call<'_>,
    request: ChatRequest,
    attach: &[PathBuf],
    held: bool,
) -> Result<Answer<ChatAnswer>, Failure> {
    let reader = call.reader()?;
    let on = chat::enabled(call.invocation, &reader)?;
    drop(reader);
    if !on {
        return Ok(error(
            polylogue_model::Code::BrowserActionsOff,
            "chat actions are off; turn them on with `polylogue set browser.actions on`",
        ));
    }
    let files = attach
        .iter()
        .map(|path| read_file(&call.path(path, "attachment")?))
        .collect::<Result<Vec<_>, _>>()?;
    // Checked before any file is stored.
    if let Err(refusal) = actions::new_action(call.request_id, &request, vec![], held) {
        return Ok(super::super::browser_refusal(refusal));
    }
    let stored = chat::store_files(call.daemon, &files)?;
    Ok(
        match actions::new_action(call.request_id, &request, stored, held) {
            Ok(new) => {
                chat::chat_then_view(call.daemon, call.invocation, UserWrite::ChatEnqueue(new))
            }
            Err(refusal) => super::super::browser_refusal(refusal),
        },
    )
}

/// A file attached to a turn, read on the daemon's host.
fn read_file(path: &Path) -> Result<ChatFile, Failure> {
    let bytes = std::fs::read(path)
        .map_err(|e| Failure::bad_value(format!("cannot read {}: {e}", path.display())))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .ok_or_else(|| Failure::bad_value(format!("{} names no file", path.display())))?;
    if let Err(refusal) = actions::check_file_name(&name) {
        return Err(Failure::new(refusal.0, refusal.0.as_str()));
    }
    Ok(ChatFile {
        media_type: media_type(&name).to_owned(),
        name,
        bytes,
    })
}

/// The media type a file's extension names; the provider checks the bytes.
fn media_type(name: &str) -> &'static str {
    let extension = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    match extension.as_deref() {
        Some("txt" | "md") => "text/plain",
        Some("json") => "application/json",
        Some("pdf") => "application/pdf",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("csv") => "text/csv",
        Some("html") => "text/html",
        _ => "application/octet-stream",
    }
}

/// `answer allow|deny` on a web-chat session: the operator's answer to the
/// oldest turn held for it.
pub(crate) fn answer(
    call: &Call<'_>,
    session: &SessionId,
    value: &str,
    reason: Option<String>,
) -> Result<Answer<ChatAnswer>, Failure> {
    let Some((provider, conversation)) = conversation(session) else {
        return Err(Failure::bad_request(format!(
            "{session} is not a web-chat conversation"
        )));
    };
    let now = call.invocation.at.unix_ms();
    let actions = call.reader()?.chat_actions()?;
    let of_session: Vec<&ChatAction> = actions
        .iter()
        .filter(|action| {
            action.needs_approval
                && action.request["provider"] == provider
                && action.request["conversation"] == conversation.as_str()
        })
        .collect();
    let held = of_session
        .iter()
        .find(|action| action.status(now).0 == ChatStatus::AwaitingApproval)
        .or_else(|| of_session.last());
    match held {
        Some(action) => answer_action(call, &action.id, value, reason),
        None => Ok(error(
            polylogue_model::Code::Refused,
            format!("{session} has no held turn to answer"),
        )),
    }
}

/// The operator's answer to one held chat action, from `answer` or the
/// popup. Over MCP it is refused: the hold exists so that the operator sees
/// what is posted under the operator's account. The first answer wins; a
/// later one is refused with what won.
pub(crate) fn answer_action(
    call: &Call<'_>,
    id: &str,
    value: &str,
    reason: Option<String>,
) -> Result<Answer<ChatAnswer>, Failure> {
    let approve = match value {
        "allow" => true,
        "deny" => false,
        other => {
            return Err(Failure::bad_value(format!(
                "a held turn is answered allow or deny, not {other:?}"
            )));
        }
    };
    if call.caller.channel() == polylogue_model::authorship::Channel::Mcp {
        return Ok(error(
            polylogue_model::Code::Refused,
            "a held web chat turn is the operator's to answer: approve it in the browser or terminal",
        ));
    }
    let now = call.invocation.at.unix_ms();
    let Some(action) = call.reader()?.chat_action(id)? else {
        return Ok(super::super::browser_refusal(actions::Refused(
            polylogue_model::Code::UnknownChatAction,
        )));
    };
    if action.status(now).0 != ChatStatus::AwaitingApproval {
        let decided = action
            .events
            .iter()
            .find(|event| matches!(event.kind.as_str(), "approved" | "declined"));
        let message = match decided {
            Some(event) => format!(
                "already_answered: the held turn {id} was {} by {}",
                if event.kind == "approved" {
                    "allowed"
                } else {
                    "denied"
                },
                event.by.as_deref().unwrap_or("someone")
            ),
            None => format!("the chat action {id} is not held for an answer"),
        };
        return Ok(error(polylogue_model::Code::Conflict, message));
    }
    Ok(chat::chat_then_view(
        call.daemon,
        call.invocation,
        UserWrite::ChatDecide {
            action: id.to_owned(),
            approve,
            by: call.invocation.authorship.author.to_string(),
            detail: reason,
        },
    ))
}
