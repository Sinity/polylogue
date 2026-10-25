//! An explicit browser selection becomes an ordinary operator note.
use super::Ctx;
use crate::daemon::{Request, Response};
use polylogue_model::{Author, MessageId, NewNote, NoteKind, Target};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    provider: String,
    conversation_id: String,
    message_id: String,
    page_url: String,
    selected_at: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SelectionNote {
    request_id: String,
    body: String,
    quote: String,
    source: Source,
}

fn invalid(message: &str) -> Response {
    crate::daemon::service::error("bad_request", message)
}

pub(super) fn save(ctx: &Ctx, input: SelectionNote) -> Response {
    let source = &input.source;
    if uuid::Uuid::parse_str(&input.request_id).is_err()
        || input.body.trim().is_empty()
        || input.quote.trim().is_empty()
        || source.conversation_id.is_empty()
        || source.message_id.is_empty()
        || source.selected_at.parse::<jiff::Timestamp>().is_err()
    {
        return invalid("the note needs a draft ID, a body, a selection and its source");
    }
    let Ok(url) = source.page_url.parse::<axum::http::Uri>() else {
        return invalid("the source page URL is invalid");
    };
    let Some(chat) = polylogue_sources::origin::by_extension_name(&source.provider)
        .and_then(|origin| origin.web.as_ref())
    else {
        return invalid("this provider has no selection-note target");
    };
    let parts: Vec<_> = url.path().split('/').filter(|s| !s.is_empty()).collect();
    if url.scheme_str() != Some("https")
        || !url
            .authority()
            .is_some_and(|a| chat.hosts.contains(&a.as_str()))
        || parts.len() != 2
        || parts[0] != chat.path
        || parts[1] != source.conversation_id
    {
        return invalid("the source page does not identify the supplied provider conversation");
    }
    let Some(session) =
        polylogue_sources::browser_capture::session_for(&source.provider, &source.conversation_id)
    else {
        return invalid("the supplied conversation ID is invalid");
    };
    let origin = polylogue_sources::browser_capture::origin_of(&source.provider);
    let Ok(message) = MessageId::parse(&format!("{origin}:{}", source.message_id)) else {
        return invalid("the supplied message ID is invalid");
    };
    // Query globally before accepting the supplied conversation. A native ID
    // shared by multiple sessions must not silently choose one occurrence.
    let query = Request::Query {
        text: format!(
            "messages where id:{} active:* duplicate:* hidden:*",
            serde_json::to_string(&message.to_string()).expect("string")
        ),
        cursor: None,
        page_size: Some(2),
        why: false,
        session: None,
        fields: vec!["id".into(), "session".into()],
    };
    let data = match ctx.ask(&query) {
        Some(Response::Ok { data }) => data,
        Some(error) => return error,
        None => {
            return crate::daemon::service::error(
                "daemon_required",
                "the daemon is unavailable; the note was not acknowledged",
            );
        }
    };
    let row = &data["rows"][0];
    if data["count"].as_u64() != Some(1)
        || row["id"].as_str() != Some(&message.to_string())
        || row["session"].as_str() != Some(&session.to_string())
    {
        return crate::daemon::service::error(
            "conflict",
            "the supplied message is missing, ambiguous or belongs to another conversation; capture the conversation and retry",
        );
    }
    let note = NewNote {
        kind: NoteKind::Note,
        key: None,
        value: Some(
            json!({"body":input.body,"quote":input.quote,"source":input.source}).to_string(),
        ),
        author: Author::Operator,
        targets: vec![Target::Session(session), Target::Message(message)],
        relations: vec![],
        revises: None,
        valid_until: None,
    };
    ctx.ask(&Request::AddNote {
        request_id: input.request_id,
        note,
    })
    .unwrap_or_else(|| {
        crate::daemon::service::error(
            "daemon_required",
            "save outcome unknown; retry this draft with the same ID",
        )
    })
}
