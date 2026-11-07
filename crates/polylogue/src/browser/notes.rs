//! An explicit browser selection becomes an ordinary operator note.
use super::BrowserHost;
use crate::daemon::{Failure, Request};
use command_surface::Answer;
use polylogue_model::{MessageId, Reference};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Source {
    pub(crate) provider: String,
    pub(crate) conversation_id: String,
    pub(crate) message_id: String,
    pub(crate) page_url: String,
    pub(crate) selected_at: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SelectionNote {
    pub(crate) request_id: String,
    pub(crate) body: String,
    pub(crate) quote: String,
    pub(crate) source: Source,
}

fn invalid(message: &str) -> Failure {
    Failure::new("bad_request", message)
}

pub(super) fn save(host: &BrowserHost, input: SelectionNote) -> Answer<Value> {
    if let Err(failure) = input.references() {
        return failure.answer();
    }
    host.ask(&Request::BrowserSelectionNote {
        request_id: input.request_id,
        body: input.body,
        quote: input.quote,
        provider: input.source.provider,
        conversation_id: input.source.conversation_id,
        message_id: input.source.message_id,
        page_url: input.source.page_url,
        selected_at: input.source.selected_at,
    })
    .unwrap_or_else(|| {
        crate::daemon::service::error(
            "daemon_required",
            "save outcome unknown; retry this draft with the same ID",
        )
    })
}

impl SelectionNote {
    pub(crate) fn references(&self) -> Result<(Reference, MessageId), Failure> {
        let source = &self.source;
        if uuid::Uuid::parse_str(&self.request_id).is_err()
            || self.body.trim().is_empty()
            || self.quote.trim().is_empty()
            || source.conversation_id.is_empty()
            || source.message_id.is_empty()
            || source.selected_at.parse::<jiff::Timestamp>().is_err()
        {
            return Err(invalid(
                "the note needs a draft ID, a body, a selection and its source",
            ));
        }
        let Ok(url) = source.page_url.parse::<axum::http::Uri>() else {
            return Err(invalid("the source page URL is invalid"));
        };
        let Some(chat) = polylogue_sources::origin::by_extension_name(&source.provider)
            .and_then(|origin| origin.web.as_ref())
        else {
            return Err(invalid("this provider has no selection-note target"));
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
            return Err(invalid(
                "the source page does not identify the supplied provider conversation",
            ));
        }
        let Some(session) = polylogue_sources::browser_capture::session_for(
            &source.provider,
            &source.conversation_id,
        ) else {
            return Err(invalid("the supplied conversation ID is invalid"));
        };
        let origin = polylogue_sources::browser_capture::origin_of(&source.provider);
        let Ok(message) = MessageId::parse(&format!("{origin}:{}", source.message_id)) else {
            return Err(invalid("the supplied message ID is invalid"));
        };
        Ok((Reference::Session(session), message))
    }
    pub(crate) fn assertion(
        &self,
        session: Reference,
        message: Reference,
    ) -> polylogue_model::assertion::NewAssertion {
        use polylogue_model::assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink};
        NewAssertion {
            kind: AssertionKind::Note,
            key: None,
            value: Some(serde_json::Value::String(format!(
                "{}\n\nQuoted selection:\n{}\n\nSource: {}\nSelected at: {}",
                self.body, self.quote, self.source.page_url, self.source.selected_at
            ))),
            valid_until: None,
            links: [session, message]
                .into_iter()
                .map(|reference| NewAssertionLink {
                    role: LinkRole::About,
                    reference,
                })
                .collect(),
        }
    }
}
