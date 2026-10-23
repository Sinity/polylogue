//! Stateless continuation of a session read on one admitted Reader.
use super::{Response, error, ok, store_error};
use base64::Engine as _;
use polylogue_model::SessionId;
use polylogue_render::model_pages::{Pager, Position};
use polylogue_store::Reader;
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

const CURSOR_VERSION: u8 = 2;

pub(super) struct Options<'a> {
    pub view: &'a str,
    pub format: &'a str,
    pub logical: bool,
    pub budget: Option<usize>,
    pub cursor: Option<&'a str>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    version: u8,
    binding: String,
    content: String,
    generation: String,
    position: [usize; 3],
}

pub(super) fn run(reader: &Reader<'_>, id: &SessionId, options: Options<'_>) -> Response {
    let Some(view) = polylogue_render::View::parse(options.view) else {
        return error("bad_request", format!("unknown view {:?}", options.view));
    };
    let session = match reader.session(id) {
        Ok(Some(session)) => session,
        Ok(None) => return error("not_found", format!("no session {id}")),
        Err(e) => return store_error(&e),
    };
    let title = match reader.display_title(id) {
        Ok(title) => title,
        Err(e) => return store_error(&e),
    };
    let selected = match reader.select_messages(id, options.logical) {
        Ok(selected) => selected,
        Err(e) => return store_error(&e),
    };
    let notices: Vec<String> = selected
        .lineage()
        .issues
        .iter()
        .map(|issue| format!("Incomplete session ancestry: {issue}"))
        .collect();
    let Some(budget) = options.budget else {
        if options.cursor.is_some() {
            return error("bad_request", "a cursor continues a read with a budget");
        }
        let Some(format) = polylogue_render::Format::parse(options.format) else {
            return error(
                "bad_request",
                format!("unknown format {:?}", options.format),
            );
        };
        let mut messages = Vec::new();
        if let Err(e) = selected.visit(|message| {
            messages.push(message);
            ControlFlow::Continue(())
        }) {
            return store_error(&e);
        }
        return match polylogue_render::session(
            &session,
            title.as_deref(),
            messages,
            view,
            format,
            &notices,
        ) {
            Ok(mut value) => {
                if options.logical {
                    value["lineage"] =
                        serde_json::to_value(selected.lineage()).expect("lineage is serializable");
                }
                ok(value)
            }
            Err(e) => error("internal", e),
        };
    };
    let digest = |value: &serde_json::Value| {
        blake3::hash(value.to_string().as_bytes())
            .to_hex()
            .to_string()
    };
    // Tokens contain hashes and numeric positions, never private source text or
    // the masking state reconstructed by replay below.
    let binding = digest(&serde_json::json!([id, options.logical, options.view]));
    let facts = match selected.fingerprint() {
        Ok(facts) => facts,
        Err(e) => return store_error(&e),
    };
    let content = digest(&serde_json::json!([facts, session, title]));
    let generation = match reader.change_position() {
        Ok(position) => position.generation,
        Err(e) => return store_error(&e),
    };
    let position = if let Some(text) = options.cursor {
        let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(text)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Cursor>(&bytes).ok());
        let Some(cursor) = decoded.filter(|cursor| cursor.version == CURSOR_VERSION) else {
            return error(
                "bad_request",
                "the read cursor is invalid or from an unsupported version",
            );
        };
        if cursor.binding != binding {
            return error(
                "bad_request",
                "the read cursor belongs to a different session, logical mode or view",
            );
        }
        if cursor.content != content || cursor.generation != generation {
            return error(
                "stale_cursor",
                "the session read changed; start the read again",
            );
        }
        Some(Position {
            message: cursor.position[0],
            piece: cursor.position[1],
            character: cursor.position[2],
        })
    } else {
        None
    };
    let mut pager = Pager::new(view, budget, position);
    let mut failure = None;
    let result = selected.visit(|message| match pager.push(message) {
        Ok(flow) => flow,
        Err(e) => {
            failure = Some(e);
            ControlFlow::Break(())
        }
    });
    if let Err(e) = result {
        return store_error(&e);
    }
    if let Some(e) = failure {
        return error("bad_request", e);
    }
    let page = match pager.finish() {
        Ok(page) => page,
        Err(e) => return error("bad_request", e),
    };
    let next = match page.next {
        Some(position) => {
            let cursor = Cursor {
                version: CURSOR_VERSION,
                binding,
                content,
                generation,
                position: [position.message, position.piece, position.character],
            };
            match serde_json::to_vec(&cursor) {
                Ok(bytes) => Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)),
                Err(e) => return error("internal", e),
            }
        }
        None => None,
    };
    let mut value = serde_json::json!({"session": session, "display_title": title, "text": page.text, "next": next});
    if options.logical {
        value["lineage"] =
            serde_json::to_value(selected.lineage()).expect("lineage is serializable");
    }
    ok(value)
}
