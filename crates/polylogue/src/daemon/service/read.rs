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

/// Document pages retain only metadata, one decoded block and the output page.
pub(super) fn document(
    reader: &Reader<'_>,
    id: &polylogue_model::DocumentId,
    options: Options<'_>,
) -> Response {
    let Some(view) = polylogue_render::View::parse(options.view) else {
        return error("bad_request", "unknown document view");
    };
    if matches!(
        view,
        polylogue_render::View::Prompts | polylogue_render::View::Actions
    ) {
        return error(
            "bad_request",
            "documents support the full and dialogue views",
        );
    }
    let selected = match reader.select_document(id) {
        Ok(Some(selected)) => selected,
        Ok(None) => return error("not_found", format!("no document {id}")),
        Err(e) => return store_error(&e),
    };
    let Some(budget) = options.budget else {
        if options.cursor.is_some() {
            return error("bad_request", "a cursor continues a read with a budget");
        }
        let Some(format) = polylogue_render::Format::parse(options.format) else {
            return error("bad_request", "unknown document format");
        };
        return match selected.document().and_then(|doc| {
            polylogue_render::documents::document(doc, view, format).map_err(Into::into)
        }) {
            Ok(mut value) => {
                match selected.sources() {
                    Ok(sources) => value["sources"] = serde_json::json!(sources),
                    Err(e) => return store_error(&e),
                };
                ok(value)
            }
            Err(e) => store_error(&e),
        };
    };
    let digest = |value: &serde_json::Value| {
        blake3::hash(value.to_string().as_bytes())
            .to_hex()
            .to_string()
    };
    let binding = digest(&serde_json::json!(["document", id, options.view]));
    let content = selected.version();
    let generation = match reader.change_position() {
        Ok(position) => position.generation,
        Err(e) => return store_error(&e),
    };
    let position = match content_position(&options, &binding, &content, &generation) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let heading = format!("[document:{} {}]\n", id, selected.purpose().as_str());
    let mut pager = polylogue_render::model_pages::BlockPager::new(view, budget, position, heading);
    let mut failure = None;
    if let Err(e) = selected.visit_blocks(|block| match pager.push(block) {
        Ok(flow) => flow,
        Err(e) => {
            failure = Some(e);
            ControlFlow::Break(())
        }
    }) {
        return store_error(&e);
    }
    if let Some(e) = failure {
        return error("bad_request", e);
    }
    let page = match pager.finish() {
        Ok(page) => page,
        Err(e) => return error("bad_request", e),
    };
    let next = content_next(page.next, binding, content, generation);
    let metadata = match selected.metadata() {
        Ok(value) => value,
        Err(e) => return store_error(&e),
    };
    ok(serde_json::json!({"document":metadata,"text":page.text,"next":next}))
}

fn content_position(
    options: &Options<'_>,
    binding: &str,
    content: &str,
    generation: &str,
) -> Result<Option<Position>, Response> {
    let Some(text) = options.cursor else {
        return Ok(None);
    };
    let cursor = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(text)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Cursor>(&bytes).ok());
    let Some(cursor) = cursor.filter(|c| c.version == CURSOR_VERSION) else {
        return Err(error("bad_request", "invalid or unsupported read cursor"));
    };
    if cursor.binding != binding {
        return Err(error(
            "bad_request",
            "the read cursor belongs to another content or view",
        ));
    }
    if cursor.content != content || cursor.generation != generation {
        return Err(error(
            "stale_cursor",
            "the content read changed; start the read again",
        ));
    }
    Ok(Some(Position {
        message: cursor.position[0],
        piece: cursor.position[1],
        character: cursor.position[2],
    }))
}
fn content_next(
    next: Option<Position>,
    binding: String,
    content: String,
    generation: String,
) -> Option<String> {
    next.map(|p| {
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(
            serde_json::to_vec(&Cursor {
                version: CURSOR_VERSION,
                binding,
                content,
                generation,
                position: [p.message, p.piece, p.character],
            })
            .expect("cursor is serializable"),
        )
    })
}

pub(super) fn event(reader: &Reader<'_>, id: &str, options: Options<'_>) -> Response {
    let Some(view) = polylogue_render::View::parse(options.view) else {
        return error("bad_request", "unknown event view");
    };
    if matches!(
        view,
        polylogue_render::View::Prompts | polylogue_render::View::Actions
    ) {
        return error("bad_request", "events support the full and dialogue views");
    }
    let Some(format) = polylogue_render::Format::parse(options.format) else {
        return error("bad_request", "unknown event format");
    };
    let exported = match reader.event_export(id) {
        Ok(Some(event)) => event,
        Ok(None) => {
            return if options.cursor.is_some() {
                error("stale_cursor", "the event is no longer available")
            } else {
                error("not_found", format!("no event {id}"))
            };
        }
        Err(e) => return store_error(&e),
    };
    let sources = match reader.event_sources(id) {
        Ok(s) => s,
        Err(e) => return store_error(&e),
    };
    let metadata = serde_json::json!({"id":exported.id,"input":exported.input,"source":exported.source,"source_chain":exported.source_chain,"source_start":exported.source_start,"source_end":exported.source_end,
        "session":exported.event.session,"kind":exported.event.kind,"at":exported.event.at,"after":exported.event.after});
    let Some(budget) = options.budget else {
        if options.cursor.is_some() {
            return error("bad_request", "a cursor continues a read with a budget");
        }
        return match polylogue_render::events::event(exported.event, format) {
            Ok(mut value) => {
                value["reference"] = metadata;
                value["sources"] = serde_json::json!(sources);
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
    let binding = digest(&serde_json::json!(["event", id, options.view]));
    let content = digest(&serde_json::json!([metadata, sources]));
    let generation = match reader.change_position() {
        Ok(p) => p.generation,
        Err(e) => return store_error(&e),
    };
    let position = match content_position(&options, &binding, &content, &generation) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let heading = format!(
        "[event:{id} {}]\n",
        polylogue_model::masking::mask(&exported.event.kind)
    );
    let mut pager = polylogue_render::model_pages::BlockPager::new(view, budget, position, heading);
    if let Err(e) = pager.push(polylogue_model::Block::Text {
        text: polylogue_render::events::text(&exported.event),
    }) {
        return error("bad_request", e);
    }
    let page = match pager.finish() {
        Ok(p) => p,
        Err(e) => return error("bad_request", e),
    };
    let next = content_next(page.next, binding, content, generation);
    ok(serde_json::json!({"event":metadata,"sources":sources,"text":page.text,"next":next}))
}
