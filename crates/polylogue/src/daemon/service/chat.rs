//! Chat action requests: the operator queues, the extension claims and
//! reports, through the one writer ([`polylogue_store::chat_actions`]).

use super::{Daemon, browser_refusal, error, machine_id, ok, store_error, writer};
use crate::browser::actions::{self, Refused};
use crate::daemon::{Request, Response};
use base64::Engine as _;
use polylogue_store::{Reader, StoreError, UserRequest, UserWrite, user_writes};

fn refused(e: &StoreError) -> Response {
    match e {
        StoreError::ChatAction(code) => browser_refusal(actions::refusal(code)),
        other => store_error(other),
    }
}

fn enabled(reader: &Reader<'_>) -> Result<bool, Response> {
    reader
        .setting("browser.actions", &machine_id())
        .map(|value| value.as_deref() == Some("on"))
        .map_err(|e| store_error(&e))
}

/// Writes `write`, then answers with the action it names.
fn write_then_view(d: &Daemon, write: UserWrite) -> Response {
    let writes = match writer(d) {
        Ok(writes) => writes,
        Err(response) => return response,
    };
    let outcome = match writes.user(UserRequest::now(user_writes::new_request_id(), write)) {
        Ok(outcome) => outcome,
        Err(StoreError::NotFound(_)) => return ok(serde_json::json!({ "action": null })),
        Err(e) => return refused(&e),
    };
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    match reader.chat_action(&outcome.id) {
        Ok(Some(a)) => ok(serde_json::json!({
            "created": !outcome.repeat,
            "action": actions::view(&a, super::now_ms()),
        })),
        Ok(None) => browser_refusal(actions::refusal("unknown_chat_action")),
        Err(e) => store_error(&e),
    }
}

pub(super) fn handle(request: Request, d: &Daemon, reader: Reader<'_>) -> Response {
    let now = super::now_ms();
    match request {
        Request::ChatEnqueue { action } => {
            match enabled(&reader) {
                Ok(true) => {}
                Ok(false) => return browser_refusal(Refused(409, "browser_actions_off")),
                Err(response) => return response,
            }
            let new = actions::new_action(&action, |blob| reader.blob_size(blob).ok().flatten());
            drop(reader);
            match new {
                Ok(new) => write_then_view(d, UserWrite::ChatEnqueue(new)),
                Err(r) => browser_refusal(r),
            }
        }
        Request::ChatUpload { bytes } => {
            drop(reader);
            let writes = match writer(d) {
                Ok(writes) => writes,
                Err(response) => return response,
            };
            let stored = d
                .archive
                .prepare_piece(&d.config.blob_roots, &bytes)
                .and_then(|(piece, file)| writes.store_blob(piece, file));
            match stored {
                Ok(hash) => ok(serde_json::json!({
                    "ok": true,
                    "attachment_ref": hash.to_string(),
                    "size_bytes": bytes.len(),
                })),
                Err(e) => store_error(&e),
            }
        }
        Request::ChatActions => match reader.chat_actions() {
            Ok(list) => ok(serde_json::json!({
                "actions": list.iter().map(|a| actions::view(a, now)).collect::<Vec<_>>(),
            })),
            Err(e) => store_error(&e),
        },
        Request::ChatAction { id } => match reader.chat_action(&id) {
            Ok(Some(a)) => ok(serde_json::json!({ "action": actions::view(&a, now) })),
            Ok(None) => browser_refusal(actions::refusal("unknown_chat_action")),
            Err(e) => store_error(&e),
        },
        Request::ChatClaim { by } => {
            let on = match enabled(&reader) {
                Ok(on) => on,
                Err(response) => return response,
            };
            drop(reader);
            if !on {
                return ok(serde_json::json!({ "enabled": false, "action": null }));
            }
            write_then_view(d, UserWrite::ChatClaim { by })
        }
        Request::ChatReport { id, update } => {
            drop(reader);
            write_then_view(
                d,
                UserWrite::ChatReport {
                    action: id,
                    by: update.owner_instance_id,
                    outcome: update.outcome,
                    phase: update.phase,
                    detail: update.detail,
                    receipt: update.receipt,
                },
            )
        }
        Request::ChatDecide {
            id,
            approve,
            by,
            detail,
        } => {
            drop(reader);
            write_then_view(
                d,
                UserWrite::ChatDecide {
                    action: id,
                    approve,
                    by,
                    detail,
                },
            )
        }
        Request::ChatAttachment { id, attachment } => {
            match reader.chat_attachment(&id, &attachment, &d.config.blob_roots) {
                Ok((a, bytes)) => ok(serde_json::json!({
                    "media_type": a.media_type,
                    "bytes": base64::engine::general_purpose::STANDARD.encode(bytes),
                })),
                Err(e) => refused(&e),
            }
        }
        other => error(
            "bad_request",
            format!("{other:?} is not a chat action request"),
        ),
    }
}
