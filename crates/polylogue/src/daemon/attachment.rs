//! Exact retained attachment bytes cross only the observed interactive CLI
//! socket boundary. Ordinary application calls return masked descriptors.
use super::service::{Daemon, error, store_error};
use super::{Request, Response};
use base64::Engine as _;
use serde::{Deserialize, Serialize};

mod operator;
pub(crate) use operator::check_cli;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BytePage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor: Option<serde_json::Value>,
    pub asset: String,
    pub blob: polylogue_model::BlobHash,
    pub size: u64,
    pub offset: u64,
    pub content_base64: String,
    pub next: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    version: u8,
    binding: String,
    generation: String,
    asset: String,
    offset: u64,
    blob: polylogue_model::BlobHash,
}

pub(super) fn socket(
    stream: &std::os::unix::net::UnixStream,
    request: Request,
    d: &Daemon,
) -> Response {
    if let Err(response) = super::service::admit(d) {
        return response;
    }
    match operator::Operator::admit(stream) {
        Ok(operator) => reveal(request, d, operator),
        Err(e) => error("operator_required", e),
    }
}

fn reveal(request: Request, d: &Daemon, _operator: operator::Operator) -> Response {
    let Request::Attachment {
        message,
        block,
        attachment,
        asset,
        length,
        cursor,
        ..
    } = request
    else {
        return error("bad_request", "not an attachment request");
    };
    if length == 0 {
        return error("bad_request", "attachment range length must be positive");
    }
    let reader = match d.reads.reader() {
        Ok(reader) => reader,
        Err(e) => return store_error(&e),
    };
    let resolution = match reader.attachment(&message, block, attachment) {
        Ok(value) => value,
        Err(e) => return store_error(&e),
    };
    let generation = match reader.change_position() {
        Ok(position) => position.generation,
        Err(e) => return store_error(&e),
    };
    let initial = cursor.is_none();
    let decoded = match cursor {
        Some(cursor) => {
            let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
                .decode(cursor)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<Cursor>(&bytes).ok());
            let Some(decoded) = decoded.filter(|cursor| cursor.version == 1) else {
                return error("bad_request", "invalid attachment cursor");
            };
            if decoded.binding != resolution.binding || decoded.generation != generation {
                return error(
                    "stale_cursor",
                    "the selected attachment changed; start again",
                );
            }
            Some(decoded)
        }
        None => None,
    };
    let chosen = asset
        .as_deref()
        .or_else(|| decoded.as_ref().map(|cursor| cursor.asset.as_str()));
    let candidate = match chosen {
        Some(input) => resolution
            .candidates
            .iter()
            .find(|candidate| candidate.input == input),
        None if resolution.candidates.len() == 1 => resolution.candidates.first(),
        None => None,
    };
    let Some(candidate) = candidate else {
        return error(
            "attachment_unavailable",
            "select an available retained asset from the attachment descriptor",
        );
    };
    if decoded
        .as_ref()
        .is_some_and(|cursor| cursor.asset != candidate.input)
    {
        return error("bad_request", "the cursor selects another asset");
    }
    let offset = decoded.as_ref().map_or(0, |cursor| cursor.offset);
    let blob = match decoded
        .as_ref()
        .map(|cursor| cursor.blob)
        .or(candidate.blob)
    {
        Some(hash) => hash,
        None => match reader.attachment_hash(candidate, &d.config.blob_roots) {
            Ok(hash) => hash,
            Err(error) => return store_error(&error),
        },
    };
    if candidate.blob.is_some_and(|known| known != blob) {
        return error("stale_cursor", "the selected attachment hash changed");
    }
    if offset > candidate.size || offset == candidate.size && offset != 0 {
        return error(
            "bad_request",
            "attachment cursor is outside the retained bytes",
        );
    }
    let end = offset.saturating_add(length as u64).min(candidate.size);
    let bytes = match reader.attachment_range(candidate, &d.config.blob_roots, offset..end) {
        Ok(bytes) => bytes,
        Err(e) => return store_error(&e),
    };
    let next = if end < candidate.size {
        match serde_json::to_vec(&Cursor {
            version: 1,
            binding: resolution.binding.clone(),
            generation,
            asset: candidate.input.clone(),
            offset: end,
            blob,
        }) {
            Ok(bytes) => Some(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)),
            Err(e) => return error("internal", e),
        }
    } else {
        None
    };
    let descriptor = if initial {
        let masked = match reader.masked_attachment(&message, block, attachment) {
            Ok(value) => value,
            Err(e) => return store_error(&e),
        };
        let mut value = match serde_json::to_value(masked) {
            Ok(value) => value,
            Err(e) => return error("internal", e),
        };
        polylogue_render::mask_json(&mut value);
        Some(value)
    } else {
        None
    };
    let page = BytePage {
        descriptor,
        asset: candidate.input.clone(),
        blob,
        size: candidate.size,
        offset,
        content_base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        next,
    };
    // This typed byte payload is encoded after operator admission. Its descriptor
    // is masked separately; encoded bytes must never enter the text masker.
    match serde_json::to_value(page) {
        Ok(data) => Response::Ok { data },
        Err(e) => error("internal", e),
    }
}
