//! Metadata outline over the same store message selection.
use super::super::{Response,error,store_error};
use polylogue_store::Reader;
use polylogue_model::SessionId;
use serde::{Serialize,Deserialize};
use base64::Engine as _;
const CURSOR_VERSION:u8=6;
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cursor {
    version: u8,
    binding: String,
    content: String,
    generation: String,
    position: [usize; 3],
    /// The selected count is bound to the same content as the cursor.
    total_messages: Option<usize>,
    /// The key of the message the position is in, so the next read starts
    /// there without reading the messages before it. Absent for documents
    /// and events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    message_key: Option<i64>,
    /// Read the page before this exclusive position.
    #[serde(default)]
    reverse: bool,
    /// A keyed hash of the rest, using the index generation's cursor key.
    seal: String,
}

impl Cursor {
    fn seal(mut self, key: &[u8; 32]) -> Result<String, serde_json::Error> {
        self.seal = String::new();
        self.seal = blake3::keyed_hash(key, &serde_json::to_vec(&self)?)
            .to_hex()
            .to_string();
        Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(serde_json::to_vec(&self)?))
    }

    /// The cursor in `token`, when this index generation sealed it. One
    /// from an earlier generation is stale (its key is gone); anything else
    /// that does not open is invalid.
    fn open(token: &str, key: &[u8; 32], generation: &str) -> Result<Self, Response> {
        let invalid = || {
            error(
                "bad_request",
                "the read cursor is invalid or from an unsupported version",
            )
        };
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(token)
            .map_err(|_| invalid())?;
        let mut cursor: Self = serde_json::from_slice(&bytes).map_err(|_| invalid())?;
        if cursor.version != CURSOR_VERSION {
            return Err(invalid());
        }
        let seal =
            blake3::Hash::from_hex(std::mem::take(&mut cursor.seal)).map_err(|_| invalid())?;
        let body = serde_json::to_vec(&cursor).map_err(|_| invalid())?;
        if seal == blake3::keyed_hash(key, &body) {
            Ok(cursor)
        } else if cursor.generation != generation {
            Err(error(
                "stale_cursor",
                "the index was rebuilt; start the read again",
            ))
        } else {
            Err(invalid())
        }
    }
}

pub(super) fn run(
    reader: &Reader<'_>,
    id: &SessionId,
    with_parents: bool,
    size: usize,
    cursor: Option<&str>,
) -> Response {
    if size == 0 {
        return error("bad_request", "outline page size must be positive");
    }
    match reader.session(id) {
        Ok(Some(_)) => {}
        Ok(None) => return error("not_found", "the session is not available"),
        Err(e) => return store_error(&e),
    }
    let selected = match reader.select_messages(id, with_parents) {
        Ok(s) => s,
        Err(e) => return store_error(&e),
    };
    let content = match selected.fingerprint() {
        Ok(s) => s,
        Err(e) => return store_error(&e),
    };
    let key = match reader.cursor_key() {
        Ok(s) => s,
        Err(e) => return store_error(&e),
    };
    let generation = match reader.change_position() {
        Ok(s) => s.generation,
        Err(e) => return store_error(&e),
    };
    let binding = blake3::hash(
        serde_json::json!(["outline", id, with_parents])
            .to_string()
            .as_bytes(),
    )
    .to_hex()
    .to_string();
    let from = if let Some(token) = cursor {
        let cursor = match Cursor::open(token, &key, &generation) {
            Ok(c) => c,
            Err(e) => return e,
        };
        if cursor.binding != binding {
            return error(
                "bad_request",
                "the outline cursor belongs to another selection",
            );
        }
        if cursor.content != content {
            return error("stale_cursor", "the outline changed; open it again");
        }
        cursor.message_key
    } else {
        None
    };
    let mut entries = match selected.outline(from, size.saturating_add(1)) {
        Ok(s) => s,
        Err(e) => return store_error(&e),
    };
    let more = entries.len() > size;
    if more {
        entries.pop();
    }
    let next = if more {
        Some(
            Cursor {
                version: CURSOR_VERSION,
                total_messages: None,
                binding,
                content,
                generation,
                position: [0; 3],
                message_key: entries.last().map(|entry| entry.key),
                seal: String::new(),
                reverse: false,
            }
            .seal(&key)
            .expect("cursor"),
        )
    } else {
        None
    };
    super::super::ok(serde_json::json!({"entries":entries,"next":next,"parents":selected.parents()}))
}
