//! A rollout's recorded identity, independent of its location or encoding.
/// Version of the rows this module produces.
pub const VERSION: u32 = 1;

parser_modules! {}

use polylogue_model::Json;
use polylogue_model::JsonExt;
use std::io::BufRead;

/// Reads the first nonempty record. Only a rollout header establishes
/// identity; file names and conversation text do not.
pub fn rollout_id(mut reader: impl BufRead) -> std::io::Result<Option<String>> {
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok(None);
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let Ok(value) = serde_json::from_slice::<Json>(&line) else {
            return Ok(None);
        };
        // A mid-2025 header has no type and names the session at the top level.
        let id = match value.str("type").filter(|t| !t.is_empty()) {
            Some("session_meta") => value.get("payload").and_then(|p| p.str("id")),
            None => value.str("id"),
            Some(_) => return Ok(None),
        };
        return Ok(id.filter(|id| !id.is_empty()).map(str::to_owned));
    }
}

/// The working directory a rollout record states for what follows it: a
/// session header's or a turn context's. A mid-2025 header has no type and
/// states it at the top level.
pub fn recorded_directory(record: &Json) -> Option<&str> {
    let holder = match record.str("type").filter(|t| !t.is_empty()) {
        Some("session_meta" | "turn_context") => record.get("payload")?,
        None if record.get("id").is_some() => record,
        _ => return None,
    };
    holder.str("cwd").filter(|cwd| !cwd.is_empty())
}
