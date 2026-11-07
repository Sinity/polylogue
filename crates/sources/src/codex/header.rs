//! A rollout's recorded identity, independent of its location or encoding.
use polylogue_model::Json;
use std::io::BufRead;

/// Reads the first nonempty record. Only a native rollout header establishes
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
        if !matches!(
            value.get("type").and_then(Json::as_str),
            Some("session_meta" | "older_header")
        ) {
            return Ok(None);
        }
        return Ok(value
            .get("payload")
            .and_then(|p| p.get("id"))
            .and_then(Json::as_str)
            .filter(|id| !id.is_empty())
            .map(str::to_owned));
    }
}
