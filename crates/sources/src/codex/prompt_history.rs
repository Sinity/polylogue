//! Prompt-history records prove references and prompt bytes only. They never
//! establish a conversation boundary, assistant response, or complete history.
use serde::Serialize;
use serde_json::Value;
#[derive(Debug, Serialize)]
pub struct Reference {
    pub native_session: String,
    pub start: u64,
    pub end: u64,
}
#[derive(Debug, Serialize)]
pub struct Projection {
    pub references: Vec<Reference>,
    pub malformed_records: u64,
    pub incomplete_tail: bool,
}
pub fn project(bytes: &[u8]) -> Result<Projection, &'static str> {
    if bytes.len() > 1024 * 1024 {
        return Err("prompt history exceeds 1 MiB");
    }
    let mut result = Projection {
        references: Vec::new(),
        malformed_records: 0,
        incomplete_tail: false,
    };
    let mut offset = 0;
    for line in bytes.split_inclusive(|b| *b == b'\n') {
        let start = offset;
        offset += line.len();
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if !line.ends_with(b"\n") {
            result.incomplete_tail = true;
            break;
        }
        if result.references.len() >= 4096 {
            return Err("prompt history exceeds 4096 references");
        }
        let value = serde_json::from_slice::<Value>(line);
        match value {
            Ok(v)
                if v["session_id"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty() && s.len() <= 4096)
                    && v["text"].is_string() =>
            {
                result.references.push(Reference {
                    native_session: v["session_id"].as_str().unwrap().into(),
                    start: start as u64,
                    end: offset as u64,
                });
            }
            _ => result.malformed_records += 1,
        }
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn references_do_not_invent_conversation_and_keep_raw_coordinates() {
        let a = b"{\"session_id\":\"opaque:one\",\"text\":\"a prompt\",\"ts\":123}\n";
        let mut bytes = a.to_vec();
        bytes.extend_from_slice(b"bad\n{\"partial\":");
        let projection = project(&bytes).unwrap();
        assert_eq!(projection.references.len(), 1);
        assert_eq!(projection.references[0].end, a.len() as u64);
        assert_eq!(projection.references[0].native_session, "opaque:one");
        assert_eq!(projection.malformed_records, 1);
        assert!(projection.incomplete_tail);
        assert!(project(b"").unwrap().references.is_empty());
    }
}
