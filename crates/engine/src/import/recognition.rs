//! Declared structures, inspected through the complete JSON stream. Field
//! order and a long first value do not affect recognition.
use super::*;
use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, SeqAccess, Visitor};
use std::collections::BTreeSet;

pub(super) const STRUCTURES: &[&str] = &[
    "ChatGPT export",
    "ChatGPT conversation",
    "claude.ai export",
    "claude.ai conversation",
    "Grok account export",
    "Google Takeout",
    "AI Studio prompt",
    "OpenRouter room",
    "OpenTelemetry GenAI",
    "Polylogue bundle",
    "live tool layout",
    "explicit format",
];

#[derive(Default)]
struct Shape {
    keys: BTreeSet<String>,
}
impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ShapeVisitor;
        impl<'de> Visitor<'de> for ShapeVisitor {
            type Value = Shape;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object or array")
            }
            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Shape, M::Error> {
                let mut shape = Shape::default();
                while let Some(key) = map.next_key::<String>()? {
                    if [
                        "mapping",
                        "current_node",
                        "conversation_id",
                        "id",
                        "uuid",
                        "chat_messages",
                        "chunkedPrompt",
                        "resourceSpans",
                        "version",
                        "messages",
                        "characters",
                        "title",
                        "conversations",
                    ]
                    .contains(&key.as_str())
                    {
                        shape.keys.insert(key);
                    }
                    map.next_value::<IgnoredAny>()?;
                }
                Ok(shape)
            }
            fn visit_seq<S: SeqAccess<'de>>(self, mut seq: S) -> Result<Shape, S::Error> {
                let shape = seq.next_element::<Shape>()?.unwrap_or_default();
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(shape)
            }
        }
        d.deserialize_any(ShapeVisitor)
    }
}

pub(super) fn declared(
    path: &Path,
    file: &mut std::fs::File,
    format: Option<&str>,
) -> std::io::Result<Option<&'static OriginSpec>> {
    use std::io::{Read, Seek};
    file.rewind()?;
    let mut head = [0; 64];
    let n = file.read(&mut head)?;
    file.rewind()?;
    if let Some(format) = format {
        return Ok(polylogue_sources::origin::by_name(format).filter(|o| {
            o.support == Support::Parsed
                && ((o.recognize)(path, &head[..n]) || o.arrival == Arrival::Import)
        }));
    }
    if polylogue_sources::polylogue_export::recognize_reader(&mut *file)? {
        file.rewind()?;
        return Ok(polylogue_sources::origin::by_name("polylogue-export"));
    }
    file.rewind()?;
    let shape = serde_json::from_reader::<_, Shape>(std::io::BufReader::new(&mut *file)).ok();
    file.rewind()?;
    let has = |keys: &[&str]| {
        shape
            .as_ref()
            .is_some_and(|s| keys.iter().all(|k| s.keys.contains(*k)))
    };
    let name = if has(&["mapping", "current_node"]) && (has(&["conversation_id"]) || has(&["id"])) {
        Some("chatgpt")
    } else if has(&["uuid", "chat_messages"]) {
        Some("claude-ai")
    } else if has(&["chunkedPrompt"]) {
        Some("ai-studio")
    } else if has(&["resourceSpans"]) {
        Some("otel-genai")
    } else if has(&["version", "messages"])
        && (has(&["id"])
            || has(&["characters"])
            || path
                .file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with("OpenRouter Chat")))
    {
        Some("openrouter")
    } else if path
        .file_name()
        .is_some_and(|n| n == "prod-grok-backend.json")
        && has(&["conversations"])
    {
        Some("grok")
    } else {
        None
    };
    if let Some(name) = name {
        return Ok(polylogue_sources::origin::by_name(name));
    }
    for name in ["browser-capture", "hook-spool", "hermes", "claude-design"] {
        if let Some(origin) = polylogue_sources::origin::by_name(name) {
            if (origin.recognize)(path, &head[..n]) {
                return Ok(Some(origin));
            }
        }
    }
    // Outside a tool's root its preserved layout is the declaration.
    Ok(ORIGINS
        .iter()
        .filter(|o| o.arrival != Arrival::Import && o.support == Support::Parsed)
        .find(|o| {
            o.locations.iter().any(|p| collect::ends_in(path, p)) && (o.recognize)(path, &head[..n])
        }))
}

pub(super) fn refuse(report: &mut ImportReport, path: &Path, reason: impl Into<String>) {
    report.refused.push(ImportRefusal {
        path: path.to_path_buf(),
        reason: reason.into(),
        structures_tried: STRUCTURES.iter().map(|s| (*s).into()).collect(),
    });
}
